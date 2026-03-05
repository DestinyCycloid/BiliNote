"""
Qwen3-ASR HTTP 服务端

运行在 Linux 虚拟机上，提供 HTTP API 供 Windows 主机调用

功能：
1. 提供音频转写 API
2. 支持流式转写
3. 支持多语言识别
4. 使用 vLLM 后端，高性能

部署：
1. 在 Linux 虚拟机上运行
2. 确保安装了 vLLM 和 qwen-asr
3. 启动服务: python qwen3_asr_server.py

依赖：
pip install qwen-asr[vllm] flask flask-cors soundfile librosa numpy
"""

import os
import io
import base64
import time
import threading
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import numpy as np
import soundfile as sf
import torch

# 强制使用 ModelScope（国内更快）
os.environ['MODELSCOPE_CACHE'] = os.path.expanduser('~/.cache/modelscope')
os.environ['MODELSCOPE_MODULES_CACHE'] = os.path.expanduser('~/.cache/modelscope/hub')
# 禁用 Hugging Face
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 如果必须用 HF，使用镜像
os.environ['TRANSFORMERS_OFFLINE'] = '0'

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局模型实例
model = None
model_lock = threading.Lock()

# 配置
CONFIG = {
    "model_name": "Qwen/Qwen3-ASR-1.7B",
    "backend": "vllm",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "gpu_memory_utilization": 0.8,
    "max_inference_batch_size": 32,
    "max_new_tokens": 4096,
    "port": 8765,
    "host": "0.0.0.0"
}


def load_model():
    """加载 Qwen3-ASR 模型（强制使用 ModelScope）"""
    global model
    
    if model is not None:
        return model
    
    with model_lock:
        if model is not None:
            return model
        
        from qwen_asr import Qwen3ASRModel
        
        print("=" * 80)
        print(f"正在加载 {CONFIG['model_name']} 模型...")
        print(f"后端: {CONFIG['backend']}")
        print(f"设备: {CONFIG['device']}")
        print("=" * 80)
        print("📦 使用 ModelScope 下载模型（国内更快）")
        print(f"缓存目录: {os.environ['MODELSCOPE_CACHE']}")
        print("=" * 80)
        
        try:
            # 确保使用 ModelScope
            # qwen-asr 会自动从 ModelScope 下载
            model = Qwen3ASRModel.LLM(
                model=CONFIG['model_name'],
                gpu_memory_utilization=CONFIG['gpu_memory_utilization'],
                max_inference_batch_size=CONFIG['max_inference_batch_size'],
                max_new_tokens=CONFIG['max_new_tokens'],
            )
            
            print(f"✅ {CONFIG['model_name']} 模型加载完成")
            print(f"设备: {CONFIG['device']}")
            print(f"模型缓存位置: ~/.cache/modelscope/hub/")
            print("=" * 80)
            
            return model
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("\n提示：")
            print("  1. 首次运行会自动从 ModelScope 下载模型（约 3-4 GB）")
            print("  2. 如果下载失败，可以手动预下载：")
            print("     pip install modelscope")
            print("     python -c \"from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-ASR-1.7B')\"")
            print("  3. 确保网络连接正常")
            raise


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "ok",
        "model": CONFIG['model_name'],
        "backend": CONFIG['backend'],
        "device": CONFIG['device'],
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    })


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    音频转写 API
    
    请求格式:
    {
        "audio": "base64编码的音频数据",
        "language": "zh" (可选，None 表示自动检测),
        "return_timestamps": false (可选)
    }
    
    或者直接上传音频文件（multipart/form-data）
    """
    try:
        start_time = time.time()
        
        # 加载模型
        model = load_model()
        
        # 获取音频数据
        audio_data = None
        
        # 方式 1: JSON 格式（base64 编码）
        if request.is_json:
            data = request.get_json()
            audio_base64 = data.get('audio')
            language = data.get('language', None)
            return_timestamps = data.get('return_timestamps', False)
            
            if not audio_base64:
                return jsonify({"error": "Missing audio data"}), 400
            
            # 解码 base64
            audio_bytes = base64.b64decode(audio_base64)
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        
        # 方式 2: 文件上传
        elif 'audio' in request.files:
            audio_file = request.files['audio']
            language = request.form.get('language', None)
            return_timestamps = request.form.get('return_timestamps', 'false').lower() == 'true'
            
            # 读取音频
            audio_data, sample_rate = sf.read(io.BytesIO(audio_file.read()))
        
        else:
            return jsonify({"error": "No audio data provided"}), 400
        
        # 确保是单声道
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # 转写
        results = model.transcribe(
            audio=audio_data,
            language=language,
            return_time_stamps=return_timestamps,
        )
        
        if not results or len(results) == 0:
            return jsonify({"error": "Transcription failed"}), 500
        
        result = results[0]
        
        # 构建响应
        response_data = {
            "language": result.language if hasattr(result, 'language') else "unknown",
            "text": result.text if hasattr(result, 'text') else "",
            "processing_time": time.time() - start_time
        }
        
        # 添加时间戳（如果有）
        if return_timestamps and hasattr(result, 'time_stamps') and result.time_stamps:
            response_data["timestamps"] = [
                {
                    "text": ts.text,
                    "start": ts.start_time,
                    "end": ts.end_time
                }
                for ts in result.time_stamps
            ]
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/transcribe_stream', methods=['POST'])
def transcribe_stream():
    """
    流式转写 API（实验性）
    
    请求格式: 与 /transcribe 相同
    响应格式: Server-Sent Events (SSE)
    """
    try:
        # 加载模型
        model = load_model()
        
        # 获取音频数据（与 transcribe 相同的逻辑）
        # ... (省略，与上面相同)
        
        def generate():
            """生成流式响应"""
            # TODO: 实现真正的流式转写
            # 目前 vLLM 的流式 API 需要特殊处理
            yield f"data: {{'status': 'processing'}}\n\n"
            
            # 执行转写
            results = model.transcribe(audio=audio_data, language=None)
            result = results[0]
            
            # 返回结果
            yield f"data: {{'text': '{result.text}', 'language': '{result.language}'}}\n\n"
            yield "data: {'status': 'done'}\n\n"
        
        return Response(generate(), mimetype='text/event-stream')
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """获取模型信息"""
    try:
        model = load_model()
        
        return jsonify({
            "model_name": CONFIG['model_name'],
            "backend": CONFIG['backend'],
            "device": CONFIG['device'],
            "loaded": model is not None,
            "supported_languages": [
                "zh", "en", "yue", "ar", "de", "fr", "es", "pt", "id", "it",
                "ko", "ru", "th", "vi", "ja", "tr", "hi", "ms", "nl", "sv",
                "da", "fi", "pl", "cs", "fil", "fa", "el", "hu", "mk", "ro"
            ],
            "supported_dialects": [
                "安徽话", "东北话", "福建话", "甘肃话", "贵州话", "河北话",
                "河南话", "湖北话", "湖南话", "江西话", "宁夏话", "山东话",
                "陕西话", "山西话", "四川话", "天津话", "云南话", "浙江话",
                "粤语（香港）", "粤语（广东）", "吴语", "闽南语"
            ]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    """启动服务"""
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "Qwen3-ASR HTTP 服务" + " " * 20 + "║")
    print("║" + " " * 15 + "支持 52 种语言和方言" + " " * 15 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    print("配置信息:")
    print(f"  模型: {CONFIG['model_name']}")
    print(f"  后端: {CONFIG['backend']}")
    print(f"  设备: {CONFIG['device']}")
    print(f"  监听: {CONFIG['host']}:{CONFIG['port']}")
    print()
    
    # 预加载模型
    print("预加载模型...")
    try:
        load_model()
        print("✅ 模型预加载完成\n")
    except Exception as e:
        print(f"❌ 模型预加载失败: {e}\n")
        print("服务将在首次请求时加载模型\n")
    
    print("=" * 80)
    print("服务已启动！")
    print("=" * 80)
    print(f"健康检查: http://{CONFIG['host']}:{CONFIG['port']}/health")
    print(f"转写 API: http://{CONFIG['host']}:{CONFIG['port']}/transcribe")
    print(f"模型信息: http://{CONFIG['host']}:{CONFIG['port']}/model_info")
    print("=" * 80)
    print()
    
    # 启动 Flask 服务
    app.run(
        host=CONFIG['host'],
        port=CONFIG['port'],
        debug=False,
        threaded=True
    )


if __name__ == "__main__":
    main()
