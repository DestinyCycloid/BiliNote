# app/routers/note.py
import json
import os
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, validator, field_validator
from dataclasses import asdict

from app.db.video_task_dao import get_task_by_video
from app.enmus.exception import NoteErrorEnum
from app.enmus.note_enums import DownloadQuality
from app.exceptions.note import NoteError
from app.services.note import NoteGenerator, logger
from app.utils.response import ResponseWrapper as R
from app.utils.url_parser import extract_video_id
from app.validators.video_url_validator import is_supported_video_url
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
import httpx
from app.enmus.task_status_enums import TaskStatus

# from app.services.downloader import download_raw_audio
# from app.services.whisperer import transcribe_audio

router = APIRouter()


@router.get("/transcribers")
def get_available_transcribers():
    """获取可用的转写器列表"""
    from app.transcriber.transcriber_provider import TranscriberType
    import os
    
    transcribers = [
        {
            "value": TranscriberType.FAST_WHISPER.value,
            "label": "Faster Whisper",
            "description": "快速、准确，支持多种语言"
        },
        {
            "value": TranscriberType.PARAFORMER_STREAMING.value,
            "label": "Paraformer-streaming",
            "description": "流式转写，中文优化，低延迟（推荐）"
        },
        {
            "value": TranscriberType.FUNASR_NANO.value,
            "label": "Fun-ASR-Nano",
            "description": "轻量级，支持31种语言"
        },
        {
            "value": TranscriberType.DEEPGRAM.value,
            "label": "Deepgram",
            "description": "云端API，需要API Key"
        },
        {
            "value": TranscriberType.GROQ.value,
            "label": "Groq",
            "description": "云端API，需要API Key"
        },
        {
            "value": TranscriberType.BCUT.value,
            "label": "必剪",
            "description": "B站官方转写"
        },
        {
            "value": TranscriberType.KUAISHOU.value,
            "label": "快手",
            "description": "快手官方转写"
        },
    ]
    
    # 如果是 Apple 平台，添加 MLX Whisper
    import platform
    if platform.system() == "Darwin":
        transcribers.insert(1, {
            "value": TranscriberType.MLX_WHISPER.value,
            "label": "MLX Whisper",
            "description": "Apple 芯片优化版本"
        })
    
    # 获取当前配置
    current_transcriber = os.getenv("TRANSCRIBER_TYPE", "fast-whisper")
    
    return R.success({
        "transcribers": transcribers,
        "current": current_transcriber
    })


@router.post("/transcriber/set")
def set_transcriber(data: dict):
    """设置当前使用的转写器"""
    transcriber_type = data.get("transcriber_type")
    
    if not transcriber_type:
        return R.error("请提供 transcriber_type 参数")
    
    # 更新 .env 文件
    from pathlib import Path
    env_file = Path(".env")
    
    if env_file.exists():
        lines = env_file.read_text(encoding="utf-8").splitlines()
        updated = False
        
        for i, line in enumerate(lines):
            if line.startswith("TRANSCRIBER_TYPE="):
                lines[i] = f"TRANSCRIBER_TYPE={transcriber_type}"
                updated = True
                break
        
        if not updated:
            lines.append(f"TRANSCRIBER_TYPE={transcriber_type}")
        
        env_file.write_text("\n".join(lines), encoding="utf-8")
        
        # 更新环境变量
        os.environ["TRANSCRIBER_TYPE"] = transcriber_type
        
        return R.success({"message": f"转写器已切换到 {transcriber_type}，重启后生效"})
    else:
        return R.error(".env 文件不存在")


class RecordRequest(BaseModel):
    video_id: str
    platform: str


class VideoRequest(BaseModel):
    video_url: str
    platform: str
    quality: DownloadQuality
    screenshot: Optional[bool] = False
    link: Optional[bool] = False
    model_name: str
    provider_id: str
    task_id: Optional[str] = None
    format: Optional[list] = []
    style: str = None
    extras: Optional[str]=None
    video_understanding: Optional[bool] = False
    video_interval: Optional[int] = 0
    grid_size: Optional[list] = []
    process_playlist: Optional[bool] = False  # 是否处理合集
    playlist_serial_mode: Optional[bool] = False  # 合集串行模式

    @field_validator("video_url")
    def validate_supported_url(cls, v):
        url = str(v)
        parsed = urlparse(url)
        if parsed.scheme in ("http", "https"):
            # 是网络链接，继续用原有平台校验
            if not is_supported_video_url(url):
                raise NoteError(code=NoteErrorEnum.PLATFORM_NOT_SUPPORTED.code,
                                message=NoteErrorEnum.PLATFORM_NOT_SUPPORTED.message)

        return v


NOTE_OUTPUT_DIR = os.getenv("NOTE_OUTPUT_DIR", "note_results")
UPLOAD_DIR = "uploads"


def save_note_to_file(task_id: str, note):
    os.makedirs(NOTE_OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(NOTE_OUTPUT_DIR, f"{task_id}.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(note), f, ensure_ascii=False, indent=2)


def run_note_task(task_id: str, video_url: str, platform: str, quality: DownloadQuality,
                  link: bool = False, screenshot: bool = False, model_name: str = None, provider_id: str = None,
                  _format: list = None, style: str = None, extras: str = None, video_understanding: bool = False,
                  video_interval=0, grid_size=[], process_playlist: bool = False, playlist_serial_mode: bool = False
                  ):

    if not model_name or not provider_id:
        raise HTTPException(status_code=400, detail="请选择模型和提供者")

    note = NoteGenerator().generate(
        video_url=video_url,
        platform=platform,
        quality=quality,
        task_id=task_id,
        model_name=model_name,
        provider_id=provider_id,
        link=link,
        _format=_format,
        style=style,
        extras=extras,
        screenshot=screenshot,
        video_understanding=video_understanding,
        video_interval=video_interval,
        grid_size=grid_size,
        process_playlist=process_playlist,
        playlist_serial_mode=playlist_serial_mode,
    )
    logger.info(f"Note generated: {task_id}")
    if not note or not note.markdown:
        logger.warning(f"任务 {task_id} 执行失败，跳过保存")
        return
    save_note_to_file(task_id, note)



@router.post('/delete_task')
def delete_task(data: RecordRequest):
    try:
        # TODO: 待持久化完成
        # NoteGenerator().delete_note(video_id=data.video_id, platform=data.platform)
        return R.success(msg='删除成功')
    except Exception as e:
        return R.error(msg=e)


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_location = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_location, "wb+") as f:
        f.write(await file.read())

    # 假设你静态目录挂载了 /uploads
    return R.success({"url": f"/uploads/{file.filename}"})


@router.post("/generate_note")
def generate_note(data: VideoRequest, background_tasks: BackgroundTasks):
    try:
        video_id = extract_video_id(data.video_url, data.platform)
        # if not video_id:
        #     raise HTTPException(status_code=400, detail="无法提取视频 ID")
        # existing = get_task_by_video(video_id, data.platform)
        # if existing:
        #     return R.error(
        #         msg='笔记已生成，请勿重复发起',
        #
        #     )
        if data.task_id:
            # 如果传了task_id，说明是重试！
            task_id = data.task_id
            # 更新之前的状态
            NoteGenerator()._update_status(task_id, TaskStatus.PENDING)
            logger.info(f"重试模式，复用已有 task_id={task_id}")
        else:
            # 正常新建任务
            task_id = str(uuid.uuid4())

        background_tasks.add_task(
            run_note_task,
            task_id=task_id,
            video_url=data.video_url,
            platform=data.platform,
            quality=data.quality,
            link=data.link,
            screenshot=data.screenshot,
            model_name=data.model_name,
            provider_id=data.provider_id,
            _format=data.format,
            style=data.style,
            extras=data.extras,
            video_understanding=data.video_understanding,
            video_interval=data.video_interval,
            grid_size=data.grid_size,
            process_playlist=data.process_playlist,
            playlist_serial_mode=data.playlist_serial_mode,
        )
        return R.success({"task_id": task_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/task_status/{task_id}")
def get_task_status(task_id: str):
    status_path = os.path.join(NOTE_OUTPUT_DIR, f"{task_id}.status.json")
    result_path = os.path.join(NOTE_OUTPUT_DIR, f"{task_id}.json")

    # 优先读状态文件
    if os.path.exists(status_path):
        with open(status_path, "r", encoding="utf-8") as f:
            status_content = json.load(f)

        status = status_content.get("status")
        message = status_content.get("message", "")

        if status == TaskStatus.SUCCESS.value:
            # 成功状态的话，继续读取最终笔记内容
            if os.path.exists(result_path):
                with open(result_path, "r", encoding="utf-8") as rf:
                    result_content = json.load(rf)
                return R.success({
                    "status": status,
                    "result": result_content,
                    "message": message,
                    "task_id": task_id
                })
            else:
                # 理论上不会出现，保险处理
                return R.success({
                    "status": TaskStatus.PENDING.value,
                    "message": "任务完成，但结果文件未找到",
                    "task_id": task_id
                })

        if status == TaskStatus.FAILED.value:
            return R.error(message or "任务失败", code=500)

        # 处理中状态
        return R.success({
            "status": status,
            "message": message,
            "task_id": task_id
        })

    # 没有状态文件，但有结果
    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            result_content = json.load(f)
        return R.success({
            "status": TaskStatus.SUCCESS.value,
            "result": result_content,
            "task_id": task_id
        })

    # 什么都没有，默认PENDING
    return R.success({
        "status": TaskStatus.PENDING.value,
        "message": "任务排队中",
        "task_id": task_id
    })


@router.get("/image_proxy")
async def image_proxy(request: Request, url: str):
    headers = {
        "Referer": "https://www.bilibili.com/",
        "User-Agent": request.headers.get("User-Agent", ""),
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers)

            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail="图片获取失败")

            content_type = resp.headers.get("Content-Type", "image/jpeg")
            return StreamingResponse(
                resp.aiter_bytes(),
                media_type=content_type,
                headers={
                    "Cache-Control": "public, max-age=86400",  #  缓存一天
                    "Content-Type": content_type,
                }
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
