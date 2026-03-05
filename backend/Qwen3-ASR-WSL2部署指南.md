# Qwen3-ASR WSL2 部署指南（推荐方案）

## 为什么选择 WSL2？

✅ **比虚拟机更简单**：无需单独的虚拟机软件
✅ **完全支持 CUDA**：NVIDIA 官方支持
✅ **性能接近原生**：比传统虚拟机快得多
✅ **无缝集成 Windows**：可以直接访问 Windows 文件系统
✅ **资源占用更少**：比虚拟机更轻量

## 系统要求

- **Windows 11** 或 **Windows 10 (版本 2004 及更高版本)**
- **NVIDIA GPU**（Pascal 架构及更新，如 GTX 1060 及以上）
- **16GB 内存**（推荐 32GB）
- **50GB 可用磁盘空间**

## 第一步：安装 WSL2

### 1. 启用 WSL2
打开 PowerShell（管理员权限）：

```powershell
# 安装 WSL2
wsl --install

# 更新到最新版本
wsl --update

# 设置 WSL2 为默认版本
wsl --set-default-version 2
```

### 2. 安装 Ubuntu
```powershell
# 安装 Ubuntu 22.04（推荐）
wsl --install -d Ubuntu-22.04

# 或者从 Microsoft Store 安装
# 搜索 "Ubuntu 22.04 LTS" 并安装
```

### 3. 重启电脑
安装完成后重启电脑。

### 4. 首次启动 Ubuntu
```powershell
# 启动 Ubuntu
wsl -d Ubuntu-22.04

# 设置用户名和密码（首次启动会提示）
```

## 第二步：安装 NVIDIA 驱动（Windows 端）

### 1. 下载并安装 NVIDIA 驱动
访问：https://www.nvidia.com/Download/index.aspx

选择你的显卡型号，下载并安装最新的 **Game Ready 驱动** 或 **Studio 驱动**。

**重要**：
- ✅ 只需在 Windows 上安装驱动
- ❌ 不要在 WSL2 内安装任何 NVIDIA 驱动

### 2. 验证驱动安装
在 WSL2 中运行：
```bash
nvidia-smi
```

如果看到 GPU 信息，说明驱动安装成功！

## 第三步：在 WSL2 中安装环境

### 1. 更新系统
```bash
sudo apt update
sudo apt upgrade -y
```

### 2. 安装 Python 3.12
```bash
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev python3-pip
```

### 3. 安装 CUDA Toolkit（WSL-Ubuntu 版本）
```bash
# 移除旧的 GPG key（如果有）
sudo apt-key del 7fa2af80

# 下载并安装 CUDA Toolkit（WSL-Ubuntu 专用版本）
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# 安装 CUDA Toolkit（不包含驱动）
sudo apt install -y cuda-toolkit-12-1

# 设置环境变量
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 4. 验证 CUDA 安装
```bash
nvcc --version
nvidia-smi
```

### 5. 创建项目目录
```bash
cd ~
mkdir qwen3-asr-service
cd qwen3-asr-service
```

### 6. 创建虚拟环境
```bash
python3.12 -m venv venv
source venv/bin/activate
```

### 7. 安装 Python 依赖
```bash
# 升级 pip
pip install --upgrade pip

# 安装 PyTorch（CUDA 12.1）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 vLLM
pip install vllm

# 安装 qwen-asr 和 modelscope
pip install qwen-asr[vllm]
pip install modelscope

# 安装服务依赖
pip install flask flask-cors soundfile librosa numpy

# 设置环境变量（强制使用 ModelScope）
echo 'export MODELSCOPE_CACHE=~/.cache/modelscope' >> ~/.bashrc
echo 'export MODELSCOPE_MODULES_CACHE=~/.cache/modelscope/hub' >> ~/.bashrc
source ~/.bashrc
```

### 8. 验证安装
```bash
# 验证 PyTorch + CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"

# 验证 vLLM
python -c "import vllm; print('vLLM OK')"

# 验证 qwen-asr
python -c "from qwen_asr import Qwen3ASRModel; print('qwen-asr OK')"
```

## 第四步：部署服务

### 1. 复制服务文件到 WSL2
在 Windows PowerShell 中：
```powershell
# 复制服务脚本
wsl cp /mnt/c/Home/demo/BiliNote/backend/qwen3_asr_server.py ~/qwen3-asr-service/
```

或者在 WSL2 中直接访问 Windows 文件：
```bash
# Windows 的 C:\ 盘映射到 /mnt/c/
cp /mnt/c/Home/demo/BiliNote/backend/qwen3_asr_server.py ~/qwen3-asr-service/
```

### 2. 启动服务
```bash
cd ~/qwen3-asr-service
source venv/bin/activate
python qwen3_asr_server.py
```

### 3. 获取 WSL2 IP 地址
在 WSL2 中运行：
```bash
hostname -I
# 或
ip addr show eth0 | grep inet
```

记下这个 IP 地址，例如：`172.x.x.x`

## 第五步：Windows 端配置

### 1. 更新 .env 配置
编辑 `backend/.env`：
```bash
TRANSCRIBER_TYPE=qwen3-asr-remote
QWEN3_ASR_REMOTE_URL=http://WSL2的IP:8765
```

### 2. 测试连接
在 Windows PowerShell 中：
```powershell
curl http://WSL2的IP:8765/health
```

## WSL2 特有优势

### 1. 访问 Windows 文件
在 WSL2 中可以直接访问 Windows 文件：
```bash
# Windows C:\ 盘
cd /mnt/c/

# 你的项目目录
cd /mnt/c/Home/demo/BiliNote/backend/
```

### 2. 在 Windows 中访问 WSL2 文件
在 Windows 文件资源管理器中输入：
```
\\wsl$\Ubuntu-22.04\home\你的用户名\qwen3-asr-service
```

### 3. 使用 Windows Terminal
推荐使用 Windows Terminal 管理 WSL2：
- 从 Microsoft Store 安装 "Windows Terminal"
- 可以同时打开多个 WSL2 标签页

## 性能优化

### 1. 使用 ModelScope 下载模型（国内更快）
```bash
# 在 WSL2 中
export MODELSCOPE_CACHE=~/.cache/modelscope
pip install modelscope

# 预下载模型
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-ASR-1.7B')"
```

### 2. 调整 WSL2 内存限制
创建 `C:\Users\你的用户名\.wslconfig`：
```ini
[wsl2]
memory=24GB  # 限制 WSL2 最大内存
processors=8  # 限制 CPU 核心数
swap=8GB  # 交换空间
```

重启 WSL2 使配置生效：
```powershell
wsl --shutdown
wsl
```

### 3. 启用 systemd（可选）
编辑 `/etc/wsl.conf`：
```bash
sudo nano /etc/wsl.conf
```

添加：
```ini
[boot]
systemd=true
```

重启 WSL2：
```powershell
wsl --shutdown
wsl
```

## 开机自启动（可选）

### 方法 1: Windows 任务计划程序
1. 打开"任务计划程序"
2. 创建基本任务
3. 触发器：登录时
4. 操作：启动程序
   - 程序：`wsl`
   - 参数：`-d Ubuntu-22.04 -u 你的用户名 -- bash -c "cd ~/qwen3-asr-service && source venv/bin/activate && nohup python qwen3_asr_server.py > server.log 2>&1 &"`

### 方法 2: systemd 服务（如果启用了 systemd）
```bash
sudo nano /etc/systemd/system/qwen3-asr.service
```

内容：
```ini
[Unit]
Description=Qwen3-ASR Service
After=network.target

[Service]
Type=simple
User=你的用户名
WorkingDirectory=/home/你的用户名/qwen3-asr-service
Environment="PATH=/home/你的用户名/qwen3-asr-service/venv/bin"
ExecStart=/home/你的用户名/qwen3-asr-service/venv/bin/python qwen3_asr_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

启用：
```bash
sudo systemctl daemon-reload
sudo systemctl enable qwen3-asr
sudo systemctl start qwen3-asr
```

## 故障排查

### 问题 1: nvidia-smi 不可用
```bash
# 检查 Windows 驱动版本
# 在 Windows PowerShell 中运行
nvidia-smi

# 确保驱动版本 >= 495
# 如果版本过低，更新 Windows 驱动
```

### 问题 2: WSL2 无法访问 GPU
```bash
# 检查 WSL 版本
wsl --version

# 更新 WSL
wsl --update

# 重启 WSL
wsl --shutdown
wsl
```

### 问题 3: 端口无法访问
```bash
# 在 WSL2 中检查服务是否运行
ps aux | grep python

# 检查端口监听
netstat -tulpn | grep 8765

# 在 Windows 中添加端口转发（如果需要）
netsh interface portproxy add v4tov4 listenport=8765 listenaddress=0.0.0.0 connectport=8765 connectaddress=WSL2的IP
```

### 问题 4: 内存不足
```bash
# 检查内存使用
free -h

# 调整 .wslconfig 增加内存限制
# 见上面的"性能优化"部分
```

## 监控和管理

### 查看服务日志
```bash
cd ~/qwen3-asr-service
tail -f server.log
```

### 监控 GPU 使用
```bash
watch -n 1 nvidia-smi
```

### 停止服务
```bash
# 找到进程 ID
ps aux | grep qwen3_asr_server

# 杀死进程
kill -9 进程ID
```

### 重启 WSL2
```powershell
# 在 Windows PowerShell 中
wsl --shutdown
wsl
```

## 与虚拟机方案对比

| 特性 | WSL2 | 虚拟机 |
|------|------|--------|
| 安装难度 | ⭐⭐ 简单 | ⭐⭐⭐⭐ 复杂 |
| 性能 | ⭐⭐⭐⭐⭐ 接近原生 | ⭐⭐⭐ 较慢 |
| 资源占用 | ⭐⭐⭐⭐⭐ 很少 | ⭐⭐ 较多 |
| CUDA 支持 | ✅ 官方支持 | ✅ 支持 |
| 文件共享 | ✅ 无缝 | ⚠️ 需要配置 |
| 网络配置 | ✅ 简单 | ⚠️ 需要配置 |
| 启动速度 | ⭐⭐⭐⭐⭐ 秒级 | ⭐⭐ 分钟级 |

## 总结

**WSL2 是在 Windows 上运行 Qwen3-ASR + vLLM 的最佳方案！**

优势：
- ✅ 安装简单，无需虚拟机软件
- ✅ 性能接近原生 Linux
- ✅ 完美支持 CUDA
- ✅ 与 Windows 无缝集成
- ✅ 资源占用少

现在你可以开始部署了！按照上面的步骤一步步来，有问题随时问我。
