# 进入后端目录
cd backend

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
.\venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 安装deepgram插件
pip install deepgram-sdk

# 安装PyTorch CUDA（可选）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 进入前端目录
cd ..\BillNote_frontend

# 安装前端依赖
npm install

# 回到项目根目录
cd ..

# 启动项目（同时启动前后端）
npm run dev