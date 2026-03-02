from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.utils.response import ResponseWrapper as R
import os
import sys
import signal

from app.services.cookie_manager import CookieConfigManager
from ffmpeg_helper import ensure_ffmpeg_or_raise

router = APIRouter()
cookie_manager = CookieConfigManager()


class CookieUpdateRequest(BaseModel):
    platform: str
    cookie: str


@router.get("/get_downloader_cookie/{platform}")
def get_cookie(platform: str):
    cookie = cookie_manager.get(platform)
    if not cookie:
        return R.success(msg='未找到Cookies')
    return R.success(
        data={"platform": platform, "cookie": cookie}
    )


@router.post("/update_downloader_cookie")
def update_cookie(data: CookieUpdateRequest):
    cookie_manager.set(data.platform, data.cookie)
    return R.success(

    )

@router.get("/sys_health")
async def sys_health():
    try:
        ensure_ffmpeg_or_raise()
        return R.success()
    except EnvironmentError:
        return R.error(msg="系统未安装 ffmpeg 请先进行安装")

@router.get("/sys_check")
async def sys_check():
    return R.success()

@router.post("/restart")
async def restart_server():
    """重启服务器"""
    import threading
    import time
    import subprocess
    import platform
    
    def do_restart():
        # 等待响应发送完成
        time.sleep(1)
        
        # Windows 系统使用不同的方法
        if platform.system() == "Windows":
            # 在 Windows 上，直接退出进程
            # 由于是通过 npm run dev 启动的，进程退出后 concurrently 会检测到
            # 但不会自动重启，所以我们需要使用 os._exit 强制退出
            os._exit(0)
        else:
            # Unix 系统使用 SIGTERM
            os.kill(os.getpid(), signal.SIGTERM)
    
    # 在后台线程中执行重启
    thread = threading.Thread(target=do_restart)
    thread.daemon = True
    thread.start()
    
    return R.success(msg="服务器正在重启...")