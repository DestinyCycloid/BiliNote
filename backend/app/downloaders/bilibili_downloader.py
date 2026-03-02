import os
from abc import ABC
from typing import Union, Optional, List

import yt_dlp

from app.downloaders.base import Downloader, DownloadQuality, QUALITY_MAP
from app.models.notes_model import AudioDownloadResult
from app.utils.path_helper import get_data_dir
from app.utils.url_parser import extract_video_id


class BilibiliDownloader(Downloader, ABC):
    def __init__(self):
        super().__init__()

    def download(
        self,
        video_url: str,
        output_dir: Union[str, None] = None,
        quality: DownloadQuality = "fast",
        need_video:Optional[bool]=False,
        process_playlist: bool = False
    ) -> Union[AudioDownloadResult, List[AudioDownloadResult]]:
        """
        下载 B 站视频音频
        
        Args:
            video_url: 视频链接
            output_dir: 输出目录
            quality: 音频质量
            need_video: 是否需要视频
            process_playlist: 是否处理合集（默认 False，只下载第一个）
            
        Returns:
            如果 process_playlist=False: 返回单个 AudioDownloadResult
            如果 process_playlist=True 且是合集: 返回 List[AudioDownloadResult]
        """
        if output_dir is None:
            output_dir = get_data_dir()
        if not output_dir:
            output_dir=self.cache_data
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "%(id)s.%(ext)s")

        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best',
            'outtmpl': output_path,
            'postprocessors': [
                {
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '64',
                }
            ],
            'noplaylist': not process_playlist,  # 根据参数决定是否处理合集
            'quiet': False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            
            # 检查是否为合集
            if process_playlist and 'entries' in info:
                # 是合集，返回列表
                results = []
                for entry in info['entries']:
                    if entry:  # 有些 entry 可能为 None
                        video_id = entry.get("id")
                        title = entry.get("title")
                        duration = entry.get("duration", 0)
                        cover_url = entry.get("thumbnail")
                        audio_path = os.path.join(output_dir, f"{video_id}.mp3")
                        
                        results.append(AudioDownloadResult(
                            file_path=audio_path,
                            title=title,
                            duration=duration,
                            cover_url=cover_url,
                            platform="bilibili",
                            video_id=video_id,
                            raw_info=entry,
                            video_path=None
                        ))
                return results
            else:
                # 单个视频
                video_id = info.get("id")
                title = info.get("title")
                duration = info.get("duration", 0)
                cover_url = info.get("thumbnail")
                audio_path = os.path.join(output_dir, f"{video_id}.mp3")

                return AudioDownloadResult(
                    file_path=audio_path,
                    title=title,
                    duration=duration,
                    cover_url=cover_url,
                    platform="bilibili",
                    video_id=video_id,
                    raw_info=info,
                    video_path=None
                )

    def download_video(
        self,
        video_url: str,
        output_dir: Union[str, None] = None,
    ) -> str:
        """
        下载视频，返回视频文件路径
        """

        if output_dir is None:
            output_dir = get_data_dir()
        os.makedirs(output_dir, exist_ok=True)
        print("video_url",video_url)
        video_id=extract_video_id(video_url, "bilibili")
        video_path = os.path.join(output_dir, f"{video_id}.mp4")
        if os.path.exists(video_path):
            return video_path

        # 检查是否已经存在


        output_path = os.path.join(output_dir, "%(id)s.%(ext)s")

        ydl_opts = {
            'format': 'bv*[ext=mp4]/bestvideo+bestaudio/best',
            'outtmpl': output_path,
            'noplaylist': True,
            'quiet': False,
            'merge_output_format': 'mp4',  # 确保合并成 mp4
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_id = info.get("id")
            video_path = os.path.join(output_dir, f"{video_id}.mp4")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件未找到: {video_path}")

        return video_path

    def delete_video(self, video_path: str) -> str:
        """
        删除视频文件
        """
        if os.path.exists(video_path):
            os.remove(video_path)
            return f"视频文件已删除: {video_path}"
        else:
            return f"视频文件未找到: {video_path}"