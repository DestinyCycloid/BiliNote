import os
from abc import ABC
from typing import Union, Optional, List
import logging

import yt_dlp

from app.downloaders.base import Downloader, DownloadQuality, QUALITY_MAP
from app.models.notes_model import AudioDownloadResult
from app.utils.path_helper import get_data_dir
from app.utils.url_parser import extract_video_id

logger = logging.getLogger(__name__)


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
            如果 process_playlist=True 且是合集: 返回 List[AudioDownloadResult]（只包含基本信息，不下载）
        """
        if output_dir is None:
            output_dir = get_data_dir()
        if not output_dir:
            output_dir=self.cache_data
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "%(id)s.%(ext)s")

        # 如果是合集处理，只获取URL列表，不下载
        if process_playlist:
            return self._get_playlist_urls(video_url, output_dir)
        
        # 单个视频处理（原有逻辑）
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
            'noplaylist': True,
            'quiet': False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            
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
    
    def _get_playlist_urls(
        self,
        video_url: str,
        output_dir: str,
    ) -> List[AudioDownloadResult]:
        """
        快速获取合集中所有视频的URL（使用 extract_flat）
        """
        ydl_opts = {
            'extract_flat': True,  # 只提取URL，不获取详细信息
            'quiet': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            
            if 'entries' not in info:
                # 不是合集
                return [self._create_placeholder_result(info, output_dir)]
            
            # 是合集，返回占位符列表
            results = []
            for idx, entry in enumerate(info['entries'], 1):
                if entry:
                    results.append(self._create_placeholder_result(entry, output_dir, idx))
            
            return results
    
    def _create_placeholder_result(self, info: dict, output_dir: str, idx: int = 1) -> AudioDownloadResult:
        """创建占位符结果（只有URL，没有详细信息）"""
        # extract_flat 模式下，只有基本信息
        raw_id = info.get("id") or info.get("url", "").split("/")[-1]
        
        # 清理 video_id：移除 URL 参数（如 ?p=1）
        video_id = raw_id.split("?")[0] if raw_id else f"video_{idx}"
        
        # 如果有分P参数，添加到文件名（使用下划线）
        if "?p=" in raw_id:
            p_num = raw_id.split("?p=")[1].split("&")[0]
            file_id = f"{video_id}_p{p_num}"
        else:
            file_id = video_id
        
        title = info.get("title", f"视频{idx}")
        url = info.get("url") or info.get("webpage_url") or f"https://www.bilibili.com/video/{raw_id}"
        
        return AudioDownloadResult(
            file_path=os.path.join(output_dir, f"{file_id}.mp3"),
            title=title,
            duration=0,
            cover_url=None,
            platform="bilibili",
            video_id=video_id,
            raw_info={"url": url, "webpage_url": url, "original_id": raw_id},  # 保存原始ID和URL
            video_path=None
        )
    
    def download_single_audio(
        self,
        audio_result: AudioDownloadResult,
        output_dir: str,
        quality: DownloadQuality = "fast"
    ) -> tuple[str, str]:
        """
        下载单个视频的音频（用于并行下载）
        
        Args:
            audio_result: 音频结果对象（包含完整的视频信息）
            output_dir: 输出目录
            quality: 音频质量
            
        Returns:
            (音频文件路径, 真实标题)
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 从 raw_info 获取视频URL
        video_url = audio_result.raw_info.get('webpage_url') or audio_result.raw_info.get('url')
        if not video_url:
            # 如果没有URL，使用 video_id 构造
            original_id = audio_result.raw_info.get('original_id', audio_result.video_id)
            video_url = f"https://www.bilibili.com/video/{original_id}"
        
        logger.info(f"开始下载音频: {audio_result.title} ({video_url})")
        
        # 构造输出路径模板
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
            'noplaylist': True,
            'quiet': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_id = info.get("id")
            real_title = info.get("title", audio_result.title)  # 获取真实标题
            
            # 处理分P视频的文件名
            if "?p=" in video_id:
                p_num = video_id.split("?p=")[1].split("&")[0]
                clean_id = video_id.split("?")[0]
                audio_path = os.path.join(output_dir, f"{clean_id}_p{p_num}.mp3")
                # 重命名文件
                original_path = os.path.join(output_dir, f"{video_id}.mp3")
                if os.path.exists(original_path):
                    os.rename(original_path, audio_path)
            else:
                audio_path = os.path.join(output_dir, f"{video_id}.mp3")
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件下载失败: {audio_path}")
        
        logger.info(f"音频下载完成: {audio_path}, 标题: {real_title}")
        return audio_path, real_title

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