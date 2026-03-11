import base64
import os
import re
import subprocess
import ffmpeg
from PIL import Image, ImageDraw, ImageFont

from app.utils.logger import get_logger
from app.utils.path_helper import get_app_dir

logger = get_logger(__name__)
class VideoReader:
    def __init__(self,
                 video_path: str,
                 grid_size=(3, 3),
                 frame_interval=2,
                 unit_width=960,
                 unit_height=540,
                 save_quality=90,
                 font_path="fonts/arial.ttf",
                 frame_dir=None,
                 grid_dir=None):
        self.video_path = video_path
        self.grid_size = grid_size
        self.frame_interval = frame_interval
        self.unit_width = unit_width
        self.unit_height = unit_height
        self.save_quality = save_quality
        self.frame_dir = frame_dir or get_app_dir("output_frames")
        self.grid_dir = grid_dir or get_app_dir("grid_output")
        print(f"视频路径：{video_path}",self.frame_dir,self.grid_dir)
        self.font_path = font_path

    def format_time(self, seconds: float) -> str:
        mm = int(seconds // 60)
        ss = int(seconds % 60)
        return f"{mm:02d}_{ss:02d}"

    def extract_time_from_filename(self, filename: str) -> float:
        match = re.search(r"frame_(\d{2})_(\d{2})\.jpg", filename)
        if match:
            mm, ss = map(int, match.groups())
            return mm * 60 + ss
        return float('inf')

    def extract_frames(self, max_frames=1000) -> list[str]:

        try:
            os.makedirs(self.frame_dir, exist_ok=True)
            duration = float(ffmpeg.probe(self.video_path)["format"]["duration"])
            timestamps = [i for i in range(0, int(duration), self.frame_interval)][:max_frames]

            image_paths = []
            for ts in timestamps:
                time_label = self.format_time(ts)
                output_path = os.path.join(self.frame_dir, f"frame_{time_label}.jpg")
                cmd = ["ffmpeg", "-ss", str(ts), "-i", self.video_path, "-frames:v", "1", "-q:v", "2", "-y", output_path,
                       "-hide_banner", "-loglevel", "error"]
                subprocess.run(cmd, check=True)
                image_paths.append(output_path)
            return image_paths
        except Exception as e:
            logger.error(f"分割帧发生错误：{str(e)}")
            raise ValueError("视频处理失败")

    def group_images(self) -> list[list[str]]:
        image_files = [os.path.join(self.frame_dir, f) for f in os.listdir(self.frame_dir) if
                       f.startswith("frame_") and f.endswith(".jpg")]
        image_files.sort(key=lambda f: self.extract_time_from_filename(os.path.basename(f)))
        group_size = self.grid_size[0] * self.grid_size[1]
        return [image_files[i:i + group_size] for i in range(0, len(image_files), group_size)]

    def concat_images(self, image_paths: list[str], name: str) -> str:
        os.makedirs(self.grid_dir, exist_ok=True)
        font = ImageFont.truetype(self.font_path, 48) if os.path.exists(self.font_path) else ImageFont.load_default()
        images = []

        for path in image_paths:
            img = Image.open(path).convert("RGB").resize((self.unit_width, self.unit_height), Image.Resampling.LANCZOS)
            timestamp = re.search(r"frame_(\d{2})_(\d{2})\.jpg", os.path.basename(path))
            time_text = f"{timestamp.group(1)}:{timestamp.group(2)}" if timestamp else ""
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), time_text, fill="yellow", font=font, stroke_width=1, stroke_fill="black")
            images.append(img)

        cols, rows = self.grid_size
        grid_img = Image.new("RGB", (self.unit_width * cols, self.unit_height * rows), (255, 255, 255))

        for i, img in enumerate(images):
            x = (i % cols) * self.unit_width
            y = (i // cols) * self.unit_height
            grid_img.paste(img, (x, y))

        save_path = os.path.join(self.grid_dir, f"{name}.jpg")
        grid_img.save(save_path, quality=self.save_quality)
        return save_path

    def encode_images_to_base64(self, image_paths: list[str]) -> list[str]:
        base64_images = []
        for path in image_paths:
            with open(path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
                base64_images.append(f"data:image/jpeg;base64,{encoded_string}")
        return base64_images

    def run(self, max_grids=30)->list[str]:
        """
        执行视频帧提取和网格图生成
        
        :param max_grids: 最大网格图数量，默认 30 张（配合分批处理机制）
        :return: base64 编码的图片 URL 列表
        """
        logger.info("开始提取视频帧...")
        try:
            # 确保目录存在
            print(self.frame_dir,self.grid_dir)
            os.makedirs(self.frame_dir, exist_ok=True)
            os.makedirs(self.grid_dir, exist_ok=True)
            #清空帧文件夹
            for file in os.listdir(self.frame_dir):
                if file.startswith("frame_"):
                    os.remove(os.path.join(self.frame_dir, file))
            print(self.frame_dir,self.grid_dir)
            #清空网格文件夹
            for file in os.listdir(self.grid_dir):
                if file.startswith("grid_"):
                    os.remove(os.path.join(self.grid_dir, file))
            print(self.frame_dir,self.grid_dir)
            
            # 先提取所有需要的帧（根据视频长度和间隔，最多 1000 帧）
            logger.info(f"根据视频长度和间隔提取帧（最多 1000 帧）")
            self.extract_frames(max_frames=1000)
            
            print("2#3",self.frame_dir,self.grid_dir)
            logger.info("开始拼接网格图...")
            image_paths = []
            groups = self.group_images()
            
            total_groups = len(groups)
            logger.info(f"共提取 {total_groups * self.grid_size[0] * self.grid_size[1]} 帧，可生成 {total_groups} 个网格图")
            
            # 如果网格图数量超过限制，智能采样
            if total_groups > max_grids:
                logger.info(f"⚠️ 网格图数量 ({total_groups}) 超过限制 ({max_grids})，进行智能采样")
                # 均匀采样：从所有组中选择 max_grids 个
                step = total_groups / max_grids
                sampled_indices = [int(i * step) for i in range(max_grids)]
                groups = [groups[i] for i in sampled_indices if i < len(groups)]
                logger.info(f"✅ 采样后保留 {len(groups)} 个网格图，覆盖整个视频时长")
            
            for idx, group in enumerate(groups, start=1):
                if len(group) < self.grid_size[0] * self.grid_size[1]:
                    logger.warning(f"⚠️ 跳过第 {idx} 组，图片不足 {self.grid_size[0] * self.grid_size[1]} 张")
                    continue
                out_path = self.concat_images(group, f"grid_{idx}")
                image_paths.append(out_path)

            logger.info(f"📤 开始编码 {len(image_paths)} 张网格图...")
            urls = self.encode_images_to_base64(image_paths)
            logger.info(f"✅ 成功生成 {len(urls)} 张网格图")
            return urls
        except Exception as e:
            logger.error(f"发生错误：{str(e)}")
            raise ValueError("视频处理失败")


