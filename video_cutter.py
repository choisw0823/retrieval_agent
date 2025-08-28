#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Video Cutter - 영상에서 특정 구간을 잘라서 저장하는 도구

사용법:
    from video_cutter import VideoCutter
    
    cutter = VideoCutter()
    cutter.cut_segment("input.mp4", 10.5, 25.3, "output.mp4")
    cutter.cut_multiple_segments("input.mp4", [(10.5, 25.3), (45.2, 60.1)], "output_dir")
"""

import os
import subprocess
import json
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VideoSegment:
    """비디오 구간 정보"""
    start_time: float  # 시작 시간 (초)
    end_time: float    # 종료 시간 (초)
    duration: float    # 구간 길이 (초)
    output_path: str   # 출력 파일 경로

class VideoCutter:
    """영상 구간 자르기 도구"""
    
    def __init__(self, ffmpeg_path: str = "ffmpeg", output_dir: str = "cut_segments"):
        """
        Args:
            ffmpeg_path: ffmpeg 실행 파일 경로
            output_dir: 출력 디렉토리
        """
        self.ffmpeg_path = ffmpeg_path
        self.output_dir = output_dir
        self._ensure_output_dir()
        
        # ffmpeg 사용 가능 여부 확인
        if not self._check_ffmpeg():
            raise RuntimeError(f"ffmpeg not found at {ffmpeg_path}. Please install ffmpeg first.")
    
    def _ensure_output_dir(self):
        """출력 디렉토리 생성"""
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
    
    def _check_ffmpeg(self) -> bool:
        """ffmpeg 사용 가능 여부 확인"""
        try:
            result = subprocess.run([self.ffmpeg_path, "-version"], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"ffmpeg check failed: {e}")
            return False
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """비디오 파일 정보 가져오기"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            cmd = [
                self.ffmpeg_path, "-i", video_path,
                "-f", "null", "-"
            ]
            
            # ffprobe로 메타데이터 추출
            probe_cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", video_path
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise RuntimeError(f"ffprobe failed: {result.stderr}")
            
            info = json.loads(result.stdout)
            
            # 비디오 스트림 정보
            video_stream = None
            for stream in info.get("streams", []):
                if stream.get("codec_type") == "video":
                    video_stream = stream
                    break
            
            if not video_stream:
                raise RuntimeError("No video stream found")
            
            # 기본 정보
            format_info = info.get("format", {})
            duration = float(format_info.get("duration", 0))
            
            video_info = {
                "file_path": video_path,
                "duration": duration,
                "size_bytes": int(format_info.get("size", 0)),
                "bitrate": int(format_info.get("bit_rate", 0)),
                "format": format_info.get("format_name", "unknown"),
                "video_codec": video_stream.get("codec_name", "unknown"),
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "fps": eval(video_stream.get("r_frame_rate", "0/1")),
                "total_frames": int(video_stream.get("nb_frames", 0))
            }
            
            logger.info(f"Video info: {duration:.2f}s, {video_info['width']}x{video_info['height']}, {video_info['fps']:.2f} fps")
            return video_info
            
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            raise
    
    def cut_segment(self, input_path: str, start_time: float, end_time: float, 
                   output_path: Optional[str] = None, 
                   codec: str = "copy", quality: str = "high") -> str:
        """
        단일 구간 자르기
        
        Args:
            input_path: 입력 비디오 파일 경로
            start_time: 시작 시간 (초)
            end_time: 종료 시간 (초)
            output_path: 출력 파일 경로 (None이면 자동 생성)
            codec: 코덱 설정 ("copy", "h264", "h265")
            quality: 품질 설정 ("high", "medium", "low")
        
        Returns:
            출력 파일 경로
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        if start_time < 0 or end_time <= start_time:
            raise ValueError(f"Invalid time range: {start_time} to {end_time}")
        
        # 출력 파일명 자동 생성
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(
                self.output_dir, 
                f"{base_name}_cut_{start_time:.1f}_{end_time:.1f}.mp4"
            )
        
        # 출력 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # ffmpeg 명령어 구성
        duration = end_time - start_time
        
        if codec == "copy":
            # 빠른 복사 (재인코딩 없음)
            cmd = [
                self.ffmpeg_path, "-y",  # 기존 파일 덮어쓰기
                "-ss", str(start_time),   # 시작 시간
                "-i", input_path,         # 입력 파일
                "-t", str(duration),      # 구간 길이
                "-c", "copy",             # 코덱 복사
                "-avoid_negative_ts", "make_zero",  # 타임스탬프 정정
                output_path
            ]
        else:
            # 재인코딩 (품질 조정 가능)
            quality_settings = {
                "high": ["-crf", "18", "-preset", "slow"],
                "medium": ["-crf", "23", "-preset", "medium"],
                "low": ["-crf", "28", "-preset", "fast"]
            }
            
            cmd = [
                self.ffmpeg_path, "-y",
                "-ss", str(start_time),
                "-i", input_path,
                "-t", str(duration),
                "-c:v", "libx264" if codec == "h264" else "libx265",
                "-c:a", "aac",
                *quality_settings.get(quality, quality_settings["medium"]),
                output_path
            ]
        
        logger.info(f"Cutting segment: {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s)")
        logger.info(f"Output: {output_path}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"ffmpeg failed: {result.stderr}")
                raise RuntimeError(f"Video cutting failed: {result.stderr}")
            
            # 출력 파일 확인
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Segment cut successfully: {output_path}")
                return output_path
            else:
                raise RuntimeError("Output file is empty or not created")
                
        except subprocess.TimeoutExpired:
            logger.error("Video cutting timed out")
            raise RuntimeError("Video cutting timed out")
        except Exception as e:
            logger.error(f"Video cutting failed: {e}")
            raise
    
    def cut_multiple_segments(self, input_path: str, segments: List[Tuple[float, float]], 
                             output_dir: Optional[str] = None, 
                             codec: str = "copy", quality: str = "high") -> List[str]:
        """
        여러 구간을 한번에 자르기
        
        Args:
            input_path: 입력 비디오 파일 경로
            segments: 구간 리스트 [(start_time, end_time), ...]
            output_dir: 출력 디렉토리 (None이면 기본 디렉토리 사용)
            codec: 코덱 설정
            quality: 품질 설정
        
        Returns:
            출력 파일 경로 리스트
        """
        if not segments:
            raise ValueError("No segments provided")
        
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        output_paths = []
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        for i, (start_time, end_time) in enumerate(segments):
            try:
                output_path = os.path.join(
                    output_dir, 
                    f"{base_name}_segment_{i+1:02d}_{start_time:.1f}_{end_time:.1f}.mp4"
                )
                
                cut_path = self.cut_segment(
                    input_path, start_time, end_time, 
                    output_path, codec, quality
                )
                output_paths.append(cut_path)
                
            except Exception as e:
                logger.error(f"Failed to cut segment {i+1}: {e}")
                continue
        
        logger.info(f"Successfully cut {len(output_paths)}/{len(segments)} segments")
        return output_paths
    
    def cut_from_spans(self, input_path: str, spans: List[Dict[str, Any]], 
                       output_dir: Optional[str] = None,
                       codec: str = "copy", quality: str = "high") -> List[str]:
        """
        Span 리스트에서 구간 자르기
        
        Args:
            input_path: 입력 비디오 파일 경로
            spans: span 리스트 [{"t0": start_time, "t1": end_time, ...}, ...]
            output_dir: 출력 디렉토리
            codec: 코덱 설정
            quality: 품질 설정
        
        Returns:
            출력 파일 경로 리스트
        """
        segments = []
        for span in spans:
            t0 = float(span.get("t0", 0))
            t1 = float(span.get("t1", 0))
            if t0 < t1:
                segments.append((t0, t1))
        
        return self.cut_multiple_segments(input_path, segments, output_dir, codec, quality)
    
    def batch_process(self, video_dir: str, output_dir: Optional[str] = None,
                     file_pattern: str = "*.mp4", **kwargs) -> Dict[str, List[str]]:
        """
        디렉토리 내 모든 비디오 파일에 대해 구간 자르기
        
        Args:
            video_dir: 비디오 파일이 있는 디렉토리
            output_dir: 출력 디렉토리
            file_pattern: 파일 패턴 (glob 형식)
            **kwargs: cut_multiple_segments에 전달할 인자들
        
        Returns:
            {입력파일: [출력파일들]} 형태의 딕셔너리
        """
        import glob
        
        if not os.path.isdir(video_dir):
            raise NotADirectoryError(f"Video directory not found: {video_dir}")
        
        video_files = glob.glob(os.path.join(video_dir, file_pattern))
        if not video_files:
            logger.warning(f"No video files found in {video_dir} with pattern {file_pattern}")
            return {}
        
        results = {}
        for video_file in video_files:
            try:
                logger.info(f"Processing: {video_file}")
                # 여기서는 예시로 10초마다 구간을 자르는 것으로 설정
                # 실제로는 spans 데이터나 다른 로직이 필요
                segments = [(i*10, (i+1)*10) for i in range(5)]  # 예시
                output_paths = self.cut_multiple_segments(video_file, segments, output_dir, **kwargs)
                results[video_file] = output_paths
            except Exception as e:
                logger.error(f"Failed to process {video_file}: {e}")
                results[video_file] = []
        
        return results

# CLI 인터페이스
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Segment Cutter")
    parser.add_argument("--input", type=str, default="/hub_data2/intern/a2aproject/retrieval/tacosvideos/videos/s14-d26-cam-002.avi", help="Input video file path")
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=150, help="End time in seconds")
    parser.add_argument("-o", "--output", default="cut_segments/s14-d26-cam-002.avi", help="Output file path")
    parser.add_argument("-c", "--codec", choices=["copy", "h264", "h265"], default="copy",
                       help="Codec to use")
    parser.add_argument("-q", "--quality", choices=["high", "medium", "low"], default="high",
                       help="Quality setting")
    parser.add_argument("-d", "--output-dir", default="cut_segments",
                       help="Output directory for multiple segments")
    
    args = parser.parse_args()
    
    try:
        cutter = VideoCutter(output_dir=args.output_dir)
        
        # 단일 구간 자르기
        output_path = cutter.cut_segment(
            args.input, args.start, args.end, 
            args.output, args.codec, args.quality
        )
        
        print(f"✅ Successfully cut video segment: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
