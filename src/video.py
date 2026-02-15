"""视频流人脸识别模块"""

import time
import cv2
import numpy as np

from .detector import FaceDetector
from .recognizer import FaceRecognizer


class VideoRecognizer:
    def __init__(
        self,
        detector: FaceDetector | None = None,
        recognizer: FaceRecognizer | None = None,
        skip_frames: int = 2,
    ):
        """
        视频流人脸识别
        :param detector: 人脸检测器
        :param recognizer: 人脸识别器
        :param skip_frames: 每隔几帧检测一次（提高性能）
        """
        self.detector = detector or FaceDetector()
        self.recognizer = recognizer or FaceRecognizer()
        self.skip_frames = skip_frames
        self._last_results: list[dict] = []

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> list[dict]:
        """
        处理单帧
        :param frame: BGR 图像
        :param frame_idx: 帧序号
        :return: [{"bbox": [...], "name": str, "confidence": float}, ...]
        """
        # 跳帧策略：非检测帧复用上次结果
        if frame_idx % (self.skip_frames + 1) != 0:
            return self._last_results

        faces = self.detector.detect_with_info(frame)
        results = []

        for face in faces:
            name, confidence = self.recognizer.recognize(face["embedding"])
            results.append({
                "bbox": face["bbox"],
                "name": name,
                "confidence": confidence,
                "score": face["score"],
            })

        self._last_results = results
        return results

    @staticmethod
    def draw_results(frame: np.ndarray, results: list[dict]) -> np.ndarray:
        """在帧上绘制识别结果"""
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            name = r["name"]
            conf = r["confidence"]

            # 颜色：已识别绿色，未知红色
            color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{name} ({conf:.2f})"
            # 背景框
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def run_camera(self, source: int | str = 0, window_name: str = "Face Recognition"):
        """
        运行摄像头实时识别
        :param source: 摄像头编号或视频文件路径
        :param window_name: 窗口名称
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[错误] 无法打开视频源: {source}")
            return

        print(f"[视频] 开始实时识别 (按 'q' 退出)")
        frame_idx = 0
        fps_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.process_frame(frame, frame_idx)
                frame = self.draw_results(frame, results)

                # FPS 显示
                now = time.time()
                fps = 1.0 / (now - fps_time) if now != fps_time else 0
                fps_time = now
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                frame_idx += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("[视频] 已停止")

    def process_video_file(self, video_path: str, output_path: str | None = None) -> list[dict]:
        """
        处理视频文件，返回每帧识别结果
        :param video_path: 输入视频路径
        :param output_path: 输出视频路径（可选，带标注）
        :return: 所有帧的识别结果汇总
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[错误] 无法打开视频: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"[视频] 处理中: {total_frames} 帧, {fps:.1f} FPS, {width}x{height}")

        all_results = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.process_frame(frame, frame_idx)
            all_results.append({"frame": frame_idx, "faces": results})

            if writer:
                annotated = self.draw_results(frame.copy(), results)
                writer.write(annotated)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  进度: {frame_idx}/{total_frames}")

        cap.release()
        if writer:
            writer.release()
            print(f"[视频] 输出已保存: {output_path}")

        # 汇总识别到的人
        seen = {}
        for fr in all_results:
            for face in fr["faces"]:
                name = face["name"]
                if name != "unknown":
                    if name not in seen:
                        seen[name] = {"count": 0, "max_confidence": 0}
                    seen[name]["count"] += 1
                    seen[name]["max_confidence"] = max(seen[name]["max_confidence"], face["confidence"])

        print(f"[视频] 识别汇总: {seen}")
        return all_results
