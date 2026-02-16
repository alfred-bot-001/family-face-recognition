"""树莓派摄像头客户端 - 抓帧→发送API→展示结果"""

import argparse
import io
import json
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import requests
from flask import Flask, Response, jsonify, send_from_directory

# ============================================================
#  配置
# ============================================================

DEFAULT_API_URL = "http://192.168.0.69:8000"
DEFAULT_CAMERA = 0
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS_LIMIT = 10  # 发送到 API 的帧率上限
DEFAULT_PORT = 5000

# ============================================================
#  全局状态
# ============================================================

latest_frame: np.ndarray | None = None       # 最新标注帧
latest_raw_frame: np.ndarray | None = None   # 最新原始帧
latest_results: list[dict] = []              # 最新识别结果
recognition_fps: float = 0.0                 # 识别 FPS
camera_fps: float = 0.0                      # 摄像头 FPS
is_running: bool = True
lock = threading.Lock()

# ============================================================
#  绘制工具
# ============================================================

def draw_results(frame: np.ndarray, results: list[dict]) -> np.ndarray:
    """在帧上绘制识别结果"""
    annotated = frame.copy()
    for r in results:
        bbox = r.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        name = r.get("name", "unknown")
        conf = r.get("confidence", 0)

        color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{name} ({conf:.2f})"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return annotated

# ============================================================
#  摄像头 + 识别线程
# ============================================================

def camera_loop(api_url: str, camera_id: int, width: int, height: int, fps_limit: int):
    """主循环：抓帧 → 发送 API → 存储结果"""
    global latest_frame, latest_raw_frame, latest_results
    global recognition_fps, camera_fps, is_running

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"[错误] 无法打开摄像头 {camera_id}")
        is_running = False
        return

    print(f"[摄像头] 已打开 ({width}x{height})")
    print(f"[API] 目标: {api_url}/recognize")

    frame_interval = 1.0 / fps_limit if fps_limit > 0 else 0
    cam_frame_count = 0
    cam_fps_start = time.time()
    rec_frame_count = 0
    rec_fps_start = time.time()
    last_send_time = 0

    while is_running:
        ret, frame = cap.read()
        if not ret:
            print("[错误] 读取帧失败")
            time.sleep(0.1)
            continue

        cam_frame_count += 1
        now = time.time()

        # 计算摄像头 FPS
        elapsed_cam = now - cam_fps_start
        if elapsed_cam >= 1.0:
            camera_fps = cam_frame_count / elapsed_cam
            cam_frame_count = 0
            cam_fps_start = now

        # 存储原始帧（用于无识别时也能看到画面）
        with lock:
            latest_raw_frame = frame.copy()

        # 限制发送频率
        if now - last_send_time < frame_interval:
            # 非识别帧也更新显示（用上次的结果画框）
            with lock:
                latest_frame = draw_results(frame, latest_results)
            continue

        last_send_time = now

        # 编码 JPEG
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

        # 发送到 API
        try:
            resp = requests.post(
                f"{api_url}/recognize",
                files={"file": ("frame.jpg", jpeg.tobytes(), "image/jpeg")},
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                faces = data.get("faces", [])
                with lock:
                    latest_results = faces
                    latest_frame = draw_results(frame, faces)

                rec_frame_count += 1
                elapsed_rec = time.time() - rec_fps_start
                if elapsed_rec >= 1.0:
                    recognition_fps = rec_frame_count / elapsed_rec
                    rec_frame_count = 0
                    rec_fps_start = time.time()
            else:
                print(f"[API] 错误: HTTP {resp.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[API] 连接失败: {e}")
            with lock:
                latest_frame = draw_results(frame, [])
                latest_results = []

    cap.release()
    print("[摄像头] 已关闭")

# ============================================================
#  Flask Web 服务
# ============================================================

flask_app = Flask(__name__, static_folder="static")


def generate_mjpeg():
    """MJPEG 流生成器"""
    while is_running:
        with lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )
        time.sleep(0.033)  # ~30fps 输出


@flask_app.route("/")
def index():
    return send_from_directory("static", "index.html")


@flask_app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@flask_app.route("/api/status")
def api_status():
    with lock:
        results = latest_results.copy()
    return jsonify({
        "recognition_fps": round(recognition_fps, 1),
        "camera_fps": round(camera_fps, 1),
        "faces": results,
        "timestamp": datetime.now().isoformat(),
        "running": is_running,
    })


# ============================================================
#  入口
# ============================================================

def main():
    global is_running

    parser = argparse.ArgumentParser(description="树莓派人脸识别客户端")
    parser.add_argument("--api", default=DEFAULT_API_URL, help=f"API 地址 (默认: {DEFAULT_API_URL})")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA, help="摄像头编号")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="分辨率宽度")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="分辨率高度")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS_LIMIT, help="识别帧率上限")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Web 服务端口")
    args = parser.parse_args()

    print(f"{'='*50}")
    print(f"  家庭人脸识别 - 树莓派客户端")
    print(f"  API: {args.api}")
    print(f"  摄像头: {args.camera} ({args.width}x{args.height})")
    print(f"  识别帧率: {args.fps} FPS")
    print(f"  Web 页面: http://0.0.0.0:{args.port}")
    print(f"{'='*50}")

    # 启动摄像头线程
    cam_thread = threading.Thread(
        target=camera_loop,
        args=(args.api, args.camera, args.width, args.height, args.fps),
        daemon=True,
    )
    cam_thread.start()

    # 启动 Flask
    try:
        flask_app.run(host="0.0.0.0", port=args.port, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        is_running = False
        cam_thread.join(timeout=3)
        print("[完成] 已退出")


if __name__ == "__main__":
    main()
