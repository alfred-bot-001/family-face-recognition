"""人脸跟踪模块 - 舵机跟随识别到的家人"""

import argparse
import json
import math
import threading
import time

import cv2
import numpy as np
import requests
import serial

# ============================================================
#  配置
# ============================================================

DEFAULT_API_URL = "http://192.168.0.69:8000"
DEFAULT_CAMERA = 0
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS_LIMIT = 8       # API 识别帧率
DEFAULT_PORT = 5000
DEFAULT_SERIAL = "/dev/ttyAMA0"
DEFAULT_BAUD = 115200

# 跟踪优先级（越靠前越优先）
PRIORITY_NAMES = ["max", "son", "wife"]

# 语音问候配置（人出现后冷却时间内不重复问候）
GREET_COOLDOWN = 300  # 秒（5分钟）
GREET_MESSAGES = {
    "son":  "你好，小虎！",
    "max":  "老大好！",
    "wife": "嫂子好！",
}
GREET_DEFAULT = "你好！"  # 未知已知人脸的默认问候

# 舵机参数（与 ugv_rpi/cv_ctrl.py 机械限位一致）
PAN_MIN, PAN_MAX = -180, 180      # 水平范围
TILT_MIN, TILT_MAX = -30, 90      # 垂直范围
TRACK_ITERATE = 0.045             # 跟踪步进系数
TRACK_SPD_RATE = 60               # 速度系数
TRACK_ACC_RATE = 0.4              # 加速度系数
AIMED_ERROR = 8                   # 瞄准误差阈值（像素）
CMD_GIMBAL = 133                  # 舵机控制指令码

# ============================================================
#  语音问候
# ============================================================

class VoiceGreeter:
    """检测到家人时语音问候（冷却时间内不重复）"""

    def __init__(self, cooldown: float = GREET_COOLDOWN):
        self.cooldown = cooldown
        self.last_greet_time: dict[str, float] = {}  # name -> 上次问候时间
        self.tts_lock = threading.Lock()
        self.engine = None
        self._init_tts()

    def _init_tts(self):
        """初始化 pyttsx3 TTS 引擎（与 ugv_rpi/audio_ctrl.py 一致）"""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 180)  # 语速
            print("[语音] TTS 引擎已初始化")
        except Exception as e:
            self.engine = None
            print(f"[语音] TTS 初始化失败: {e}")

    def should_greet(self, name: str) -> bool:
        """判断是否需要问候（冷却时间外）"""
        if name == "unknown":
            return False
        last = self.last_greet_time.get(name, 0)
        return time.time() - last > self.cooldown

    def greet(self, name: str):
        """异步播放问候语音"""
        if not self.engine:
            return
        if not self.should_greet(name):
            return

        self.last_greet_time[name] = time.time()
        msg = GREET_MESSAGES.get(name, GREET_DEFAULT)
        add_log("INFO", f"🔊 语音问候: {name} → {msg}")

        threading.Thread(target=self._speak, args=(msg,), daemon=True).start()

    def _speak(self, text: str):
        """TTS 播放（线程安全）"""
        with self.tts_lock:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                add_log("ERROR", f"语音播放失败: {e}")

    def check_faces(self, faces: list[dict]):
        """检查所有识别到的人脸，触发问候"""
        for face in faces:
            name = face.get("name", "unknown")
            if name != "unknown":
                self.greet(name)


# ============================================================
#  舵机控制器
# ============================================================

class GimbalController:
    """通过串口控制云台舵机"""

    def __init__(self, port: str = DEFAULT_SERIAL, baud: int = DEFAULT_BAUD):
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            self.connected = True
            print(f"[舵机] 已连接: {port} @ {baud}")
        except Exception as e:
            self.ser = None
            self.connected = False
            print(f"[舵机] 连接失败: {e}")

        self.pan_angle = 0.0    # 当前水平角度
        self.tilt_angle = 0.0   # 当前垂直角度
        self.lock = threading.Lock()

    def send_command(self, data: dict):
        """发送 JSON 指令到底盘"""
        if not self.connected:
            return
        try:
            cmd = json.dumps(data) + "\n"
            self.ser.write(cmd.encode("utf-8"))
        except Exception as e:
            print(f"[舵机] 发送失败: {e}")

    def move_to(self, pan: float, tilt: float, speed: int = 10, acc: int = 1):
        """绝对位置控制"""
        with self.lock:
            self.pan_angle = max(PAN_MIN, min(PAN_MAX, pan))
            self.tilt_angle = max(TILT_MIN, min(TILT_MAX, tilt))
            self.send_command({
                "T": CMD_GIMBAL,
                "X": self.pan_angle,
                "Y": self.tilt_angle,
                "SPD": speed,
                "ACC": acc,
            })

    def track_target(self, frame_cx: int, frame_cy: int,
                     target_x: int, target_y: int,
                     iterate: float = TRACK_ITERATE) -> float:
        """
        跟踪目标：根据目标在画面中的偏移调整舵机
        返回目标到画面中心的距离（像素）
        """
        distance = math.sqrt((target_x - frame_cx) ** 2 + (frame_cy - target_y) ** 2)

        with self.lock:
            # 计算角度增量（与 ugv_rpi/cv_ctrl.py gimbal_track 一致）
            self.pan_angle += (target_x - frame_cx) * iterate
            self.tilt_angle += (frame_cy - target_y) * iterate

            # 限幅
            self.pan_angle = max(PAN_MIN, min(PAN_MAX, self.pan_angle))
            self.tilt_angle = max(TILT_MIN, min(TILT_MAX, self.tilt_angle))

            # 速度和加速度根据距离动态调整
            spd = max(1, int(distance * TRACK_SPD_RATE / 100))
            acc = max(1, int(distance * TRACK_ACC_RATE))

            self.send_command({
                "T": CMD_GIMBAL,
                "X": self.pan_angle,
                "Y": self.tilt_angle,
                "SPD": spd,
                "ACC": acc,
            })

        return distance

    def center(self):
        """回中"""
        self.move_to(0, 0, speed=20, acc=5)

    def close(self):
        if self.ser:
            self.ser.close()

# ============================================================
#  人脸跟踪器
# ============================================================

class FaceTracker:
    """从 API 获取识别结果，驱动舵机跟踪家人"""

    def __init__(self, api_url: str, gimbal: GimbalController,
                 width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT):
        self.api_url = api_url
        self.gimbal = gimbal
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2

        # 状态
        self.tracking_name = None
        self.tracking_confidence = 0.0
        self.last_seen_time = 0.0
        self.lost_timeout = 3.0  # 丢失目标后多久回中

    def select_target(self, faces: list[dict]) -> dict | None:
        """
        从识别结果中选择跟踪目标
        优先级：PRIORITY_NAMES 中的顺序 > 最大人脸
        """
        if not faces:
            return None

        # 过滤掉 unknown
        known_faces = [f for f in faces if f.get("name", "unknown") != "unknown"]

        if not known_faces:
            return None

        # 按优先级选择
        for priority_name in PRIORITY_NAMES:
            candidates = [f for f in known_faces if f["name"] == priority_name]
            if candidates:
                # 多个同名取置信度最高的
                return max(candidates, key=lambda f: f.get("confidence", 0))

        # 没有优先目标，取置信度最高的已知人脸
        return max(known_faces, key=lambda f: f.get("confidence", 0))

    def get_face_center(self, face: dict) -> tuple[int, int]:
        """获取人脸中心坐标"""
        bbox = face.get("bbox", [0, 0, 0, 0])
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        return cx, cy

    def update(self, faces: list[dict]):
        """根据识别结果更新跟踪"""
        target = self.select_target(faces)

        if target:
            self.tracking_name = target["name"]
            self.tracking_confidence = target.get("confidence", 0)
            self.last_seen_time = time.time()

            tx, ty = self.get_face_center(target)
            distance = self.gimbal.track_target(
                self.center_x, self.center_y, tx, ty
            )

            if distance < AIMED_ERROR:
                status = "🎯 锁定"
            else:
                status = "🔄 追踪"

            print(f"  {status} {self.tracking_name} "
                  f"(置信度: {self.tracking_confidence:.2f}, "
                  f"偏移: {distance:.0f}px, "
                  f"舵机: {self.gimbal.pan_angle:.1f}°, {self.gimbal.tilt_angle:.1f}°)")
        else:
            # 没有目标
            if self.tracking_name and time.time() - self.last_seen_time > self.lost_timeout:
                print(f"  ⚠️ 丢失目标 {self.tracking_name}，回中...")
                self.gimbal.center()
                self.tracking_name = None

# ============================================================
#  主循环（摄像头 + API + 跟踪 + Web）
# ============================================================

from flask import Flask, Response, jsonify, send_from_directory

# 全局状态
latest_frame: np.ndarray | None = None
latest_results: list[dict] = []
tracker_status: dict = {}
is_running = True
lock = threading.Lock()

# 日志缓冲
from collections import deque
log_buffer = deque(maxlen=100)

def add_log(level: str, msg: str):
    """添加系统日志"""
    ts = time.strftime("%H:%M:%S")
    entry = {"time": ts, "level": level, "msg": msg}
    log_buffer.appendleft(entry)
    print(f"[{level}] {ts} {msg}")

flask_app = Flask(__name__, static_folder="static")


def draw_tracking_results(frame: np.ndarray, faces: list[dict],
                          tracking_name: str | None) -> np.ndarray:
    """在帧上绘制识别结果，高亮跟踪目标"""
    annotated = frame.copy()
    for f in faces:
        bbox = f.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        name = f.get("name", "unknown")
        conf = f.get("confidence", 0)
        is_target = (name == tracking_name)

        if is_target:
            color = (0, 255, 255)  # 黄色 = 跟踪目标
            thickness = 3
        elif name != "unknown":
            color = (0, 255, 0)    # 绿色 = 已知
            thickness = 2
        else:
            color = (0, 0, 255)    # 红色 = 未知
            thickness = 1

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = f"{'>>> ' if is_target else ''}{name} ({conf:.2f})"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - h - 8), (x1 + w, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 画中心十字准星
    h, w = annotated.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(annotated, (cx - 15, cy), (cx + 15, cy), (255, 255, 255), 1)
    cv2.line(annotated, (cx, cy - 15), (cx, cy + 15), (255, 255, 255), 1)

    return annotated


def open_camera(camera_id: int, width: int, height: int):
    """
    打开摄像头（OpenCV 统一处理 USB 和 CSI）
    RPi5 上 OpenCV 通过 V4L2/GStreamer 可直接访问 CSI 摄像头
    """
    cap = cv2.VideoCapture(camera_id)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ret, frame = cap.read()
        if ret:
            actual_h, actual_w = frame.shape[:2]
            add_log("INFO", f"摄像头已打开: /dev/video{camera_id} ({actual_w}x{actual_h})")
            return ("opencv", cap)
        else:
            cap.release()
            add_log("WARN", f"摄像头 {camera_id} 打开成功但读取失败")
    else:
        add_log("WARN", f"摄像头 {camera_id} 无法打开")

    add_log("ERROR", "无法打开摄像头")
    return (None, None)


def read_frame(cam_type, cam_obj):
    """从摄像头读取一帧，返回 BGR numpy 数组"""
    try:
        ret, frame = cam_obj.read()
        return frame if ret else None
    except Exception as e:
        add_log("ERROR", f"读取帧异常: {e}")
    return None


def close_camera(cam_type, cam_obj):
    """关闭摄像头"""
    try:
        cam_obj.release()
    except Exception as e:
        add_log("WARN", f"关闭摄像头异常: {e}")


def make_placeholder_frame(width: int, height: int, text: str = "摄像头未连接") -> np.ndarray:
    """生成占位图（参考 ugv_rpi 的 camera read failed 画面）"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)
    # 中心文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    x = (width - tw) // 2
    y = (height + th) // 2
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 255), 2)
    return frame


def camera_tracking_loop(api_url: str, camera_id: int, width: int, height: int,
                         fps_limit: int, gimbal: GimbalController,
                         greeter: VoiceGreeter):
    """主循环：摄像头 → API → 跟踪 → 舵机 → 语音"""
    global latest_frame, latest_results, tracker_status, is_running

    tracker = FaceTracker(api_url, gimbal, width, height)

    cam_type, cam_obj = open_camera(camera_id, width, height)
    retry_interval = 5  # 摄像头重试间隔（秒）
    last_retry = 0
    frame_interval = 1.0 / fps_limit
    last_send = 0
    frame_count = 0
    api_ok_count = 0
    api_err_count = 0
    read_fail_count = 0

    while is_running:
        # 没有摄像头时：显示占位图 + 定期重试
        if cam_type is None:
            with lock:
                latest_frame = make_placeholder_frame(width, height, "Camera Disconnected - Retrying...")
            now = time.time()
            if now - last_retry > retry_interval:
                last_retry = now
                add_log("INFO", "尝试重新连接摄像头...")
                cam_type, cam_obj = open_camera(camera_id, width, height)
            time.sleep(0.2)
            continue

        frame = read_frame(cam_type, cam_obj)
        if frame is None:
            read_fail_count += 1
            if read_fail_count == 1:
                add_log("WARN", "读取帧失败，尝试中...")
            if read_fail_count > 30:
                add_log("ERROR", f"连续 {read_fail_count} 次读取失败，重新打开摄像头")
                close_camera(cam_type, cam_obj)
                cam_type, cam_obj = None, None
                read_fail_count = 0
            time.sleep(0.1)
            continue

        read_fail_count = 0
        frame_count += 1
        if frame_count == 1:
            add_log("INFO", f"首帧获取成功: {frame.shape}")

        now = time.time()

        # 非识别帧：用上次结果更新显示
        if now - last_send < frame_interval:
            with lock:
                latest_frame = draw_tracking_results(frame, latest_results, tracker.tracking_name)
            continue

        last_send = now

        # 编码发送
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        try:
            resp = requests.post(
                f"{api_url}/recognize",
                files={"file": ("frame.jpg", jpeg.tobytes(), "image/jpeg")},
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                faces = data.get("faces", [])
                api_ok_count += 1

                if api_ok_count == 1:
                    add_log("INFO", f"API 首次响应成功，检测到 {len(faces)} 张脸")

                # 更新跟踪
                tracker.update(faces)

                # 语音问候
                greeter.check_faces(faces)

                with lock:
                    latest_results = faces
                    latest_frame = draw_tracking_results(frame, faces, tracker.tracking_name)
                    tracker_status = {
                        "tracking": tracker.tracking_name,
                        "confidence": round(tracker.tracking_confidence, 3),
                        "pan": round(gimbal.pan_angle, 1),
                        "tilt": round(gimbal.tilt_angle, 1),
                        "faces_count": len(faces),
                        "known_count": len([f for f in faces if f.get("name") != "unknown"]),
                        "frame_count": frame_count,
                        "api_ok": api_ok_count,
                        "api_err": api_err_count,
                        "greet_history": {k: time.strftime("%H:%M:%S", time.localtime(v))
                                          for k, v in greeter.last_greet_time.items()},
                    }
            else:
                api_err_count += 1
                add_log("ERROR", f"API HTTP {resp.status_code}")
        except requests.exceptions.RequestException as e:
            api_err_count += 1
            if api_err_count <= 3 or api_err_count % 10 == 0:
                add_log("ERROR", f"API 连接失败: {e}")
            with lock:
                latest_results = []
                latest_frame = draw_tracking_results(frame, [], None)

    close_camera(cam_type, cam_obj)
    gimbal.center()
    gimbal.close()


# ============================================================
#  Flask 路由
# ============================================================

def generate_mjpeg():
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
        time.sleep(0.033)


@flask_app.route("/")
def index():
    return send_from_directory("static", "tracker.html")


@flask_app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@flask_app.route("/api/status")
def api_status():
    with lock:
        return jsonify({
            **tracker_status,
            "faces": latest_results.copy(),
            "running": is_running,
        })


@flask_app.route("/api/logs")
def api_logs():
    return jsonify(list(log_buffer))


@flask_app.route("/api/gimbal/center", methods=["POST"])
def gimbal_center():
    """手动回中"""
    gimbal_instance.center()
    return jsonify({"ok": True})


# ============================================================
#  入口
# ============================================================

gimbal_instance: GimbalController = None


def main():
    global is_running, gimbal_instance

    parser = argparse.ArgumentParser(description="家庭人脸跟踪 - 舵机追踪家人")
    parser.add_argument("--api", default=DEFAULT_API_URL, help="API 地址")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA, help="摄像头编号")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS_LIMIT, help="识别帧率")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Web 端口")
    parser.add_argument("--serial", default=DEFAULT_SERIAL, help="舵机串口")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="波特率")
    parser.add_argument("--no-gimbal", action="store_true", help="禁用舵机（仅显示）")
    args = parser.parse_args()

    print(f"{'='*50}")
    print(f"  家庭人脸跟踪 - 舵机追踪模式")
    print(f"  API: {args.api}")
    print(f"  摄像头: {args.camera} ({args.width}x{args.height})")
    print(f"  舵机: {args.serial} ({'禁用' if args.no_gimbal else '启用'})")
    print(f"  跟踪优先级: {' > '.join(PRIORITY_NAMES)}")
    print(f"  Web: http://0.0.0.0:{args.port}")
    print(f"{'='*50}")

    # 初始化舵机
    if args.no_gimbal:
        gimbal_instance = GimbalController.__new__(GimbalController)
        gimbal_instance.connected = False
        gimbal_instance.ser = None
        gimbal_instance.pan_angle = 0
        gimbal_instance.tilt_angle = 0
        gimbal_instance.lock = threading.Lock()
    else:
        gimbal_instance = GimbalController(args.serial, args.baud)
        gimbal_instance.center()
        time.sleep(0.5)

    # 初始化语音问候
    greeter_instance = VoiceGreeter(cooldown=GREET_COOLDOWN)

    # 启动摄像头+跟踪线程
    cam_thread = threading.Thread(
        target=camera_tracking_loop,
        args=(args.api, args.camera, args.width, args.height, args.fps,
              gimbal_instance, greeter_instance),
        daemon=True,
    )
    cam_thread.start()

    # Flask
    try:
        flask_app.run(host="0.0.0.0", port=args.port, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        is_running = False
        cam_thread.join(timeout=3)
        if gimbal_instance.connected:
            gimbal_instance.center()
            gimbal_instance.close()
        print("[完成] 已退出")


if __name__ == "__main__":
    main()
