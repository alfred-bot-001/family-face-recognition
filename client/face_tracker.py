"""人脸跟踪 + 多多语音对话 — 集成版主程序"""

import argparse
import asyncio
import json
import math
import os
import subprocess
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
DEFAULT_FPS_LIMIT = 8
DEFAULT_PORT = 5000
DEFAULT_SERIAL = "/dev/ttyAMA0"
DEFAULT_BAUD = 115200

# 跟踪优先级
PRIORITY_NAMES = ["son", "max", "wife"]

# 语音问候
GREET_COOLDOWN = 999999
GREET_MESSAGES = {
    "son": "你好，小虎！",
    "max": "老大好！",
    "wife": "嫂子好！",
}
GREET_DEFAULT = "你好！"

# 摄像头空闲关闭
IDLE_CAMERA_TIMEOUT = 1800  # 30分钟无人脸+无交互则关闭摄像头

# 舵机参数
PAN_MIN, PAN_MAX = -180, 180
TILT_MIN, TILT_MAX = -30, 90
PAN_OFFSET = -40
TRACK_ITERATE = 0.045
TRACK_SPD_RATE = 60
TRACK_ACC_RATE = 0.4
AIMED_ERROR = 8
CMD_GIMBAL = 133

# 多多对话配置
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION_MS = 60
FRAME_SIZE = SAMPLE_RATE * FRAME_DURATION_MS // 1000  # 960
AUDIO_PLAY = "plughw:2,0"   # USB PnP Audio Device 扬声器
AUDIO_REC = "plughw:5,0"    # M2 麦克风阵列录音
WAKE_WORD = "多多"
M2_SERIAL_PORT = "/dev/m2_mic"
M2_BAUD = 115200
XIAOZHI_WS_URL = "ws://192.168.0.69:8100/xiaozhi/v1/"
XIAOZHI_DEVICE_ID = "pi-laosan-001"

# 视觉识别 (智控台 VLLM API)
VISION_API_URL = "http://192.168.0.69:8103/mcp/vision/explain"
VISION_AUTH_SECRET = "f9e18e72-09ad-4cdf-9a34-62ee2ff2adfc"
_VISION_KEYWORDS = ['看看', '看一下', '你看', '看到了什么', '看到什么', '前面有什么', '周围有什么', '眼前', '看一看', '看我', '手里拿', '拿的什么', '这是什么', '那是什么', '什么东西']

# 拍照功能
_PHOTO_KEYWORDS = ['拍照', '拍个照', '拍张照', '照相', '拍一张', '来一张', '茄子']
PHOTO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "photos")
TELEGRAM_BOT_TOKEN = "8517750579:AAEHgdBOp8A2T-ORimYcwbMyFhmxfUBDMkM"
TELEGRAM_CHAT_ID = "7929939096"

SHERPA_ASR_DIR = os.path.join(os.path.dirname(__file__), "models",
                              "sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16", "96")

import logging
from emotions import play_emotion_from_text, play_emotion
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("main")


def _contains_wake(text):
    t = "".join(ch for ch in text if not ch.isspace())
    if "多多" in t:
        return True
    if t.startswith("多") or t.startswith("哆"):
        return True
    return False


# ============================================================
#  日志
# ============================================================
from collections import deque
log_buffer = deque(maxlen=100)

def add_log(level: str, msg: str):
    ts = time.strftime("%H:%M:%S")
    entry = {"time": ts, "level": level, "msg": msg}
    log_buffer.appendleft(entry)
    log.info(f"[{level}] {msg}")


# ============================================================
#  Opus 编解码
# ============================================================
try:
    import opuslib_next as opuslib
    _encoder = opuslib.Encoder(SAMPLE_RATE, CHANNELS, opuslib.APPLICATION_VOIP)
    _decoder = opuslib.Decoder(SAMPLE_RATE, CHANNELS)
    OPUS_OK = True
except ImportError:
    OPUS_OK = False
    log.warning("opuslib_next 未安装，多多对话不可用")

def pcm_to_opus(pcm: bytes) -> bytes:
    return _encoder.encode(pcm, FRAME_SIZE)

def opus_to_pcm(data: bytes) -> bytes:
    return _decoder.decode(data, FRAME_SIZE)


# ============================================================
#  本地 TTS (espeak → M2 扬声器)
# ============================================================
_tts_lock = threading.Lock()

def speak(text: str, device: str = AUDIO_PLAY):
    with _tts_lock:
        try:
            subprocess.run(
                f'espeak -v zh -s 320 --stdout "{text}" | aplay -D {device} -q',
                shell=True, stderr=subprocess.DEVNULL, timeout=10
            )
        except Exception as e:
            log.error(f"TTS 播放失败: {e}")

def speak_async(text: str, device: str = AUDIO_PLAY):
    threading.Thread(target=speak, args=(text, device), daemon=True).start()

def edge_tts_speak(text: str, voice: str = "en-US-AnaNeural", device: str = AUDIO_PLAY):
    """用 edge-tts 生成语音并通过指定声卡播放（不经过LLM）"""
    import tempfile, os
    with _tts_lock:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            subprocess.run(
                ["edge-tts", "--voice", voice, "--text", text, "--write-media", tmp_path],
                capture_output=True, timeout=15
            )
            # mp3 → pcm via ffmpeg → aplay 指定声卡
            subprocess.run(
                f"ffmpeg -y -i {tmp_path} -f s16le -ar 24000 -ac 1 - 2>/dev/null | aplay -D {device} -f S16_LE -r 24000 -c 1 -q",
                shell=True, timeout=30
            )
        except Exception as e:
            log.error(f"edge-tts 播放失败: {e}")
        finally:
            if tmp_path:
                try: os.unlink(tmp_path)
                except: pass


# ============================================================
#  M2 硬件唤醒词检测 (正确帧协议)
# ============================================================
# M2 状态 (全局，页面可读)
m2_state = {
    "connected": False,
    "heartbeat": 0,
    "events": [],       # 唤醒事件列表 [{time, angle, raw}]
    "last_wake": None,
}

class M2WakeWordListener:
    """M2 麦克风阵列串口监听
    帧格式: 0xA5 | user_id(1) | msg_type(1) | data_len(2,LE) | msg_id(2) | data(N) | checksum(1)
    msg_type: 0x01=心跳, 0x04=唤醒(含角度JSON), 0x05=设置, 0xFF=确认
    """
    def __init__(self, port=M2_SERIAL_PORT, baud=M2_BAUD, should_pause_fn=None):
        self.port = port
        self.baud = baud
        self.should_pause_fn = should_pause_fn
        self.paused = False
        self.active = True
        self._cooldown = 2.0
        self._last_detect = 0
        self.ser = None
        self._open_serial()

    def _open_serial(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            m2_state["connected"] = True
            log.info(f"M2 硬件唤醒词监听已初始化: {self.port}")
            return True
        except Exception as e:
            log.warning(f"M2 串口打开失败: {e}")
            m2_state["connected"] = False
            self.ser = None
            return False

    def _read_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.ser.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def start(self, on_wake):
        threading.Thread(target=self._listen, args=(on_wake,), daemon=True).start()

    def _listen(self, on_wake):
        reconnect_interval = 5
        last_reconnect = 0

        while self.active:
            if self.ser is None:
                now = time.time()
                if now - last_reconnect > reconnect_interval:
                    last_reconnect = now
                    if not self._open_serial():
                        time.sleep(1)
                        continue
                else:
                    time.sleep(0.5)
                    continue

            if self.paused:
                time.sleep(0.1)
                continue

            try:
                # 找帧头 0xA5
                b = self.ser.read(1)
                if not b or b[0] != 0xA5:
                    continue

                # user_id + msg_type
                header = self._read_exact(2)
                if not header:
                    continue
                user_id = header[0]
                msg_type = header[1]

                if user_id != 0x01:
                    continue

                # data_len (2 bytes LE)
                len_bytes = self._read_exact(2)
                if not len_bytes:
                    continue
                data_len = len_bytes[0] | (len_bytes[1] << 8)
                if data_len > 4096:
                    continue

                # msg_id(2) + data(N) + checksum(1)
                rest = self._read_exact(2 + data_len + 1)
                if not rest:
                    continue

                data = rest[2:2 + data_len]

                if msg_type == 0x01:
                    m2_state["heartbeat"] += 1

                elif msg_type == 0x04:
                    # 唤醒事件 — 解析角度
                    angle = -1
                    raw = ""
                    try:
                        json_str = data.decode("utf-8", errors="ignore")
                        raw = json_str
                        parsed = json.loads(json_str)
                        info = json.loads(parsed["content"]["info"])
                        angle = info.get("ivw", {}).get("angle", -1)
                    except Exception:
                        raw = data.hex() if data else ""

                    event = {
                        "time": time.strftime("%H:%M:%S"),
                        "timestamp": time.time(),
                        "angle": angle,
                        "raw": raw[:200],
                    }
                    m2_state["events"].append(event)
                    m2_state["last_wake"] = event
                    if len(m2_state["events"]) > 50:
                        m2_state["events"] = m2_state["events"][-50:]

                    add_log("INFO", f"🎯 M2 硬件唤醒! 角度={angle}°")

                    now = time.time()
                    if now - self._last_detect > self._cooldown:
                        self._last_detect = now
                        on_wake()

            except Exception as e:
                err_str = str(e)
                if "returned no data" in err_str or "disconnected" in err_str:
                    log.warning(f"M2 串口断线，将重连")
                    m2_state["connected"] = False
                    try:
                        self.ser.close()
                    except Exception:
                        pass
                    self.ser = None
                else:
                    log.error(f"M2 串口错误: {e}")
                    time.sleep(0.5)

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False
        try:
            if self.ser and self.ser.in_waiting > 0:
                self.ser.reset_input_buffer()
        except Exception:
            pass

    def stop(self):
        self.active = False
        m2_state["connected"] = False
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass


# ============================================================
#  sherpa-onnx 唤醒词检测 (备用)
# ============================================================
class WakeWordListener:
    def __init__(self, device=AUDIO_REC, should_pause_fn=None):
        import sherpa_onnx
        import numpy as np_
        self.sherpa_onnx = sherpa_onnx
        self.np = np_
        self.device = device
        self.should_pause_fn = should_pause_fn
        self.paused = False
        self.active = True
        self._proc = None
        self._cooldown = 2.0
        self._last_detect = 0

        encoder = os.path.join(SHERPA_ASR_DIR, "encoder-epoch-99-avg-1.onnx")
        decoder = os.path.join(SHERPA_ASR_DIR, "decoder-epoch-99-avg-1.onnx")
        joiner = os.path.join(SHERPA_ASR_DIR, "joiner-epoch-99-avg-1.onnx")
        tokens = os.path.join(SHERPA_ASR_DIR, "tokens.txt")

        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=encoder, decoder=decoder, joiner=joiner, tokens=tokens,
            num_threads=4, sample_rate=SAMPLE_RATE, feature_dim=80, provider="cpu",
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=300,
        )
        log.info("sherpa-onnx 流式ASR 加载完成")

    def start(self, on_wake):
        threading.Thread(target=self._listen, args=(on_wake,), daemon=True).start()

    def _listen(self, on_wake):
        chunk_samples = int(SAMPLE_RATE * 0.1)
        chunk_bytes = chunk_samples * 2
        last_text = ""

        while self.active:
            if self.paused or (self.should_pause_fn and self.should_pause_fn()):
                if self._proc is not None:
                    try:
                        self._proc.terminate()
                    except Exception:
                        pass
                    self._proc = None
                time.sleep(0.1)
                continue

            if self._proc is None or self._proc.poll() is not None:
                self._proc = subprocess.Popen(
                    ["arecord", "-D", self.device, "-f", "S16_LE",
                     "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw", "-q"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                stream = self.recognizer.create_stream()
                last_text = ""
                log.info(f"👂 监听唤醒词: {WAKE_WORD}")

            data = self._proc.stdout.read(chunk_bytes)
            if not data:
                time.sleep(0.05)
                continue

            if self.paused:
                continue

            samples = self.np.frombuffer(data, dtype=self.np.int16).astype(self.np.float32) / 32768.0
            stream.accept_waveform(SAMPLE_RATE, samples)
            while self.recognizer.is_ready(stream):
                self.recognizer.decode_stream(stream)
            text = self.recognizer.get_result(stream).strip()
            if text and text != last_text:
                log.info(f"asr: {text}")
                last_text = text
                if _contains_wake(text):
                    self._trigger(on_wake, text, stream)
            if self.recognizer.is_endpoint(stream):
                if text and _contains_wake(text):
                    self._trigger(on_wake, text, stream)
                self.recognizer.reset(stream)
                last_text = ""

    def _trigger(self, on_wake, text, stream):
        now = time.time()
        if now - self._last_detect < self._cooldown:
            return
        self._last_detect = now
        log.info(f"🎯 唤醒词! ({text})")
        self.recognizer.reset(stream)
        on_wake()

    def pause(self):
        self.paused = True
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                pass
            self._proc = None

    def resume(self):
        self.paused = False

    def stop(self):
        self.active = False
        if self._proc:
            self._proc.terminate()


# ============================================================
#  多多 WebSocket 客户端
# ============================================================

# ============================================================
#  视觉识别 (抓帧 → 智控台 VLLM API)
# ============================================================
def _get_vision_token() -> str:
    """生成智控台 Vision API 的 JWT token"""
    import jwt as pyjwt
    from datetime import datetime, timedelta, timezone
    expire = datetime.now(timezone.utc) + timedelta(hours=1)
    payload = {"device_id": XIAOZHI_DEVICE_ID, "exp": expire.timestamp()}
    return pyjwt.encode(payload, VISION_AUTH_SECRET, algorithm="HS256")

def _vision_describe(prompt: str = "请用简短的中文描述你看到的画面，不超过3句话。") -> str | None:
    """抓取当前摄像头画面，调用智控台视觉API描述"""
    global latest_raw_frame
    frame = latest_raw_frame
    if frame is None:
        return "我现在看不到东西，摄像头可能没开。"
    try:
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        token = _get_vision_token()
        resp = requests.post(
            VISION_API_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Device-Id": XIAOZHI_DEVICE_ID,
                "Client-Id": XIAOZHI_DEVICE_ID,
            },
            files={
                "question": (None, prompt),
                "image": ("frame.jpg", jpeg.tobytes(), "image/jpeg"),
            },
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success"):
                return data.get('result', data.get('response', '我看到了，但说不出来。')).strip()
            else:
                log.error(f"Vision API 业务错误: {data.get('message')}")
                return "识别出了点问题，稍后再试。"
        else:
            log.error(f"Vision API HTTP错误: {resp.status_code}")
            return "识别出了点问题，稍后再试。"
    except Exception as e:
        log.error(f"Vision 识别失败: {e}")
        return "识别出了点问题，稍后再试。"


class XiaozhiClient:
    def __init__(self, ws_url: str, device_id: str, gimbal=None):
        self.ws_url = ws_url
        self.device_id = device_id
        self.gimbal = gimbal
        self.session_id = None
        self.ws = None
        self.connected = False
        self.is_speaking = False
        self.is_listening = False
        self._rec_proc = None
        self._play_proc = None
        self._send_task = None
        self._mute = False  # 打断时静音，忽略残余音频帧
        self._stt_ignore_until = 0  # 忽略自己发的detect回显直到此时间戳

    async def connect(self):
        import websockets
        headers = {
            "Device-Id": self.device_id,
            "Client-Id": self.device_id,
            "Protocol-Version": "1",
        }
        log.info(f"🔗 多多连接 {self.ws_url}")
        try:
            self.ws = await websockets.connect(
                self.ws_url, max_size=None, additional_headers=headers,
                ping_interval=20, ping_timeout=20, close_timeout=10,
            )
        except TypeError:
            import websockets
            self.ws = await websockets.connect(
                self.ws_url, max_size=None, extra_headers=headers,
                ping_interval=20, ping_timeout=20, close_timeout=10,
            )

        hello = {
            "type": "hello", "version": 1, "transport": "websocket",
            "device_id": self.device_id, "device_name": "老三-树莓派",
            "features": {"mcp": False},
            "audio_params": {
                "format": "opus", "sample_rate": SAMPLE_RATE,
                "channels": CHANNELS, "frame_duration": FRAME_DURATION_MS,
            },
        }
        await self.ws.send(json.dumps(hello))
        resp = await asyncio.wait_for(self.ws.recv(), timeout=10)
        try:
            data = json.loads(resp)
            self.session_id = data.get("session_id", "")
            self.connected = True
            log.info(f"✅ 多多连接成功, session: {self.session_id}")
            add_log("INFO", f"🔗 多多已连接")
        except Exception as e:
            log.error(f"多多握手失败: {e}")
            return False
        return True

    async def _keepalive(self):
        """每60秒发ping保活，防止服务端超时断连"""
        try:
            while self.connected:
                await asyncio.sleep(60)
                if self.connected and self.ws:
                    try:
                        await self.ws.send(json.dumps({
                            "session_id": self.session_id,
                            "type": "ping"
                        }))
                        log.debug("keepalive ping sent")
                    except Exception:
                        break
        except asyncio.CancelledError:
            pass

    async def message_loop(self):
        keepalive_task = asyncio.create_task(self._keepalive())
        try:
            self._last_active = time.time()
            async for message in self.ws:
                self._last_active = time.time()
                if isinstance(message, bytes):
                    if self._mute:
                        continue  # 打断中，丢弃残余音频帧
                    self._audio_count = getattr(self, '_audio_count', 0) + 1
                    if self._audio_count <= 3 or self._audio_count % 100 == 0:
                        log.info(f"🔈 收到音频帧 #{self._audio_count}, {len(message)} bytes")
                    try:
                        pcm = opus_to_pcm(message)
                        self._play_pcm(pcm)
                    except Exception as e:
                        log.error(f"音频解码错误: {e}")
                else:
                    data = json.loads(message)
                    await self._handle(data)
        except Exception as e:
            log.error(f"多多连接断开: {e}")
            self.connected = False
            add_log("WARN", f"多多断线: {e}")
        finally:
            keepalive_task.cancel()

    async def _handle(self, msg: dict):
        t = msg.get("type", "")
        if t == "tts":
            state = msg.get("state", "")
            if state == "start":
                self._mute = False  # 新的 TTS 开始，解除静音
                self.is_speaking = True
                add_log("INFO", "🔊 多多开始说话")
                if self.is_listening:
                    await self.stop_listening()
            elif state == "sentence_start":
                text = msg.get('text', '')
                add_log("INFO", f"💬 {text}")
                # 情绪动作（检测 LLM 回复文本触发）
                if self.gimbal and getattr(self.gimbal, 'connected', False):
                    play_emotion_from_text(self.gimbal, text)
                # 休眠检测已移到 STT（用户说的话触发，不检测 LLM 回复）
            elif state == "stop":
                self.is_speaking = False
                if self._play_proc:
                    try:
                        self._play_proc.stdin.close()
                    except Exception:
                        pass
                    try:
                        self._play_proc.wait(timeout=2)
                    except Exception:
                        self._play_proc.kill()
                    self._play_proc = None
                add_log("INFO", "🔊 多多说话结束")
                # 假睡由 _do_fake_sleep 处理，不在这里
        elif t == "stt":
            stt_text = msg.get('text', '')
            add_log("INFO", f"🎤 识别: {stt_text}")
            # 跳过自己发的detect消息回显（时间窗口内忽略）
            if time.time() < self._stt_ignore_until:
                return
            # 用户说了休息相关的话 → 直接打断并假睡
            _sleep_kw = ['你休息', '去休息', '去睡', '你睡', '关机', '待机', '休眠']
            if any(kw in stt_text for kw in _sleep_kw):
                asyncio.ensure_future(self._do_fake_sleep())
            # 视觉识别: 用户说"看看"类关键词 → 打断LLM + 启动视觉
            if any(kw in stt_text for kw in _VISION_KEYWORDS):
                add_log("INFO", "👁️ 触发视觉识别，打断当前对话...")
                self._mute = True
                asyncio.ensure_future(self._abort_and_vision(stt_text))
            # 拍照: 用户说"拍照"类关键词
            if any(kw in stt_text for kw in _PHOTO_KEYWORDS):
                add_log("INFO", "📸 触发拍照...")
                self._mute = True
                asyncio.ensure_future(self._do_take_photo())
        elif t == "llm":
            add_log("INFO", f"🤖 {msg.get('text', '')}")
        elif t == "hello":
            self.session_id = msg.get("session_id", self.session_id)

    def _play_pcm(self, pcm: bytes):
        try:
            if self._play_proc is None or self._play_proc.poll() is not None:
                log.info(f"🔊 启动 aplay 进程 (设备: {AUDIO_PLAY})")
                self._play_proc = subprocess.Popen(
                    ["aplay", "-D", AUDIO_PLAY, "-f", "S16_LE",
                     "-r", str(SAMPLE_RATE), "-c", "1", "-q"],
                    stdin=subprocess.PIPE, stderr=subprocess.PIPE
                )
                # 检查是否启动成功
                time.sleep(0.05)
                if self._play_proc.poll() is not None:
                    stderr = self._play_proc.stderr.read(500).decode(errors='ignore')
                    log.error(f"aplay 启动失败: {stderr}")
                    self._play_proc = None
                    return
            self._play_proc.stdin.write(pcm)
            self._play_proc.stdin.flush()
        except BrokenPipeError:
            log.warning("aplay 管道断开，重置播放进程")
            self._play_proc = None
        except Exception as e:
            log.error(f"播放错误: {e}")
            self._play_proc = None

    async def _do_take_photo(self):
        """拍照：打断 → 说'茄子' → 拍照保存 → 发Telegram"""
        # 1. 打断服务端
        try:
            abort = {"session_id": self.session_id, "type": "abort", "reason": "photo"}
            await self.ws.send(json.dumps(abort))
        except Exception:
            pass

        # 2. 停止播放
        self.is_speaking = False
        if self._play_proc:
            try: self._play_proc.stdin.close()
            except Exception: pass
            try: self._play_proc.kill(); self._play_proc.wait(timeout=1)
            except Exception: pass
            self._play_proc = None

        await asyncio.sleep(0.1)
        self._mute = False

        # 3. 说"茄子！"
        self._stt_ignore_until = time.time() + 10
        detect_msg = {
            "session_id": self.session_id,
            "type": "listen", "state": "detect",
            "text": "请只回复：三、二、一，茄子！",
        }
        try:
            await self.ws.send(json.dumps(detect_msg))
        except Exception:
            pass

        # 4. 等TTS说完再拍
        await asyncio.sleep(3)

        # 5. 后台拍照+发送
        threading.Thread(target=self._take_photo_work, daemon=True).start()

    def _take_photo_work(self):
        """后台：抓帧保存+发Telegram"""
        import datetime
        global latest_frame
        touch_activity()

        frame = latest_frame
        if frame is None:
            add_log("ERROR", "📸 拍照失败：摄像头未开启")
            _xiaozhi_speak("拍照失败了，摄像头没开呢")
            return

        # 保存到 photos/YYYY-MM-DD/
        today = datetime.date.today().strftime("%Y-%m-%d")
        photo_dir = os.path.join(PHOTO_DIR, today)
        os.makedirs(photo_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%H%M%S")
        filename = f"photo_{ts}.jpg"
        filepath = os.path.join(photo_dir, filename)
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        add_log("INFO", f"📸 照片已保存: {filepath}")

        # 发 Telegram
        try:
            with open(filepath, 'rb') as f:
                resp = requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                    data={"chat_id": TELEGRAM_CHAT_ID, "caption": f"📸 多多拍照 {today} {ts}"},
                    files={"photo": (filename, f, "image/jpeg")},
                    timeout=15,
                )
            if resp.status_code == 200:
                add_log("INFO", "📸 照片已发送到 Telegram")
                _xiaozhi_speak("照片拍好了，已经发到你手机上啦")
            else:
                add_log("ERROR", f"📸 Telegram 发送失败: {resp.status_code}")
                _xiaozhi_speak("照片拍好了，但是发送失败了")
        except Exception as e:
            add_log("ERROR", f"📸 Telegram 发送异常: {e}")
            _xiaozhi_speak("照片拍好了，但是发送出了点问题")

    async def _do_fake_sleep(self):
        """假睡：打断对话 → 播'呼呼呼' → 低头 → 关摄像头"""
        add_log("INFO", "😴 多多准备假睡...")

        # 1. 打断服务端 LLM
        self._mute = True
        try:
            abort = {"session_id": self.session_id, "type": "abort", "reason": "sleep"}
            await self.ws.send(json.dumps(abort))
        except Exception:
            pass

        # 2. 停止播放
        self.is_speaking = False
        if self._play_proc:
            try: self._play_proc.stdin.close()
            except Exception: pass
            try: self._play_proc.kill(); self._play_proc.wait(timeout=1)
            except Exception: pass
            self._play_proc = None

        await asyncio.sleep(0.1)
        self._mute = False

        # 3. 播"呼呼呼"睡觉声
        self._stt_ignore_until = time.time() + 8
        detect_msg = {
            "session_id": self.session_id,
            "type": "listen", "state": "detect",
            "text": "请只回复：呼～呼～呼～",
        }
        try:
            await self.ws.send(json.dumps(detect_msg))
        except Exception:
            pass

        # 4. 等TTS播完再低头+关摄像头
        await asyncio.sleep(3)
        add_log("INFO", "😴 多多假睡中（低头+关摄像头）")
        if self.gimbal and getattr(self.gimbal, 'connected', False):
            self.gimbal.move_to(0, -30, speed=3, acc=1)
        _camera_wake_event.clear()

    async def _abort_and_vision(self, user_text: str):
        """打断服务端LLM → 说'让我看看' → 视觉识别 → 说结果"""
        # 1. 打断服务端
        try:
            abort = {"session_id": self.session_id, "type": "abort", "reason": "vision_request"}
            await self.ws.send(json.dumps(abort))
        except Exception:
            pass

        # 2. 停止播放
        self.is_speaking = False
        if self._play_proc:
            try:
                self._play_proc.stdin.close()
            except Exception:
                pass
            try:
                self._play_proc.kill()
                self._play_proc.wait(timeout=1)
            except Exception:
                pass
            self._play_proc = None

        await asyncio.sleep(0.1)
        self._mute = False

        # 3. 后台线程：先说"让我看看" → 视觉识别 → 说结果（全部用本地edge-tts）
        self._stt_ignore_until = time.time() + 30
        threading.Thread(target=self._do_vision_work, args=(user_text,), daemon=True).start()

    def _do_vision_work(self, user_text: str):
        """后台线程：先说让我看看 → 抓帧 → vision → 说结果（全部本地edge-tts）"""
        add_log("INFO", "👁️ 让我看看...")
        edge_tts_speak("让我看看")
        add_log("INFO", "📸 抓取画面...")
        touch_activity()
        time.sleep(0.3)
        prompt = f"用户说：\'{user_text}\'。请用简短的中文描述你从摄像头看到的画面，像跟小朋友说话一样，不超过3句话。"
        result = _vision_describe(prompt)
        add_log("INFO", f"👁️ 识别结果: {result}")
        if result and "识别出了点问题" not in result and "说不出来" not in result:
            edge_tts_speak(result)
        elif result:
            add_log("WARN", "👁️ 视觉识别失败，不播报")

    async def announce_online(self):
        if not self.connected:
            return
        detect = {
            "session_id": self.session_id,
            "type": "listen", "state": "detect",
            "text": "请只回复：多多启动了",
        }
        await self.ws.send(json.dumps(detect))

    async def on_wake_word(self):
        if not self.connected:
            log.warning("多多未连接，忽略唤醒")
            return

        add_log("INFO", f"🎙️ 唤醒词触发，打断并开始对话")

        # 1. 立即静音，丢弃残余音频帧
        self._mute = True

        # 2. 通知服务端打断
        abort = {"session_id": self.session_id, "type": "abort", "reason": "wake_word_detected"}
        try:
            await self.ws.send(json.dumps(abort))
        except Exception:
            pass

        # 3. 停止播放，彻底释放音频设备
        self.is_speaking = False
        if self._play_proc:
            try:
                self._play_proc.stdin.close()
            except Exception:
                pass
            try:
                self._play_proc.kill()
                self._play_proc.wait(timeout=2)
            except Exception:
                pass
            self._play_proc = None

        # 4. 停止录音
        if self.is_listening:
            self._stop_recording()
            if self._send_task:
                self._send_task.cancel()
            try:
                stop = {"session_id": self.session_id, "type": "listen", "state": "stop"}
                await self.ws.send(json.dumps(stop))
            except Exception:
                pass

        # 5. 等一下确保设备完全释放
        await asyncio.sleep(0.1)

        # 云台点头
        if self.gimbal and getattr(self.gimbal, 'connected', False):
            threading.Thread(target=gimbal_online_nod, args=(self.gimbal,), daemon=True).start()
            # 头灯微亮表示听到
            from emotions import _light, _light_off, LIGHT_DIM
            _light(self.gimbal, 0, LIGHT_DIM)
            threading.Timer(1.0, _light_off, args=(self.gimbal,)).start()

        # 直接开始录音（点头已在上方触发）
        start = {
            "session_id": self.session_id,
            "type": "listen", "state": "start", "mode": "auto",
        }
        await self.ws.send(json.dumps(start))
        self.is_listening = True
        self._audio_count = 0  # 重置音频帧计数
        self._send_task = asyncio.create_task(self._record_and_send())

    async def _record_and_send(self):
        frame_bytes = FRAME_SIZE * 2
        add_log("INFO", "🎙️ 启动录音 arecord...")
        self._rec_proc = subprocess.Popen(
            ["arecord", "-D", AUDIO_REC, "-f", "S16_LE",
             "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw", "-q"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        loop = asyncio.get_event_loop()
        warmup_frames = 12
        sent_frames = 0
        try:
            while self.is_listening and self.connected:
                data = await loop.run_in_executor(None, self._rec_proc.stdout.read, frame_bytes)
                if not data:
                    # arecord 可能启动失败
                    if self._rec_proc.poll() is not None:
                        stderr = self._rec_proc.stderr.read(500).decode(errors='ignore')
                        add_log("ERROR", f"arecord 退出: {stderr}")
                        break
                    continue
                if len(data) == frame_bytes:
                    if warmup_frames > 0:
                        warmup_frames -= 1
                        continue
                    opus = pcm_to_opus(data)
                    await self.ws.send(opus)
                    sent_frames += 1
                    if sent_frames == 1:
                        add_log("INFO", "🎙️ 录音中，开始发送音频...")
        except Exception as e:
            log.error(f"录音发送错误: {e}")
        finally:
            self._stop_recording()

    def _stop_recording(self):
        self.is_listening = False
        if self._rec_proc:
            try:
                self._rec_proc.terminate()
            except Exception:
                pass
            self._rec_proc = None

    async def stop_listening(self):
        self._stop_recording()
        if self._send_task:
            self._send_task.cancel()
        stop = {"session_id": self.session_id, "type": "listen", "state": "stop"}
        try:
            await self.ws.send(json.dumps(stop))
        except Exception:
            pass
        add_log("INFO", "🎙️ 停止录音")


# ============================================================
#  多多后台运行 (asyncio 线程)
# ============================================================
_xiaozhi_client: XiaozhiClient = None
_xiaozhi_loop: asyncio.AbstractEventLoop = None


def _xiaozhi_speak(text: str):
    """通过多多服务端 TTS 播放文字"""
    if not _xiaozhi_client or not _xiaozhi_client.connected or not _xiaozhi_loop:
        return
    _xiaozhi_client._stt_ignore_until = time.time() + 10
    import json as _json
    detect = {
        "session_id": _xiaozhi_client.session_id,
        "type": "listen", "state": "detect",
        "text": f"请一字不差地复述以下内容，不要添加任何其他文字：{text}",
    }

    async def _send():
        try:
            await _xiaozhi_client.ws.send(_json.dumps(detect))
        except Exception as e:
            log.error(f"多多 TTS 发送失败: {e}")

    asyncio.run_coroutine_threadsafe(_send(), _xiaozhi_loop)

def start_xiaozhi_thread(gimbal, ws_url=None):
    """在后台线程中启动多多对话客户端"""
    global _xiaozhi_loop
    _xiaozhi_loop = asyncio.new_event_loop()
    url = ws_url or XIAOZHI_WS_URL

    def run():
        asyncio.set_event_loop(_xiaozhi_loop)
        _xiaozhi_loop.run_until_complete(_xiaozhi_main(gimbal, url))

    t = threading.Thread(target=run, daemon=True, name="xiaozhi")
    t.start()
    return t

async def _xiaozhi_main(gimbal, ws_url):
    global _xiaozhi_client
    first_connect = True

    while True:
        client = XiaozhiClient(ws_url, XIAOZHI_DEVICE_ID, gimbal=gimbal)
        _xiaozhi_client = client

        if not await client.connect():
            add_log("ERROR", "多多连接失败，3秒后重试")
            await asyncio.sleep(3)
            continue

        if first_connect:
            await client.announce_online()
            first_connect = False

        await asyncio.sleep(0.5)
        while client.is_speaking:
            await asyncio.sleep(0.2)

        # 唤醒词监听 — M2 硬件唤醒（随时可打断）
        listeners = []

        try:
            m2_listener = M2WakeWordListener(should_pause_fn=None)
            listeners.append(m2_listener)
            add_log("INFO", f"👂 M2 硬件唤醒词监听已启动")
        except Exception as e:
            log.warning(f"M2 硬件唤醒不可用: {e}")

        loop = asyncio.get_event_loop()

        def on_wake():
            # M2 唤醒随时可触发，不暂停监听，直接打断并重新开始
            touch_activity()
            asyncio.run_coroutine_threadsafe(client.on_wake_word(), loop)

        for l in listeners:
            l.start(on_wake)

        await client.message_loop()

        for l in listeners:
            l.stop()
        _xiaozhi_client = None

        add_log("WARN", "多多断线，3秒后重连")
        await asyncio.sleep(3)


# ============================================================
#  语音问候 (通过多多 TTS)
# ============================================================
class VoiceGreeter:
    def __init__(self, cooldown: float = GREET_COOLDOWN):
        self.cooldown = cooldown
        self.last_greet_time: dict[str, float] = {}
        self.available = True
        log.info("[语音] 问候通过多多 TTS")

    def should_greet(self, name: str) -> bool:
        if name == "unknown":
            return False
        last = self.last_greet_time.get(name, 0)
        return time.time() - last > self.cooldown

    def greet(self, name: str):
        if not self.should_greet(name):
            return
        # 多多正在说话/录音时不打断
        if _xiaozhi_client and (_xiaozhi_client.is_speaking or _xiaozhi_client.is_listening):
            return
        self.last_greet_time[name] = time.time()
        msg = GREET_MESSAGES.get(name, GREET_DEFAULT)
        add_log("INFO", f"🔊 语音问候: {name} → {msg}")
        self.speak(msg)

    def speak(self, text: str):
        """通过多多 TTS 播放，如果多多未连接则回退 espeak"""
        if _xiaozhi_client and _xiaozhi_client.connected:
            _xiaozhi_speak(text)
        else:
            speak_async(text)

    def check_faces(self, faces: list[dict]):
        for face in faces:
            name = face.get("name", "unknown")
            if name != "unknown":
                self.greet(name)


# ============================================================
#  舵机控制器
# ============================================================
class GimbalController:
    def __init__(self, port: str = DEFAULT_SERIAL, baud: int = DEFAULT_BAUD):
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            self.connected = True
            print(f"[舵机] 已连接: {port} @ {baud}")
        except Exception as e:
            self.ser = None
            self.connected = False
            print(f"[舵机] 连接失败: {e}")

        self.pan_angle = 0.0
        self.tilt_angle = 0.0
        self.lock = threading.Lock()

        if self.connected:
            threading.Thread(target=self._read_feedback, daemon=True).start()

    def _read_feedback(self):
        while self.connected:
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline()
                    if line:
                        data = json.loads(line.decode('utf-8', errors='replace').strip())
                        if data.get('T') == 1001 and 'pan' in data and 'tilt' in data:
                            with self.lock:
                                self.pan_angle = data['pan']
                                self.tilt_angle = data['tilt']
                else:
                    time.sleep(0.02)
            except Exception:
                time.sleep(0.05)

    def send_command(self, data: dict):
        if not self.connected:
            return
        try:
            cmd = json.dumps(data) + "\n"
            self.ser.write(cmd.encode("utf-8"))
        except Exception as e:
            print(f"[舵机] 发送失败: {e}")

    def move_to(self, pan: float, tilt: float, speed: int = 10, acc: int = 1):
        with self.lock:
            self.pan_angle = max(PAN_MIN, min(PAN_MAX, pan))
            self.tilt_angle = max(TILT_MIN, min(TILT_MAX, tilt))
            self.send_command({
                "T": CMD_GIMBAL,
                "X": self.pan_angle, "Y": self.tilt_angle,
                "SPD": speed, "ACC": acc,
            })

    def track_target(self, frame_cx, frame_cy, target_x, target_y,
                     iterate=TRACK_ITERATE):
        distance = math.sqrt((target_x - frame_cx) ** 2 + (frame_cy - target_y) ** 2)
        with self.lock:
            self.pan_angle += (target_x - frame_cx) * iterate
            self.tilt_angle += (frame_cy - target_y) * iterate
            self.pan_angle = max(PAN_MIN, min(PAN_MAX, self.pan_angle))
            self.tilt_angle = max(TILT_MIN, min(TILT_MAX, self.tilt_angle))
            spd = max(1, int(distance * TRACK_SPD_RATE / 100))
            acc = max(1, int(distance * TRACK_ACC_RATE))
            self.send_command({
                "T": CMD_GIMBAL,
                "X": self.pan_angle, "Y": self.tilt_angle,
                "SPD": spd, "ACC": acc,
            })
        return distance

    def center(self):
        self.move_to(0, 0, speed=20, acc=5)

    def close(self):
        if self.ser:
            self.ser.close()


# 舵机表情动作
_motion_lock = threading.Lock()
_last_happy_ts = 0.0
_last_happy_day = ""
_last_online_ts = 0.0

def gimbal_happy_swing(gimbal, amp=6.0, step_delay=0.18):
    if not getattr(gimbal, "connected", False):
        return
    with _motion_lock:
        base_pan = gimbal.pan_angle
        base_tilt = gimbal.tilt_angle
        for dx in [amp, -amp, amp * 0.6, -amp * 0.6, 0]:
            gimbal.move_to(base_pan + dx, base_tilt, speed=12, acc=3)
            time.sleep(step_delay)

def gimbal_online_nod(gimbal, amp=25.0, step_delay=0.25):
    if not getattr(gimbal, "connected", False):
        return
    with _motion_lock:
        base_pan = gimbal.pan_angle
        base_tilt = gimbal.tilt_angle
        for dy in [amp, -amp * 0.6, amp * 0.4, 0]:
            gimbal.move_to(base_pan, base_tilt + dy, speed=10, acc=2)
            time.sleep(step_delay)


# ============================================================
#  人脸跟踪器
# ============================================================
class FaceTracker:
    def __init__(self, api_url, gimbal, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
        self.api_url = api_url
        self.gimbal = gimbal
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2
        self.tracking_name = None
        self.tracking_confidence = 0.0
        self.last_seen_time = 0.0
        self.lost_timeout = 3.0

    def select_target(self, faces):
        if not faces:
            return None
        known_faces = [f for f in faces if f.get("name", "unknown") != "unknown"]
        if not known_faces:
            return None
        for priority_name in PRIORITY_NAMES:
            candidates = [f for f in known_faces if f["name"] == priority_name]
            if candidates:
                return max(candidates, key=lambda f: f.get("confidence", 0))
        return max(known_faces, key=lambda f: f.get("confidence", 0))

    def get_face_center(self, face):
        bbox = face.get("bbox", [0, 0, 0, 0])
        return (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2

    def update(self, faces):
        target = self.select_target(faces)
        if target:
            self.tracking_name = target["name"]
            self.tracking_confidence = target.get("confidence", 0)
            self.last_seen_time = time.time()
            tx, ty = self.get_face_center(target)
            self.gimbal.track_target(self.center_x, self.center_y, tx, ty)
        else:
            if self.tracking_name and time.time() - self.last_seen_time > self.lost_timeout:
                self.gimbal.center()
                self.tracking_name = None


# ============================================================
#  摄像头 + 主循环
# ============================================================
from flask import Flask, Response, jsonify, send_from_directory, request

latest_frame = None
latest_raw_frame = None  # 原始帧（不带标注，给视觉识别用）
latest_results = []
tracker_status = {}
is_running = True
lock = threading.Lock()

# 摄像头空闲休眠
last_activity_time = time.time()
camera_sleeping = False
_camera_wake_event = threading.Event()

def touch_activity():
    """刷新活动时间，唤醒摄像头"""
    global last_activity_time, camera_sleeping
    last_activity_time = time.time()
    if camera_sleeping:
        camera_sleeping = False
        _camera_wake_event.set()
        add_log("INFO", "📷 唤醒摄像头")


flask_app = Flask(__name__, static_folder="static")


def draw_tracking_results(frame, faces, tracking_name, gesture=None):
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
            color = (0, 255, 255); thickness = 3
        elif name != "unknown":
            color = (0, 255, 0); thickness = 2
        else:
            color = (0, 0, 255); thickness = 1
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        label = f"{'>>> ' if is_target else ''}{name} ({conf:.2f})"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - h - 8), (x1 + w, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    h_frame, w_frame = annotated.shape[:2]
    if gesture:
        gname = gesture.get("gesture", "none")
        hands = gesture.get("hands_count", 0)
        color = (0, 255, 255) if gname == "open_palm" else (255, 200, 0)
        text = f"Gesture: {gname} ({hands} hand)"
        cv2.putText(annotated, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        for hi in gesture.get("hand_landmarks", []):
            bbox = hi.get("bbox")
            if bbox and len(bbox) == 4:
                x1 = int(bbox[0] * w_frame)
                y1 = int(bbox[1] * h_frame)
                x2 = int(bbox[2] * w_frame)
                y2 = int(bbox[3] * h_frame)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, hi.get("gesture", gname), (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cx, cy = w_frame // 2, h_frame // 2
    cv2.line(annotated, (cx - 15, cy), (cx + 15, cy), (255, 255, 255), 1)
    cv2.line(annotated, (cx, cy - 15), (cx, cy + 15), (255, 255, 255), 1)
    return annotated


def open_camera(camera_id, width, height):
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
    add_log("ERROR", "无法打开摄像头")
    return (None, None)


def read_frame(cam_type, cam_obj):
    try:
        ret, frame = cam_obj.read()
        return frame if ret else None
    except Exception:
        return None


def close_camera(cam_type, cam_obj):
    try:
        cam_obj.release()
    except Exception:
        pass


def make_placeholder_frame(width, height, text="摄像头未连接"):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)
    cv2.putText(frame, text, ((width - tw) // 2, (height + th) // 2), font, 0.7, (0, 0, 255), 2)
    return frame


def camera_tracking_loop(api_url, camera_id, width, height, fps_limit, gimbal, greeter, gesture_det):
    global latest_frame, latest_raw_frame, latest_results, tracker_status, is_running
    global _last_happy_ts, _last_happy_day
    global last_activity_time, camera_sleeping

    tracker = FaceTracker(api_url, gimbal, width, height)
    cam_type, cam_obj = open_camera(camera_id, width, height)
    retry_interval = 5
    last_retry = 0
    frame_interval = 1.0 / fps_limit
    last_send = 0
    frame_count = 0
    api_ok_count = 0
    api_err_count = 0
    read_fail_count = 0

    while is_running:
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
            if read_fail_count > 30:
                add_log("ERROR", f"连续 {read_fail_count} 次读取失败，重新打开摄像头")
                close_camera(cam_type, cam_obj)
                cam_type, cam_obj = None, None
                read_fail_count = 0
            time.sleep(0.1)
            continue

        read_fail_count = 0
        frame_count += 1
        now = time.time()

        # 空闲休眠检查：5分钟无人脸+无交互 → 关闭摄像头
        if not camera_sleeping and (now - last_activity_time > IDLE_CAMERA_TIMEOUT):
            add_log("INFO", "😴 30分钟无活动，关闭摄像头进入休眠")
            # 舵机低头60°，模拟睡觉
            if gimbal_instance and getattr(gimbal_instance, 'connected', False):
                gimbal_instance.move_to(0, -30, speed=5, acc=1)
            close_camera(cam_type, cam_obj)
            cam_type, cam_obj = None, None
            camera_sleeping = True
            with lock:
                latest_frame = make_placeholder_frame(width, height, "Sleeping - Say wake word to activate")
                latest_results = []
            _camera_wake_event.clear()
            _camera_wake_event.wait()  # 阻塞等待唤醒
            add_log("INFO", "📷 摄像头唤醒中...")
            # 舵机抬头复位
            if gimbal_instance and getattr(gimbal_instance, 'connected', False):
                gimbal_instance.center()
            cam_type, cam_obj = open_camera(camera_id, width, height)
            last_retry = time.time()
            continue

        gesture = None

        if now - last_send < frame_interval:
            with lock:
                latest_raw_frame = frame.copy()
                latest_frame = draw_tracking_results(frame, latest_results, tracker.tracking_name, gesture)
                tracker_status["gesture"] = gesture
            continue

        last_send = now
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        try:
            resp = requests.post(
                f"{api_url}/recognize",
                files={"file": ("frame.jpg", jpeg.tobytes(), "image/jpeg")},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                faces = data.get("faces", [])
                api_ok_count += 1
                if api_ok_count == 1:
                    add_log("INFO", f"API 首次响应成功，检测到 {len(faces)} 张脸")
                tracker.update(faces)
                greeter.check_faces(faces)

                if any(f.get("name") == "son" for f in faces):
                    today = time.strftime("%Y-%m-%d")
                    if _last_happy_day != today:
                        _last_happy_day = today
                        _last_happy_ts = time.time()
                        threading.Thread(target=gimbal_happy_swing, args=(gimbal,), daemon=True).start()
                        add_log("INFO", "😊 看到小虎，今日首次开心摆动")

                # 有人脸时刷新活动时间
                if faces:
                    touch_activity()

                with lock:
                    latest_results = faces
                    latest_raw_frame = frame.copy()
                    latest_frame = draw_tracking_results(frame, faces, tracker.tracking_name, gesture)
                    # 多多状态
                    xz_status = "未连接"
                    if _xiaozhi_client:
                        if _xiaozhi_client.is_speaking:
                            xz_status = "说话中"
                        elif _xiaozhi_client.is_listening:
                            xz_status = "录音中"
                        elif _xiaozhi_client.connected:
                            xz_status = "待命"
                    tracker_status = {
                        "tracking": tracker.tracking_name,
                        "confidence": round(tracker.tracking_confidence, 3),
                        "pan": round(gimbal.pan_angle - PAN_OFFSET, 1),
                        "tilt": round(gimbal.tilt_angle, 1),
                        "faces_count": len(faces),
                        "known_count": len([f for f in faces if f.get("name") != "unknown"]),
                        "frame_count": frame_count,
                        "api_ok": api_ok_count,
                        "api_err": api_err_count,
                        "greet_history": {k: time.strftime("%H:%M:%S", time.localtime(v))
                                          for k, v in greeter.last_greet_time.items()},
                        "gesture": gesture,
                        "xiaozhi": xz_status,
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
                latest_raw_frame = frame.copy()
                latest_frame = draw_tracking_results(frame, [], None, gesture)

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
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(0.033)


@flask_app.route("/")
def index():
    return send_from_directory("static", "tracker.html")

@flask_app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

@flask_app.route("/api/status")
def api_status():
    with lock:
        return jsonify({**tracker_status, "faces": latest_results.copy(), "running": is_running})

@flask_app.route("/api/logs")
def api_logs():
    return jsonify(list(log_buffer))

@flask_app.route("/api/gimbal/center", methods=["POST"])
def gimbal_center():
    gimbal_instance.center()
    return jsonify({"ok": True})

@flask_app.route("/api/gimbal/calibrate", methods=["POST"])
def gimbal_calibrate():
    with gimbal_instance.lock:
        gimbal_instance.pan_angle = 0.0
        gimbal_instance.tilt_angle = 0.0
    add_log("INFO", "🔧 舵机校准：当前位置已标记为 0°")
    return jsonify({"ok": True, "pan": 0, "tilt": 0})

@flask_app.route("/api/gimbal/express", methods=["POST"])
def gimbal_express():
    global _last_happy_ts, _last_online_ts
    data = request.get_json(silent=True) or {}
    action = (data.get("action") or "").strip()
    if action == "happy":
        if time.time() - _last_happy_ts > 1.0:
            _last_happy_ts = time.time()
            threading.Thread(target=gimbal_happy_swing, args=(gimbal_instance,), daemon=True).start()
        return jsonify({"ok": True, "action": "happy"})
    if action == "online":
        if time.time() - _last_online_ts > 1.0:
            _last_online_ts = time.time()
            threading.Thread(target=gimbal_online_nod, args=(gimbal_instance,), daemon=True).start()
        return jsonify({"ok": True, "action": "online"})
    return jsonify({"ok": False, "error": "invalid action"}), 400

@flask_app.route("/api/volume", methods=["GET"])
def get_volume():
    try:
        out = subprocess.check_output(["amixer", "-c", "3", "get", "Speaker"],
                                      stderr=subprocess.DEVNULL).decode()
        for line in out.split("\n"):
            if "%" in line:
                pct = int(line.split("[")[1].split("%")[0])
                return jsonify({"volume": pct})
    except Exception:
        pass
    return jsonify({"volume": -1})

@flask_app.route("/api/volume", methods=["POST"])
def set_volume():
    data = request.get_json() or {}
    vol = max(0, min(100, int(data.get("volume", 50))))
    try:
        subprocess.run(["amixer", "-c", "3", "set", "Speaker", f"{vol}%"],
                       capture_output=True, timeout=5)
        add_log("INFO", f"🔊 音量设置: {vol}%")
        return jsonify({"ok": True, "volume": vol})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@flask_app.route("/api/xiaozhi/status")
def xiaozhi_status():
    """多多对话状态"""
    if _xiaozhi_client:
        return jsonify({
            "connected": _xiaozhi_client.connected,
            "speaking": _xiaozhi_client.is_speaking,
            "listening": _xiaozhi_client.is_listening,
            "session": _xiaozhi_client.session_id,
        })
    return jsonify({"connected": False})

@flask_app.route("/api/m2/state")
def m2_api_state():
    """M2 麦克风阵列状态（心跳、唤醒事件、角度）"""
    return jsonify(m2_state)

@flask_app.route("/api/photo", methods=["POST"])
def api_take_photo():
    """网页端触发拍照"""
    import datetime
    global latest_frame
    touch_activity()
    frame = latest_frame
    if frame is None:
        return jsonify({"ok": False, "error": "摄像头未开启"})
    today = datetime.date.today().strftime("%Y-%m-%d")
    photo_dir = os.path.join(PHOTO_DIR, today)
    os.makedirs(photo_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%H%M%S")
    filename = f"photo_{ts}.jpg"
    filepath = os.path.join(photo_dir, filename)
    cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    add_log("INFO", f"📸 照片已保存: {filepath}")
    # 发 Telegram
    try:
        with open(filepath, 'rb') as f:
            resp = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": f"📸 拍照 {today} {ts}"},
                files={"photo": (filename, f, "image/jpeg")},
                timeout=15,
            )
        if resp.status_code == 200:
            add_log("INFO", "📸 照片已发送到 Telegram")
            return jsonify({"ok": True, "file": filepath})
        else:
            return jsonify({"ok": True, "file": filepath, "telegram": False})
    except Exception as e:
        return jsonify({"ok": True, "file": filepath, "telegram": False, "error": str(e)})

@flask_app.route("/api/xiaozhi/logs")
def xiaozhi_logs():
    """多多对话日志（从 log_buffer 过滤）"""
    keywords = ["多多", "唤醒", "🎯", "🎙", "🔊", "💬", "🤖", "🎤", "录音",
                "asr:", "asr final:", "M2", "sherpa", "连接", "断线", "对话"]
    lines = []
    for entry in log_buffer:
        msg = entry.get("msg", "")
        if any(k in msg for k in keywords):
            lines.append(f"{entry['time']} [{entry['level']}] {msg}")
    # 页面期望 {"lines": [...]}，最多30条，倒序变正序
    lines = lines[:30]
    lines.reverse()  # newest first
    return jsonify({"lines": lines})


# ============================================================
#  入口
# ============================================================
gimbal_instance: GimbalController = None

def main():
    global is_running, gimbal_instance

    parser = argparse.ArgumentParser(description="家庭人脸跟踪 + 多多语音对话")
    parser.add_argument("--api", default=DEFAULT_API_URL)
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS_LIMIT)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--serial", default=DEFAULT_SERIAL)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--no-gimbal", action="store_true")
    parser.add_argument("--no-xiaozhi", action="store_true", help="禁用多多对话")
    parser.add_argument("--xiaozhi-ws", default=XIAOZHI_WS_URL)
    args = parser.parse_args()

    xiaozhi_ws = args.xiaozhi_ws

    print(f"{'='*50}")
    print(f"  家庭人脸跟踪 + 多多语音对话")
    print(f"  API: {args.api}")
    print(f"  摄像头: {args.camera} ({args.width}x{args.height})")
    print(f"  舵机: {args.serial} ({'禁用' if args.no_gimbal else '启用'})")
    print(f"  多多: {'禁用' if args.no_xiaozhi else xiaozhi_ws}")
    print(f"  跟踪优先级: {' > '.join(PRIORITY_NAMES)}")
    print(f"  Web: http://0.0.0.0:{args.port}")
    print(f"{'='*50}")

    # 舵机
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

    # 语音
    greeter_instance = VoiceGreeter(cooldown=GREET_COOLDOWN)

    gesture_instance = None

    # 多多对话线程（启动语音由多多 announce_online 播报）
    if not args.no_xiaozhi and OPUS_OK:
        start_xiaozhi_thread(gimbal_instance, ws_url=xiaozhi_ws)
        add_log("INFO", "🤖 多多对话线程已启动")
    elif not OPUS_OK:
        add_log("WARN", "opuslib_next 未安装，多多对话禁用")

    # 摄像头+跟踪线程
    cam_thread = threading.Thread(
        target=camera_tracking_loop,
        args=(args.api, args.camera, args.width, args.height, args.fps,
              gimbal_instance, greeter_instance, gesture_instance),
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
