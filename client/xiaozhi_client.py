#!/usr/bin/env python3
"""
小智语音对话客户端 — 运行在老三(树莓派)上
唤醒词: "悟空悟空" (sherpa-onnx KeywordSpotter)
保持 WebSocket 长连接，检测唤醒词后开始录音对话

参考: py-xiaozhi 项目协议实现
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
import requests
# import uuid

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("xiaozhi")

# ============================================================
#  配置
# ============================================================
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION_MS = 60
FRAME_SIZE = SAMPLE_RATE * FRAME_DURATION_MS // 1000  # 960
AUDIO_PLAY = "plughw:3,0"
AUDIO_REC = "plughw:2,0"
WAKE_WORD = "多多"
SHERPA_ASR_DIR = os.path.join(os.path.dirname(__file__), "models", "sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16", "96")

def _contains_wake(text):
    """唤醒词匹配：多多（容错）"""
    t = "".join(ch for ch in text if not ch.isspace())
    if "多多" in t:
        return True
    # 容错：多/哆 连续两次
    for i in range(len(t) - 1):
        if t[i] in ("多", "哆") and t[i + 1] in ("多", "哆"):
            return True
    return False

# ============================================================
#  Opus
# ============================================================
import opuslib_next as opuslib
_encoder = opuslib.Encoder(SAMPLE_RATE, CHANNELS, opuslib.APPLICATION_VOIP)
_decoder = opuslib.Decoder(SAMPLE_RATE, CHANNELS)

def pcm_to_opus(pcm: bytes) -> bytes:
    return _encoder.encode(pcm, FRAME_SIZE)

def opus_to_pcm(data: bytes) -> bytes:
    return _decoder.decode(data, FRAME_SIZE)

# ============================================================
#  本地 TTS
# ============================================================
def speak(text: str):
    subprocess.run(
        f'espeak -v zh -s 320 --stdout "{text}" | aplay -D {AUDIO_PLAY} -q',
        shell=True, stderr=subprocess.DEVNULL
    )

def speak_async(text: str):
    threading.Thread(target=speak, args=(text,), daemon=True).start()

# ============================================================
#  唤醒词检测 (sherpa-onnx 流式ASR + 模糊匹配)
# ============================================================
class WakeWordListener:
    def __init__(self, device=AUDIO_REC):
        import sherpa_onnx
        import numpy as np
        self.sherpa_onnx = sherpa_onnx
        self.np = np
        self.device = device
        self.paused = False
        self.active = True
        self._proc = None
        self._cooldown = 2.0
        self._last_detect = 0

        encoder = os.path.join(SHERPA_ASR_DIR, "encoder-epoch-99-avg-1.onnx")
        decoder = os.path.join(SHERPA_ASR_DIR, "decoder-epoch-99-avg-1.onnx")
        joiner = os.path.join(SHERPA_ASR_DIR, "joiner-epoch-99-avg-1.onnx")
        tokens = os.path.join(SHERPA_ASR_DIR, "tokens.txt")

        log.info(f"加载 sherpa-onnx 流式ASR: {SHERPA_ASR_DIR}")
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
        chunk_samples = int(SAMPLE_RATE * 0.1)  # 100ms
        chunk_bytes = chunk_samples * 2
        read_count = 0
        last_text = ""

        while self.active:
            # 等待非暂停状态
            if self.paused:
                time.sleep(0.1)
                continue

            # 启动/重启 arecord
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

            read_count += 1
            if read_count % 300 == 0:
                log.info(f"[debug] audio chunks: {read_count}, paused={self.paused}")

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
                if text:
                    log.info(f"asr final: {text}")
                    if _contains_wake(text):
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
        # 停掉 arecord 释放设备，让录音对话可以用
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
#  小智客户端 (长连接)
# ============================================================
class PalmGestureListener:
    """通过 face_tracker 的 /api/status 轮询手掌状态：出现开始录音，消失结束录音"""

    def __init__(self, status_url="http://127.0.0.1:5000/api/status", interval=0.2, hold_seconds=0.8):
        self.status_url = status_url
        self.interval = interval
        self.hold_seconds = hold_seconds
        self.active = True
        self._last_palm_ts = 0.0
        self._recording = False

    def start(self, on_palm_start, on_palm_end):
        def _run():
            log.info("✋ 手掌触发模式已启用（张手开始，放下结束）")
            while self.active:
                palm = False
                try:
                    r = requests.get(self.status_url, timeout=0.5)
                    if r.ok:
                        st = r.json()
                        g = (st.get("gesture") or {}).get("gesture", "none")
                        palm = (g == "open_palm")
                except Exception:
                    pass

                now = time.time()
                if palm:
                    self._last_palm_ts = now
                    if not self._recording:
                        self._recording = True
                        on_palm_start()
                else:
                    # 手掌消失持续一段时间后停止，避免抖动
                    if self._recording and (now - self._last_palm_ts) > self.hold_seconds:
                        self._recording = False
                        on_palm_end()

                time.sleep(self.interval)

        threading.Thread(target=_run, daemon=True).start()

    def stop(self):
        self.active = False


class XiaozhiClient:
    def __init__(self, ws_url: str, device_id: str):
        self.ws_url = ws_url
        self.device_id = device_id
        self.session_id = None
        self.ws = None
        self.connected = False
        self.is_speaking = False  # 服务端在说话
        self.is_listening = False  # 正在录音
        self._rec_proc = None
        self._play_proc = None
        self._loop = None
        self._send_task = None

    async def connect(self):
        """建立长连接"""
        import websockets
        headers = {
            "Device-Id": self.device_id,
            "Client-Id": self.device_id,
            "Protocol-Version": "1",
        }
        log.info(f"🔗 连接 {self.ws_url}")
        try:
            self.ws = await websockets.connect(
                self.ws_url, max_size=None,
                additional_headers=headers,
                ping_interval=20, ping_timeout=20,
                close_timeout=10,
            )
        except TypeError:
            self.ws = await websockets.connect(
                self.ws_url, max_size=None,
                extra_headers=headers,
                ping_interval=20, ping_timeout=20,
                close_timeout=10,
            )
        # 发 hello
        hello = {
            "type": "hello",
            "version": 1,
            "transport": "websocket",
            "device_id": self.device_id,
            "device_name": "老三-树莓派",
            "features": {"mcp": False},
            "audio_params": {
                "format": "opus",
                "sample_rate": SAMPLE_RATE,
                "channels": CHANNELS,
                "frame_duration": FRAME_DURATION_MS,
            },
        }
        await self.ws.send(json.dumps(hello))
        log.info("📤 已发送 hello")

        # 等 hello 响应
        resp = await asyncio.wait_for(self.ws.recv(), timeout=10)
        try:
            data = json.loads(resp)
            self.session_id = data.get("session_id", "")
            log.info(f"✅ 连接成功, session: {self.session_id}")
            self.connected = True
        except Exception as e:
            log.error(f"握手失败: {e}, resp: {str(resp)[:200]}")
            return False
        return True

    async def message_loop(self):
        """持续接收消息"""
        try:
            async for message in self.ws:
                if isinstance(message, bytes):
                    # Opus 音频 → 解码播放
                    self._audio_count = getattr(self, '_audio_count', 0) + 1
                    if self._audio_count <= 3 or self._audio_count % 50 == 0:
                        log.info(f"🔈 收到音频帧 #{self._audio_count}, {len(message)} bytes")
                    try:
                        pcm = opus_to_pcm(message)
                        self._play_pcm(pcm)
                    except Exception as e:
                        log.error(f"音频解码/播放错误: {e}")
                else:
                    data = json.loads(message)
                    await self._handle(data)
        except Exception as e:
            log.error(f"连接断开: {e}")
            self.connected = False

    async def _handle(self, msg: dict):
        t = msg.get("type", "")
        if t == "tts":
            state = msg.get("state", "")
            if state == "start":
                self.is_speaking = True
                log.info("🔊 服务端开始说话")
            elif state == "sentence_start":
                log.info(f"💬 {msg.get('text', '')}")
            elif state == "stop":
                self.is_speaking = False
                log.info("🔊 服务端说话结束")
                # 停止录音，让唤醒词监听恢复
                if self.is_listening:
                    await self.stop_listening()
        elif t == "stt":
            log.info(f"🎤 识别: {msg.get('text', '')}")
        elif t == "llm":
            log.info(f"🤖 {msg.get('text', '')}")
        elif t == "hello":
            self.session_id = msg.get("session_id", self.session_id)
            log.info(f"hello 响应, session: {self.session_id}")

    def _play_pcm(self, pcm: bytes):
        """直接写入 aplay 进程"""
        try:
            if self._play_proc is None or self._play_proc.poll() is not None:
                log.info(f"🔊 启动 aplay 进程 (设备: {AUDIO_PLAY}, 采样率: {SAMPLE_RATE})")
                self._play_proc = subprocess.Popen(
                    ["aplay", "-D", AUDIO_PLAY, "-f", "S16_LE",
                     "-r", str(SAMPLE_RATE), "-c", "1", "-q"],
                    stdin=subprocess.PIPE, stderr=subprocess.PIPE
                )
            self._play_proc.stdin.write(pcm)
            self._play_proc.stdin.flush()
        except Exception as e:
            log.error(f"播放错误: {e}")
            if self._play_proc:
                err = self._play_proc.stderr.read(200) if self._play_proc.stderr else b""
                log.error(f"aplay stderr: {err}")
            self._play_proc = None

    async def on_wake_word(self):
        """唤醒词触发：开始一轮对话"""
        if not self.connected:
            log.warning("未连接，忽略唤醒")
            return
        if self.is_listening:
            return

        log.info(f"🎙️ 唤醒词触发，开始对话: {WAKE_WORD}")

        if self.is_speaking:
            abort = {"session_id": self.session_id, "type": "abort", "reason": "wake_word_detected"}
            await self.ws.send(json.dumps(abort))
            if self._play_proc:
                try:
                    self._play_proc.terminate()
                except Exception:
                    pass
                self._play_proc = None

        # 本地应答：收到唤醒词后先说“我在”
        speak_async("我在")

        # 关键修复：不发送 detect(text=唤醒词)，避免服务端把唤醒词当问题
        # 等本地“我在”播完再开始收指令
        await asyncio.sleep(0.9)

        start = {
            "session_id": self.session_id,
            "type": "listen",
            "state": "start",
            "mode": "auto",
        }
        await self.ws.send(json.dumps(start))

        self.is_listening = True
        self._send_task = asyncio.create_task(self._record_and_send())

    async def _record_and_send(self):
        """录音并通过 WebSocket 发送 Opus 帧"""
        frame_bytes = FRAME_SIZE * 2
        self._rec_proc = subprocess.Popen(
            ["arecord", "-D", AUDIO_REC, "-f", "S16_LE",
             "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw", "-q"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        loop = asyncio.get_event_loop()
        log.info("🎙️ 录音中...")
        # 丢弃起始约0.7秒，避免把唤醒词尾音当成用户指令
        warmup_frames = 12
        sent_frames = 0
        try:
            while self.is_listening and self.connected:
                data = await loop.run_in_executor(None, self._rec_proc.stdout.read, frame_bytes)
                if len(data) == frame_bytes:
                    if warmup_frames > 0:
                        warmup_frames -= 1
                        continue
                    opus = pcm_to_opus(data)
                    await self.ws.send(opus)
                    sent_frames += 1
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
        """停止录音"""
        self._stop_recording()
        if self._send_task:
            self._send_task.cancel()
        stop = {"session_id": self.session_id, "type": "listen", "state": "stop"}
        try:
            await self.ws.send(json.dumps(stop))
        except Exception:
            pass
        log.info("🎙️ 停止录音")


# ============================================================
#  主程序
# ============================================================
async def main(ws_url: str):
    # 使用固定 device_id，避免服务端把同一设备当新设备导致配置漂移
    device_id = "pi-laosan-001"
    log.info(f"设备ID: {device_id}")

    client = XiaozhiClient(ws_url, device_id)

    # 连接
    if not await client.connect():
        log.error("连接失败，退出")
        return

    speak_async("小机器人上线了")

    # 唤醒词监听
    listener = WakeWordListener()
    loop = asyncio.get_event_loop()

    def on_wake():
        listener.pause()
        asyncio.run_coroutine_threadsafe(client.on_wake_word(), loop)

        def wait_and_resume():
            # 防卡死：最多等待25秒，超时也强制恢复监听
            start_ts = time.time()
            time.sleep(2)
            while True:
                if not client.is_listening and not client.is_speaking:
                    break
                if time.time() - start_ts > 25:
                    log.warning("恢复监听等待超时，强制恢复")
                    break
                time.sleep(0.5)
            time.sleep(0.8)
            listener.resume()
            log.info(f"👂 继续监听: {WAKE_WORD}")

        threading.Thread(target=wait_and_resume, daemon=True).start()

    listener.start(on_wake)

    # 消息循环（保持长连接）
    await client.message_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="小智语音对话客户端")
    parser.add_argument("--ws", default="ws://192.168.0.69:8100/xiaozhi/v1/")
    parser.add_argument("--play-device", default=AUDIO_PLAY)
    parser.add_argument("--rec-device", default=AUDIO_REC)
    args = parser.parse_args()

    AUDIO_PLAY = args.play_device
    AUDIO_REC = args.rec_device

    asyncio.run(main(args.ws))
