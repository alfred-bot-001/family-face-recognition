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
import uuid

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
WAKE_WORD = "乐迪"
# sherpa-onnx 常见误识别变体
# (唤醒词检测已改用 KWS，无需模糊匹配)
SHERPA_KWS_DIR = os.path.join(os.path.dirname(__file__), "models", "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20")

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
#  唤醒词检测 (sherpa-onnx KeywordSpotter)
# ============================================================
class WakeWordListener:
    def __init__(self, device=AUDIO_REC):
        import sherpa_onnx
        import numpy as np
        self.np = np
        self.device = device
        self.paused = False
        self.active = True
        self._proc = None
        self._cooldown = 1.5
        self._last_detect = 0

        encoder = os.path.join(SHERPA_KWS_DIR, "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx")
        decoder = os.path.join(SHERPA_KWS_DIR, "decoder-epoch-13-avg-2-chunk-16-left-64.onnx")
        joiner = os.path.join(SHERPA_KWS_DIR, "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx")
        tokens = os.path.join(SHERPA_KWS_DIR, "tokens.txt")
        keywords = os.path.join(SHERPA_KWS_DIR, "keywords.txt")

        log.info(f"加载 sherpa-onnx KWS: {SHERPA_KWS_DIR}")
        self.kws = sherpa_onnx.KeywordSpotter(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            keywords_file=keywords,
            num_threads=4,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            max_active_paths=2,
            keywords_score=1.8,
            keywords_threshold=0.2,
            num_trailing_blanks=1,
            provider="cpu",
        )
        log.info("sherpa-onnx KWS 加载完成")

    def start(self, on_wake):
        threading.Thread(target=self._listen, args=(on_wake,), daemon=True).start()

    def _listen(self, on_wake):
        log.info(f"👂 监听唤醒词: {WAKE_WORD}")
        self._proc = subprocess.Popen(
            ["arecord", "-D", self.device, "-f", "S16_LE",
             "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw", "-q"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        stream = self.kws.create_stream()
        chunk_samples = int(SAMPLE_RATE * 0.1)  # 100ms
        chunk_bytes = chunk_samples * 2
        read_count = 0
        while self.active and self._proc.poll() is None:
            data = self._proc.stdout.read(chunk_bytes)
            read_count += 1
            if read_count % 100 == 0:
                log.info(f"[debug] audio chunks: {read_count}, paused={self.paused}")
            if not data or self.paused:
                if self.paused:
                    time.sleep(0.1)
                continue
            samples = self.np.frombuffer(data, dtype=self.np.int16).astype(self.np.float32) / 32768.0
            stream.accept_waveform(SAMPLE_RATE, samples)
            while self.kws.is_ready(stream):
                self.kws.decode_stream(stream)
            result = self.kws.get_result(stream)
            if result:
                log.info(f"KWS 检测到: {result}")
                self._trigger(on_wake, result)

    def _trigger(self, on_wake, result):
        now = time.time()
        if now - self._last_detect < self._cooldown:
            return
        self._last_detect = now
        log.info(f"🎯 唤醒词!")
        on_wake()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.active = False
        if self._proc:
            self._proc.terminate()

# ============================================================
#  小智客户端 (长连接)
# ============================================================
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
        """唤醒词触发"""
        if not self.connected:
            log.warning("未连接，忽略唤醒")
            return

        log.info("🎙️ 唤醒词触发，开始对话")

        # 如果服务端在说话，先打断
        if self.is_speaking:
            abort = {"session_id": self.session_id, "type": "abort", "reason": "wake_word_detected"}
            await self.ws.send(json.dumps(abort))
            # 关闭播放进程
            if self._play_proc:
                try:
                    self._play_proc.terminate()
                except Exception:
                    pass
                self._play_proc = None

        # 发唤醒词检测消息
        detect = {
            "session_id": self.session_id,
            "type": "listen",
            "state": "detect",
            "text": WAKE_WORD,
        }
        await self.ws.send(json.dumps(detect))

        # 发开始监听
        start = {
            "session_id": self.session_id,
            "type": "listen",
            "state": "start",
            "mode": "auto",
        }
        await self.ws.send(json.dumps(start))

        # 开始录音发送
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
        try:
            while self.is_listening and self.connected:
                data = await loop.run_in_executor(None, self._rec_proc.stdout.read, frame_bytes)
                if len(data) == frame_bytes:
                    opus = pcm_to_opus(data)
                    await self.ws.send(opus)
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
    device_id = f"pi-{uuid.uuid4().hex[:8]}"
    log.info(f"设备ID: {device_id}")

    client = XiaozhiClient(ws_url, device_id)

    # 连接
    if not await client.connect():
        log.error("连接失败，退出")
        return

    speak_async("悟空上线了")

    # 唤醒词监听
    listener = WakeWordListener()
    loop = asyncio.get_event_loop()

    def on_wake():
        listener.pause()
        asyncio.run_coroutine_threadsafe(client.on_wake_word(), loop)
        # 等服务端说完后恢复监听
        def wait_and_resume():
            time.sleep(2)  # 等唤醒处理
            while client.is_listening or client.is_speaking:
                time.sleep(0.5)
            time.sleep(1)
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
