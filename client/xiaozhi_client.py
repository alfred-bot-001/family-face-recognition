#!/usr/bin/env python3
"""
小智语音对话客户端 — 运行在老三(树莓派)上
唤醒词: "乐迪" (通过 vosk 离线识别)
通过 WebSocket 连接小智后端服务，实现语音对话

流程:
1. vosk 持续监听麦克风，检测唤醒词 "乐迪"
2. 检测到后连接 WebSocket，开始录音并发送 Opus 帧
3. 服务端 VAD 检测到静音后处理 ASR→LLM→TTS
4. 接收 Opus 音频帧解码播放
5. 对话结束回到监听状态
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import struct
import sys
import threading
import time
import uuid
from collections import deque

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
VOSK_MODEL = os.path.join(os.path.dirname(__file__), "models", "vosk-model-small-cn-0.22")
TALK_TIMEOUT = 15  # 对话最长时间(秒)

# ============================================================
#  Opus 编解码
# ============================================================
import opuslib_next as opuslib
_encoder = opuslib.Encoder(SAMPLE_RATE, CHANNELS, opuslib.APPLICATION_VOIP)
_decoder = opuslib.Decoder(SAMPLE_RATE, CHANNELS)


def pcm_to_opus(pcm: bytes) -> bytes:
    return _encoder.encode(pcm, FRAME_SIZE)


def opus_to_pcm(data: bytes) -> bytes:
    return _decoder.decode(data, FRAME_SIZE)


# ============================================================
#  本地语音 (espeak)
# ============================================================
def speak(text: str):
    """本地 TTS"""
    subprocess.run(
        f'espeak -v zh -s 320 --stdout "{text}" | aplay -D {AUDIO_PLAY} -q',
        shell=True, stderr=subprocess.DEVNULL
    )


def speak_async(text: str):
    threading.Thread(target=speak, args=(text,), daemon=True).start()


# ============================================================
#  唤醒词检测 (vosk 离线)
# ============================================================
class WakeWordListener:
    """用 vosk 持续监听麦克风，检测唤醒词"""

    def __init__(self, model_path=VOSK_MODEL, device=AUDIO_REC, wake_word=WAKE_WORD):
        from vosk import Model, KaldiRecognizer
        log.info(f"加载 vosk 模型: {model_path}")
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        self.device = device
        self.wake_word = wake_word
        self.active = True
        self.paused = False  # 对话期间暂停检测

    def listen(self, on_wake):
        """阻塞式监听，检测到唤醒词调用 on_wake()"""
        log.info(f"👂 监听唤醒词: {self.wake_word}")
        proc = subprocess.Popen(
            ["arecord", "-D", self.device, "-f", "S16_LE",
             "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw", "-q"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        chunk_size = 4000  # ~125ms of audio
        try:
            while self.active:
                data = proc.stdout.read(chunk_size)
                if not data:
                    break
                if self.paused:
                    continue
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        log.debug(f"vosk: {text}")
                        if self.wake_word in text:
                            log.info(f"🎯 唤醒词检测到! ({text})")
                            on_wake()
                else:
                    partial = json.loads(self.recognizer.PartialResult())
                    text = partial.get("partial", "")
                    if self.wake_word in text:
                        log.info(f"🎯 唤醒词检测到! (partial: {text})")
                        self.recognizer.Reset()
                        on_wake()
        finally:
            proc.terminate()

    def pause(self):
        self.paused = True

    def resume(self):
        self.recognizer.Reset()
        self.paused = False

    def stop(self):
        self.active = False


# ============================================================
#  小智 WebSocket 对话
# ============================================================
async def do_conversation(ws_url: str, device_id: str):
    """进行一次完整对话:连接→录音→等回复→播放→断开"""
    import websockets

    log.info(f"🔗 连接 {ws_url}")
    try:
        async with websockets.connect(ws_url, max_size=None, close_timeout=5) as ws:
            # === 握手 ===
            hello = {
                "type": "hello",
                "device_id": device_id,
                "device_name": "老三-树莓派",
                "device_mac": "AA:BB:CC:DD:EE:FF",
                "token": "",
                "features": {"mcp": False}
            }
            await ws.send(json.dumps(hello))

            resp = await asyncio.wait_for(ws.recv(), timeout=10)
            data = json.loads(resp)
            if data.get("type") != "hello" or not data.get("session_id"):
                log.error(f"握手失败: {data}")
                return
            session_id = data["session_id"]
            log.info(f"✅ 握手成功, session: {session_id}")

            # === 发送 listen start ===
            await ws.send(json.dumps({"type": "listen", "state": "start", "mode": "auto"}))

            # === 录音并发送 ===
            stop_recording = threading.Event()
            rec_proc = subprocess.Popen(
                ["arecord", "-D", AUDIO_REC, "-f", "S16_LE",
                 "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw", "-q"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )

            async def send_audio():
                frame_bytes = FRAME_SIZE * 2
                loop = asyncio.get_event_loop()
                while not stop_recording.is_set():
                    data = await loop.run_in_executor(None, rec_proc.stdout.read, frame_bytes)
                    if len(data) == frame_bytes:
                        try:
                            opus = pcm_to_opus(data)
                            await ws.send(opus)
                        except Exception:
                            break

            # === 接收消息 ===
            player_queue = deque()
            tts_done = asyncio.Event()
            conversation_done = asyncio.Event()

            def play_audio():
                """播放线程"""
                play_proc = None
                while not conversation_done.is_set():
                    if player_queue:
                        buf = bytearray()
                        while player_queue and len(buf) < SAMPLE_RATE * 2:
                            buf.extend(player_queue.popleft())
                        if buf:
                            try:
                                play_proc = subprocess.Popen(
                                    ["aplay", "-D", AUDIO_PLAY, "-f", "S16_LE",
                                     "-r", str(SAMPLE_RATE), "-c", "1", "-q"],
                                    stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
                                )
                                play_proc.stdin.write(bytes(buf))
                                play_proc.stdin.close()
                                play_proc.wait(timeout=10)
                            except Exception as e:
                                log.error(f"播放错误: {e}")
                    else:
                        time.sleep(0.01)

            play_thread = threading.Thread(target=play_audio, daemon=True)
            play_thread.start()

            async def recv_messages():
                try:
                    async for message in ws:
                        if isinstance(message, bytes):
                            try:
                                pcm = opus_to_pcm(message)
                                player_queue.append(pcm)
                            except Exception:
                                pass
                        else:
                            msg = json.loads(message)
                            msg_type = msg.get("type", "")

                            if msg_type == "tts":
                                state = msg.get("state", "")
                                if state == "start":
                                    log.info("🔊 开始播放回复")
                                    stop_recording.set()
                                    rec_proc.terminate()
                                elif state == "sentence_start":
                                    log.info(f"💬 {msg.get('text', '')}")
                                elif state == "stop":
                                    log.info("🔊 回复结束")
                                    # 等音频播完
                                    await asyncio.sleep(1)
                                    tts_done.set()
                                    return

                            elif msg_type == "stt":
                                text = msg.get("text", "")
                                log.info(f"🎤 识别: {text}")
                                # 识别到文本后停止录音
                                stop_recording.set()
                                rec_proc.terminate()

                            elif msg_type == "llm":
                                log.info(f"🤖 {msg.get('text', '')}")

                except Exception as e:
                    log.error(f"接收错误: {e}")

            # 并行: 发音频 + 收消息
            send_task = asyncio.create_task(send_audio())
            recv_task = asyncio.create_task(recv_messages())

            # 超时保护
            try:
                await asyncio.wait_for(tts_done.wait(), timeout=30)
            except asyncio.TimeoutError:
                log.warning("⏰ 对话超时")

            # 清理
            stop_recording.set()
            rec_proc.terminate()
            conversation_done.set()
            send_task.cancel()
            recv_task.cancel()

            # 等播放线程结束
            time.sleep(1)
            log.info("✅ 对话结束")

    except Exception as e:
        log.error(f"对话错误: {e}")


# ============================================================
#  主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="小智语音对话客户端 (老三)")
    parser.add_argument("--ws", default="ws://192.168.0.69:8100/xiaozhi/v1/",
                        help="小智 WebSocket 地址")
    parser.add_argument("--play-device", default=AUDIO_PLAY, help="播放设备")
    parser.add_argument("--rec-device", default=AUDIO_REC, help="录音设备")
    args = parser.parse_args()

    global AUDIO_PLAY, AUDIO_REC
    AUDIO_PLAY = args.play_device
    AUDIO_REC = args.rec_device

    device_id = f"pi-{uuid.uuid4().hex[:8]}"
    log.info(f"设备ID: {device_id}")

    # 启动语音
    speak("乐迪上线了")

    # 唤醒词监听
    listener = WakeWordListener(device=args.rec_device)

    def on_wake():
        listener.pause()
        speak("我在")
        try:
            asyncio.run(do_conversation(args.ws, device_id))
        except Exception as e:
            log.error(f"对话失败: {e}")
        finally:
            listener.resume()
            log.info(f"👂 继续监听唤醒词: {WAKE_WORD}")

    try:
        listener.listen(on_wake)
    except KeyboardInterrupt:
        log.info("退出")
        listener.stop()


if __name__ == "__main__":
    main()
