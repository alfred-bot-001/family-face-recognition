"""
emotions.py — 多多表情动作模块

通过舵机云台 + 电机轮子表达情绪。
所有动作执行完后回到 base 位置，线程安全。

硬件指令:
  舵机: {"T":133, "X":pan, "Y":tilt, "SPD":spd, "ACC":acc}
  电机: {"T":1, "L":left_speed, "R":right_speed}  (正=前进, 负=后退)
"""

import time
import threading
import logging

log = logging.getLogger("emotions")

# 电机指令 ID
CMD_MOTOR = 1

# 全局动作锁，防止多个动作同时执行
_emotion_lock = threading.Lock()


def _run_steps(gimbal, steps, base_pan=None, base_tilt=None):
    """执行一系列舵机步骤，每步格式: (pan_offset, tilt_offset, speed, acc, delay)"""
    if base_pan is None:
        base_pan = gimbal.pan_angle
    if base_tilt is None:
        base_tilt = gimbal.tilt_angle
    for dx, dy, spd, acc, delay in steps:
        gimbal.move_to(base_pan + dx, base_tilt + dy, speed=spd, acc=acc)
        time.sleep(delay)
    # 回到基准
    gimbal.move_to(base_pan, base_tilt, speed=10, acc=2)


def _drive(gimbal, left, right, duration):
    """驱动电机一段时间后停止"""
    gimbal.send_command({"T": CMD_MOTOR, "L": left, "R": right})
    time.sleep(duration)
    gimbal.send_command({"T": CMD_MOTOR, "L": 0, "R": 0})


def _safe_run(gimbal, fn):
    """线程安全执行动作"""
    if not getattr(gimbal, "connected", False):
        return
    acquired = _emotion_lock.acquire(blocking=False)
    if not acquired:
        return  # 有动作在执行，跳过
    try:
        fn(gimbal)
    except Exception as e:
        log.error(f"表情动作异常: {e}")
    finally:
        _emotion_lock.release()


# ============================================================
#  表情动作定义
# ============================================================

def _happy(gimbal):
    """开心 😊 — 快速左右摇头 + 微仰头 + 轮子小扭"""
    base_pan, base_tilt = gimbal.pan_angle, gimbal.tilt_angle
    # 摇头
    for dx in [15, -15, 12, -12, 8, -8, 0]:
        gimbal.move_to(base_pan + dx, base_tilt + 5, speed=15, acc=4)
        time.sleep(0.12)
    # 轮子小扭
    _drive(gimbal, 80, -80, 0.15)
    _drive(gimbal, -80, 80, 0.15)
    # 归位
    gimbal.move_to(base_pan, base_tilt, speed=10, acc=2)


def _excited(gimbal):
    """超开心 🎉 — 大幅摇头 + 点头 + 前冲再退"""
    base_pan, base_tilt = gimbal.pan_angle, gimbal.tilt_angle
    # 大幅摇
    for dx in [25, -25, 20, -20, 0]:
        gimbal.move_to(base_pan + dx, base_tilt, speed=18, acc=5)
        time.sleep(0.12)
    # 点头
    for dy in [20, -10, 15, 0]:
        gimbal.move_to(base_pan, base_tilt + dy, speed=12, acc=3)
        time.sleep(0.15)
    # 前冲再退
    _drive(gimbal, 120, 120, 0.2)
    _drive(gimbal, -120, -120, 0.2)
    gimbal.move_to(base_pan, base_tilt, speed=10, acc=2)


def _angry(gimbal):
    """生气 😠 — 快速甩头 + 低头 + 前冲急停"""
    base_pan, base_tilt = gimbal.pan_angle, gimbal.tilt_angle
    # 快速甩头
    for dx in [30, -30, 25, -25]:
        gimbal.move_to(base_pan + dx, base_tilt, speed=20, acc=5)
        time.sleep(0.08)
    # 低头
    gimbal.move_to(base_pan, base_tilt - 15, speed=15, acc=3)
    time.sleep(0.2)
    # 前冲
    _drive(gimbal, 150, 150, 0.15)
    _drive(gimbal, 0, 0, 0.1)
    gimbal.move_to(base_pan, base_tilt, speed=10, acc=2)


def _shy(gimbal):
    """害羞 😳 — 慢转头到一侧 + 微低头 + 后退"""
    base_pan, base_tilt = gimbal.pan_angle, gimbal.tilt_angle
    # 慢慢转头躲避
    gimbal.move_to(base_pan + 35, base_tilt - 10, speed=5, acc=1)
    time.sleep(0.6)
    gimbal.move_to(base_pan + 40, base_tilt - 15, speed=3, acc=1)
    time.sleep(0.4)
    # 后退一点
    _drive(gimbal, -80, -80, 0.2)
    time.sleep(0.3)
    # 慢慢回来
    gimbal.move_to(base_pan, base_tilt, speed=5, acc=1)


def _sad(gimbal):
    """伤心 😢 — 缓慢低头 + 微微左右摇"""
    base_pan, base_tilt = gimbal.pan_angle, gimbal.tilt_angle
    # 缓慢低头
    gimbal.move_to(base_pan, base_tilt - 25, speed=3, acc=1)
    time.sleep(0.8)
    # 微微摇头（叹气感）
    for dx in [5, -5, 3, -3, 0]:
        gimbal.move_to(base_pan + dx, base_tilt - 25, speed=3, acc=1)
        time.sleep(0.3)
    time.sleep(0.5)
    gimbal.move_to(base_pan, base_tilt, speed=5, acc=1)


def _curious(gimbal):
    """好奇 🤔 — 歪头 + 前倾 + 缓慢前进"""
    base_pan, base_tilt = gimbal.pan_angle, gimbal.tilt_angle
    # 歪头
    gimbal.move_to(base_pan + 20, base_tilt + 10, speed=8, acc=2)
    time.sleep(0.4)
    # 前倾
    gimbal.move_to(base_pan + 20, base_tilt + 20, speed=5, acc=1)
    time.sleep(0.3)
    # 缓慢前进
    _drive(gimbal, 60, 60, 0.3)
    time.sleep(0.3)
    gimbal.move_to(base_pan, base_tilt, speed=8, acc=2)


def _greet(gimbal):
    """打招呼 👋 — 仰头 + 左右摆 + 原地小转"""
    base_pan, base_tilt = gimbal.pan_angle, gimbal.tilt_angle
    # 仰头
    gimbal.move_to(base_pan, base_tilt + 20, speed=12, acc=3)
    time.sleep(0.2)
    # 左右摆动
    for dx in [20, -20, 15, -15, 0]:
        gimbal.move_to(base_pan + dx, base_tilt + 15, speed=12, acc=3)
        time.sleep(0.15)
    # 原地小转
    _drive(gimbal, 100, -100, 0.2)
    _drive(gimbal, -100, 100, 0.2)
    gimbal.move_to(base_pan, base_tilt, speed=10, acc=2)


def _cute(gimbal):
    """撒娇 🥺 — 歪头 + 快速小点头 + 小扭"""
    base_pan, base_tilt = gimbal.pan_angle, gimbal.tilt_angle
    # 歪头
    gimbal.move_to(base_pan - 15, base_tilt + 5, speed=8, acc=2)
    time.sleep(0.3)
    # 快速小点头
    for dy in [8, -3, 6, -2, 4, 0]:
        gimbal.move_to(base_pan - 15, base_tilt + dy, speed=12, acc=3)
        time.sleep(0.1)
    # 小扭
    _drive(gimbal, 60, -60, 0.12)
    _drive(gimbal, -60, 60, 0.12)
    gimbal.move_to(base_pan, base_tilt, speed=8, acc=2)


def _sleepy(gimbal):
    """困了 😴 — 慢慢低头 + 微摇（打瞌睡）"""
    base_pan, base_tilt = gimbal.pan_angle, gimbal.tilt_angle
    for _ in range(2):
        gimbal.move_to(base_pan, base_tilt - 15, speed=3, acc=1)
        time.sleep(0.5)
        gimbal.move_to(base_pan + 3, base_tilt - 8, speed=3, acc=1)
        time.sleep(0.4)
    gimbal.move_to(base_pan, base_tilt - 25, speed=2, acc=1)
    time.sleep(0.5)
    gimbal.move_to(base_pan, base_tilt, speed=5, acc=1)


def _surprise(gimbal):
    """惊讶 😲 — 快速仰头 + 定住 + 后退"""
    base_pan, base_tilt = gimbal.pan_angle, gimbal.tilt_angle
    # 快速仰头
    gimbal.move_to(base_pan, base_tilt + 30, speed=20, acc=5)
    time.sleep(0.3)
    # 定住
    time.sleep(0.4)
    # 后退
    _drive(gimbal, -100, -100, 0.2)
    time.sleep(0.2)
    gimbal.move_to(base_pan, base_tilt, speed=10, acc=2)


# ============================================================
#  情绪 → 动作映射
# ============================================================

EMOTION_MAP = {
    "happy":    _happy,
    "excited":  _excited,
    "angry":    _angry,
    "shy":      _shy,
    "sad":      _sad,
    "curious":  _curious,
    "greet":    _greet,
    "cute":     _cute,
    "sleepy":   _sleepy,
    "surprise": _surprise,
}

# 关键词 → 情绪 (按优先级排列，先匹配先触发)
KEYWORD_EMOTION = [
    # 超开心
    (["太棒了", "太好了", "万岁", "耶", "好厉害", "真棒"], "excited"),
    # 开心
    (["哈哈", "嘻嘻", "开心", "高兴", "好玩", "有趣", "喜欢", "爱你", "谢谢"], "happy"),
    # 生气
    (["生气", "讨厌", "烦", "哼", "不行", "不可以", "坏蛋"], "angry"),
    # 害羞
    (["害羞", "不好意思", "脸红", "羞", "嘿嘿"], "shy"),
    # 伤心
    (["伤心", "难过", "哭", "呜呜", "可怜", "对不起", "抱歉", "遗憾"], "sad"),
    # 好奇
    (["好奇", "为什么", "怎么回事", "让我看看", "有意思", "奇怪"], "curious"),
    # 撒娇
    (["求你", "拜托", "好不好", "嘛", "人家"], "cute"),
    # 困了
    (["困了", "好累", "打哈欠", "瞌睡", "累了"], "sleepy"),
    # 惊讶
    (["哇", "天哪", "不会吧", "真的吗", "吓", "惊", "厉害"], "surprise"),
    # 打招呼
    (["你好", "早上好", "下午好", "晚上好", "嗨"], "greet"),
]


def detect_emotion(text: str) -> str | None:
    """从文本中检测情绪，返回情绪名或 None"""
    for keywords, emotion in KEYWORD_EMOTION:
        for kw in keywords:
            if kw in text:
                return emotion
    return None


def play_emotion(gimbal, emotion: str):
    """异步播放指定情绪动作（非阻塞）"""
    fn = EMOTION_MAP.get(emotion)
    if not fn:
        log.warning(f"未知情绪: {emotion}")
        return
    threading.Thread(
        target=_safe_run, args=(gimbal, fn), daemon=True
    ).start()


def play_emotion_from_text(gimbal, text: str):
    """从文本检测情绪并播放动作（非阻塞），返回检测到的情绪名"""
    emotion = detect_emotion(text)
    if emotion:
        log.info(f"🎭 检测到情绪: {emotion} ← '{text[:30]}'")
        play_emotion(gimbal, emotion)
    return emotion
