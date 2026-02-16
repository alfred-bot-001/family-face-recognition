"""手势检测模块 - 基于 MediaPipe Hands 检测挥手动作"""

import time
from collections import deque

import mediapipe as mp
import numpy as np


class WaveDetector:
    """检测挥手动作

    原理：
    1. MediaPipe Hands 检测手部关键点
    2. 判断手掌张开（5 指伸直）
    3. 跟踪手腕 x 坐标的左右摆动
    4. 短时间内方向变化 >= 3 次 = 挥手
    """

    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.mp_hands = mp.solutions.hands

        # 挥手检测状态
        self.wrist_history: deque = deque(maxlen=15)  # (timestamp, x_pos)
        self.last_wave_time = 0
        self.wave_cooldown = 3.0  # 挥手检测冷却（秒）

        # 挥手判定参数
        self.min_direction_changes = 3   # 最少方向变化次数
        self.history_window = 1.5        # 检测时间窗口（秒）
        self.min_movement = 0.03         # 最小移动距离（归一化坐标）

    def _is_finger_extended(self, landmarks, tip_id: int, pip_id: int) -> bool:
        """判断手指是否伸直（指尖高于 PIP 关节）"""
        return landmarks[tip_id].y < landmarks[pip_id].y

    def _is_thumb_extended(self, landmarks) -> bool:
        """判断拇指是否伸直"""
        return abs(landmarks[self.mp_hands.HandLandmark.THUMB_TIP].x -
                   landmarks[self.mp_hands.HandLandmark.WRIST].x) > 0.05

    def _is_open_palm(self, landmarks) -> bool:
        """判断手掌是否张开（至少 4 指伸直）"""
        extended = 0
        # 四指
        finger_tips = [
            (self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_PIP),
            (self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
            (self.mp_hands.HandLandmark.RING_FINGER_TIP, self.mp_hands.HandLandmark.RING_FINGER_PIP),
            (self.mp_hands.HandLandmark.PINKY_TIP, self.mp_hands.HandLandmark.PINKY_PIP),
        ]
        for tip_id, pip_id in finger_tips:
            if self._is_finger_extended(landmarks, tip_id, pip_id):
                extended += 1

        if self._is_thumb_extended(landmarks):
            extended += 1

        return extended >= 4

    def _check_wave_pattern(self) -> bool:
        """分析手腕位置历史，判断是否有挥手模式"""
        now = time.time()

        # 只看时间窗口内的数据
        recent = [(t, x) for t, x in self.wrist_history if now - t < self.history_window]
        if len(recent) < 4:
            return False

        # 统计方向变化
        direction_changes = 0
        prev_dx = 0
        for i in range(1, len(recent)):
            dx = recent[i][1] - recent[i - 1][1]
            if abs(dx) < self.min_movement:
                continue
            if prev_dx != 0 and (dx > 0) != (prev_dx > 0):
                direction_changes += 1
            prev_dx = dx

        return direction_changes >= self.min_direction_changes

    def detect(self, frame_bgr: np.ndarray) -> dict:
        """
        检测手势

        返回:
            {
                "hands_count": int,
                "wave_detected": bool,
                "hand_landmarks": list[dict],  # 每只手的关键信息
            }
        """
        now = time.time()
        result = {
            "hands_count": 0,
            "wave_detected": False,
            "hand_landmarks": [],
        }

        frame_rgb = frame_bgr[:, :, ::-1]  # BGR → RGB（避免 cvtColor 拷贝）
        mp_result = self.hands.process(frame_rgb)

        if not mp_result.multi_hand_landmarks:
            self.wrist_history.clear()
            return result

        result["hands_count"] = len(mp_result.multi_hand_landmarks)

        for hand_lms in mp_result.multi_hand_landmarks:
            lms = hand_lms.landmark
            wrist = lms[self.mp_hands.HandLandmark.WRIST]
            palm_open = self._is_open_palm(lms)

            hand_info = {
                "wrist_x": round(wrist.x, 3),
                "wrist_y": round(wrist.y, 3),
                "palm_open": palm_open,
            }
            result["hand_landmarks"].append(hand_info)

            # 只有张开手掌才跟踪挥手
            if palm_open:
                self.wrist_history.append((now, wrist.x))

                # 检查挥手（冷却期外）
                if now - self.last_wave_time > self.wave_cooldown:
                    if self._check_wave_pattern():
                        result["wave_detected"] = True
                        self.last_wave_time = now
                        self.wrist_history.clear()

        return result
