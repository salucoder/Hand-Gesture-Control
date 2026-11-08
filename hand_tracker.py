"""Hand detection with MediaPipe + small helpers for finger gestures."""
import math
import cv2
# pyrefly: ignore [missing-import]
import mediapipe as mp

TIPS = [8, 12, 16, 20]   # index, middle, ring, pinky fingertips
PIPS = [6, 10, 14, 18]   # the middle joint of each of those fingers


class HandTracker:
    def __init__(self, detect_conf=0.7, track_conf=0.6):
        self._mp_hands = mp.solutions.hands
        self._drawer = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=detect_conf,
            min_tracking_confidence=track_conf,
        )
        self.lm = None  # landmarks of the current hand

    def process(self, frame_bgr, draw=True):
        """Find a hand in the frame. Returns True if one was found."""
        result = self._hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self.lm = None
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            self.lm = hand.landmark
            if draw:
                self._drawer.draw_landmarks(frame_bgr, hand, self._mp_hands.HAND_CONNECTIONS)
        return self.lm is not None

    def point(self, idx, w, h):
        """Pixel position of a landmark."""
        return self.lm[idx].x * w, self.lm[idx].y * h

    def fingers_up(self):
        """[index, middle, ring, pinky] -> True if the finger is straight up.
        Works best with the hand upright, palm facing the camera."""
        return [self.lm[t].y < self.lm[p].y for t, p in zip(TIPS, PIPS)]

    def _dist(self, a, b):
        return math.hypot(self.lm[a].x - self.lm[b].x, self.lm[a].y - self.lm[b].y)

    def pinch_ratio(self, a, b):
        """Distance between two landmarks divided by hand size,
        so it works whether your hand is near or far from the camera."""
        return self._dist(a, b) / (self._dist(0, 9) or 1e-6)

    def close(self):
        self._hands.close()
