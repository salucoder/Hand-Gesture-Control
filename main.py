"""Control your mouse with your fingertips using a webcam.

Gestures (one hand, palm facing the camera):
  Index finger up ............ move the cursor
  Thumb + index pinch ........ left click (hold ~0.5s to drag)
  Thumb + middle pinch ....... right click
  Index + middle up .......... scroll (move hand up/down)
  Closed fist ................ pause (rest your hand)
Quit: press q or Esc in the camera window.
Emergency stop: slam the real mouse into a screen corner (PyAutoGUI failsafe).
"""
import time
import cv2
import numpy as np
import pyautogui
from hand_tracker import HandTracker

# ---- Settings you can tune ----
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
MARGIN = 100            # border of the camera view you don't need to reach
SMOOTHING = 5           # higher = smoother but laggier cursor
PINCH_THRESHOLD = 0.30  # higher = easier to trigger a pinch
HOLD_TO_DRAG = 0.5      # seconds of pinch before drag starts
CLICK_COOLDOWN = 0.5    # seconds between right clicks
SCROLL_GAIN = 8         # higher = faster scrolling
# --------------------------------

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


class FingerMouse:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        self.prev_x, self.prev_y = pyautogui.position()
        self.pinch_start = None
        self.dragging = False
        self.last_right = 0.0
        self.prev_scroll_y = None

    def release(self):
        """Safely end any drag/pinch (hand lost, or program exit)."""
        if self.dragging:
            pyautogui.mouseUp()
        self.dragging = False
        self.pinch_start = None
        self.prev_scroll_y = None

    def _left_pinch(self, pinching, now):
        """Short pinch = click on release. Long pinch = drag."""
        if pinching:
            if self.pinch_start is None:
                self.pinch_start = now
            elif not self.dragging and now - self.pinch_start > HOLD_TO_DRAG:
                pyautogui.mouseDown()
                self.dragging = True
        elif self.pinch_start is not None:
            if self.dragging:
                pyautogui.mouseUp()
            else:
                pyautogui.click()
            self.dragging = False
            self.pinch_start = None

    def _move(self, tracker, w, h):
        ix, iy = tracker.point(8, w, h)
        tx = np.interp(ix, (MARGIN, w - MARGIN), (0, self.screen_w - 1))
        ty = np.interp(iy, (MARGIN, h - MARGIN), (0, self.screen_h - 1))
        self.prev_x += (tx - self.prev_x) / SMOOTHING
        self.prev_y += (ty - self.prev_y) / SMOOTHING
        pyautogui.moveTo(self.prev_x, self.prev_y)

    def update(self, tracker, w, h):
        """Read the current hand pose, act on it, return the mode name."""
        now = time.time()
        index_up, middle_up, ring_up, pinky_up = up = tracker.fingers_up()
        pinch_left = tracker.pinch_ratio(4, 8) < PINCH_THRESHOLD
        pinch_right = tracker.pinch_ratio(4, 12) < PINCH_THRESHOLD

        if pinch_right and not pinch_left:
            self._left_pinch(False, now)
            self.prev_scroll_y = None
            if now - self.last_right > CLICK_COOLDOWN:
                pyautogui.rightClick()
                self.last_right = now
            return "Right click"

        if index_up and middle_up and not ring_up and not pinky_up and not pinch_left:
            self._left_pinch(False, now)
            y = tracker.point(8, w, h)[1]
            if self.prev_scroll_y is not None:
                dy = self.prev_scroll_y - y
                if abs(dy) > 2:
                    pyautogui.scroll(int(dy * SCROLL_GAIN))
            self.prev_scroll_y = y
            return "Scroll"

        self.prev_scroll_y = None

        if not any(up) and not pinch_left:
            self._left_pinch(False, now)
            return "Paused (fist)"

        # Move mode. Freeze the cursor during a short pinch so the click
        # lands where you aimed, not where your finger slipped to.
        if self.pinch_start is None or self.dragging:
            self._move(tracker, w, h)
        self._left_pinch(pinch_left, now)
        return "Dragging" if self.dragging else ("Pinch" if pinch_left else "Move")


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        print("Could not open the webcam. Try a different CAM_INDEX.")
        return

    tracker = HandTracker()
    mouse = FingerMouse()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror so it feels natural
            h, w = frame.shape[:2]

            if tracker.process(frame):
                mode = mouse.update(tracker, w, h)
            else:
                mouse.release()
                mode = "No hand"

            cv2.rectangle(frame, (MARGIN, MARGIN), (w - MARGIN, h - MARGIN), (255, 0, 255), 2)
            cv2.putText(frame, mode, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Finger Mouse (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    except pyautogui.FailSafeException:
        print("Failsafe triggered. Stopped.")
    finally:
        mouse.release()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
