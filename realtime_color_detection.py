"""Real-time red and blue color detection using a laptop webcam and OpenCV.

Press Q or Esc to quit.
"""

import argparse
import sys

import cv2
import numpy as np


# HSV ranges for red. Red wraps around the hue boundary, so it needs 2 ranges.
RED_LOWER_1 = np.array([0, 100, 70], dtype=np.uint8)
RED_UPPER_1 = np.array([10, 255, 255], dtype=np.uint8)
RED_LOWER_2 = np.array([170, 100, 70], dtype=np.uint8)
RED_UPPER_2 = np.array([179, 255, 255], dtype=np.uint8)

# Approximate HSV range for blue; adjust if the room lighting requires it.
BLUE_LOWER = np.array([95, 100, 50], dtype=np.uint8)
BLUE_UPPER = np.array([130, 255, 255], dtype=np.uint8)

MIN_CONTOUR_AREA = 600  # Ignore tiny detections/noise; adjust for object size.
KERNEL = np.ones((5, 5), dtype=np.uint8)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Remove small noise and fill small gaps in a binary color mask."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    return mask


def draw_detections(
    frame: np.ndarray,
    mask: np.ndarray,
    label: str,
    box_color: tuple[int, int, int],
) -> int:
    """Draw a bounding box and label around each sufficiently large contour."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + width, y + height), box_color, 2)

        text = f"{label} (area: {int(area)})"
        text_y = max(y - 10, 25)
        cv2.putText(
            frame,
            text,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            box_color,
            2,
            cv2.LINE_AA,
        )
        detected_count += 1

    return detected_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect red and blue areas from a webcam feed.")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0; try 1 if the wrong camera opens).",
    )
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(
            f"Error: Could not open camera index {args.camera}. "
            "Check camera permissions or try --camera 1.",
            file=sys.stderr,
        )
        return 1

    # Request a moderate resolution to keep detection responsive on most laptops.
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Red/blue color detection started. Press Q or Esc in the video window to quit.")

    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("Warning: Could not read a frame from the webcam.", file=sys.stderr)
                break

            # OpenCV webcam frames are BGR; convert them to HSV for color filtering.
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Red crosses the HSV hue boundary, so merge the two red masks.
            red_mask_1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
            red_mask_2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
            red_mask = clean_mask(cv2.bitwise_or(red_mask_1, red_mask_2))
            blue_mask = clean_mask(cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER))

            red_count = draw_detections(frame, red_mask, "RED", (0, 0, 255))
            blue_count = draw_detections(frame, blue_mask, "BLUE", (255, 0, 0))

            # Status panel
            cv2.rectangle(frame, (8, 8), (265, 78), (30, 30, 30), thickness=-1)
            cv2.putText(
                frame,
                f"RED objects: {red_count}",
                (18, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"BLUE objects: {blue_count}",
                (18, 63),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 100, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Real-Time Red & Blue Color Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or Esc
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
