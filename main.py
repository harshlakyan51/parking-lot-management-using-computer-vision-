import cv2
import numpy as np
from util import get_parking_spots_bboxes, empty_or_not

def calc_diff(im1, im2):
    """Calculate the absolute mean difference between two grayscale image patches."""
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    return np.abs(np.mean(im1_gray.astype("float") - im2_gray.astype("float")))

# Load mask and video
mask = cv2.imread('./mask_1920_1080.png', 0)
cap = cv2.VideoCapture('./samples/parking_1920_1080_loop.mp4')

# Get parking spot bounding boxes
spots = get_parking_spots_bboxes(cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S))
spots_status = [None] * len(spots)
previous_frame = None

frame_nmr, step = 0, 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]

    if frame_nmr % step == 0:
        if previous_frame is not None:
            diffs = []
            for x, y, w, h in spots:
                # Ensure bounding boxes are within frame boundaries
                x1, y1 = max(x, 0), max(y, 0)
                x2, y2 = min(x + w, frame_width), min(y + h, frame_height)
                roi_curr = frame[y1:y2, x1:x2]
                roi_prev = previous_frame[y1:y2, x1:x2]
                if roi_curr.shape == roi_prev.shape and roi_curr.size > 0:
                    diffs.append(calc_diff(roi_curr, roi_prev))
                else:
                    diffs.append(0)  # fallback if shape mismatch

            max_diff = max(diffs) if diffs else 1
            changed_spots = [i for i, d in enumerate(diffs) if d / max_diff > 0.4]
        else:
            changed_spots = range(len(spots))  # first frame

        # Update statuses
        for i in changed_spots:
            x, y, w, h = spots[i]
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + w, frame_width), min(y + h, frame_height)
            roi = frame[y1:y2, x1:x2]
            spots_status[i] = empty_or_not(roi)

        previous_frame = frame.copy()

    # Draw boxes
    for (x, y, w, h), status in zip(spots, spots_status):
        color = (0, 255, 0) if status else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Put text for count
    cv2.putText(frame, f'Available spots: {sum(spots_status)} / {len(spots_status)}',
                (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Only resize for display (not processing)
    display_frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("Output", display_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()
