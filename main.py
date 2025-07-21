import cv2
import numpy as np
from util import get_parking_spots_bboxes, empty_or_not

def calc_diff(im1, im2):
    """Calculate the absolute mean difference between two images using NumPy."""
    return np.abs(np.mean(im1 - im2))

# Load mask and video
mask = cv2.imread('./mask_1920_1080.png', 0)
cap = cv2.VideoCapture('./samples/parking_1920_1080_loop.mp4')

# Get parking spots
spots = get_parking_spots_bboxes(cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S))
spots_status = [None] * len(spots)
previous_frame = None

frame_nmr, step = 0, 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_nmr % step == 0:
        if previous_frame is not None:
            # Compute differences for all spots
            diffs = [calc_diff(frame[y:y+h, x:x+w], previous_frame[y:y+h, x:x+w]) for x, y, w, h in spots]
            max_diff = max(diffs) if diffs else 1  # Avoid division by zero
            changed_spots = [i for i, d in enumerate(diffs) if d / max_diff > 0.4]
        else:
            changed_spots = range(len(spots))  # Process all spots initially

        # Update spot status
        for i in changed_spots:
            x, y, w, h = spots[i]
            spots_status[i] = empty_or_not(frame[y:y+h, x:x+w])

        previous_frame = frame.copy()

    # Draw bounding boxes
    for (x, y, w, h), status in zip(spots, spots_status):
        color = (0, 255, 0) if status else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Display available spots count
    cv2.putText(frame, f'Available spots: {sum(spots_status)} / {len(spots_status)}',
                (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show frame
    cv2.imshow('Parking Lot', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()
