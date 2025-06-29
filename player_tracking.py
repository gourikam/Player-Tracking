import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

CONFIDENCE_THRESHOLD = 0.2
MIN_BOX_WIDTH = 20
MIN_BOX_HEIGHT = 40
MAX_BOX_WIDTH = 400
MAX_BOX_HEIGHT = 800
MAX_PLAYERS = 22

model = YOLO("best.pt")

# Initialize Deep SORT tracker
tracker = DeepSort(
    max_age=20,               
    n_init=5,                 
    max_iou_distance=0.4,     
    nn_budget=300,
    embedder="mobilenet",
    half=True
)

# Track IDs
track_id_map = {}     
next_player_id = 1

cap = cv2.VideoCapture("15sec_input_720p.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
ret, frame = cap.read()
if not ret:
    print("Error: Could not read video.")
    exit()

H, W = frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter("player_tracking.mp4", fourcc, fps * 2, (W, H))

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []

    for r in results.boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        conf = float(r.conf)
        cls = int(r.cls[0])
        class_name = model.names[cls]

        if class_name != 'player' or conf < CONFIDENCE_THRESHOLD:
            continue

        w, h = x2 - x1, y2 - y1
        if w < MIN_BOX_WIDTH or h < MIN_BOX_HEIGHT or w > MAX_BOX_WIDTH or h > MAX_BOX_HEIGHT:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        try:
            hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        except:
            continue

        yellow_mask = cv2.inRange(hsv_crop, (20, 100, 100), (35, 255, 255))
        yellow_ratio = cv2.countNonZero(yellow_mask) / (crop.size / 3)
        if yellow_ratio > 0.10:
            continue

        bbox = [x1, y1, w, h]
        detections.append((bbox, conf, class_name))

    # Deep SORT tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        internal_id = track.track_id

        if internal_id not in track_id_map:
            if len(track_id_map) >= MAX_PLAYERS:
                continue
            track_id_map[internal_id] = next_player_id
            next_player_id += 1

        player_id = track_id_map[internal_id]

        x1, y1, x2, y2 = map(int, track.to_ltrb())
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        try:
            hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        except:
            continue

        mask_red1 = cv2.inRange(hsv_crop, (0, 70, 50), (10, 255, 255))
        mask_red2 = cv2.inRange(hsv_crop, (160, 70, 50), (180, 255, 255))
        red_pixels = cv2.countNonZero(mask_red1) + cv2.countNonZero(mask_red2)

        mask_blue = cv2.inRange(hsv_crop, (100, 70, 50), (130, 255, 255))
        blue_pixels = cv2.countNonZero(mask_blue)

        if red_pixels > blue_pixels:
            box_color = (0, 0, 255)  # Red
        elif blue_pixels > red_pixels:
            box_color = (255, 0, 0)  # Blue
        else:
            continue  

        # Draw final box and ID only
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, f"{player_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    cv2.imshow("Player Tracking (Final)", frame)
    output.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output.release()
cv2.destroyAllWindows()
