import numpy as np
import cv2
from ultralytics import YOLO
from shapely.geometry import Point, Polygon

model = YOLO("yolov8n.pt")
video_path = "video.mp4"
confidence_threshold = 0.7

outer_polygon_coords = [(290, 1055), (680, 220), (1055, 220), (935, 1080)]
inner_polygon_coords = [(400, 950), (550, 500), (900, 500), (900, 950)]
outer_poly = Polygon(outer_polygon_coords)
inner_poly = Polygon(inner_polygon_coords)

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(
    "result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)
car_count = 0
tracked_ids_in_inner = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.polylines(
        frame, [np.array(outer_polygon_coords, np.int32)], True, (0, 0, 255), 3
    )
    cv2.polylines(
        frame, [np.array(inner_polygon_coords, np.int32)], True, (0, 255, 0), 3
    )

    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=confidence_threshold,
        verbose=False,
    )

    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if box.id is not None:
                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                point = Point(cx, cy)

                if outer_poly.contains(point):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                    if inner_poly.contains(point):
                        if track_id not in tracked_ids_in_inner:
                            tracked_ids_in_inner.add(track_id)
                            car_count += 1

    cv2.putText(
        frame,
        f"car = {car_count}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 255),
        3,
    )
    out.write(frame)
    cv2.imshow("Rec", frame)
    if cv2.waitKey(1) and 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
