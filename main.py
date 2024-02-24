from ultralytics import YOLO
print(f"""number of gay people in the room: {YOLO('yolov8l.pt').predict("images/img3.jpg", save=True, classes=[0,], conf=0.5, iou=0.2, line_width=2)[0].boxes.shape[0]}""")
