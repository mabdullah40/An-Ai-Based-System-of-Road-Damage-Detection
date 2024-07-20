from ultralytics import YOLO
import cv2
import math
import cvzone
from sort import *

model = YOLO('best.pt')

class_names = list(model.names.values())
print(class_names)
paths=r'E:\FYP_University\Uttilize Model\Video\v2.mp4'
cap = cv2.VideoCapture(paths)
cap.set(3, 480) 
cap.set(4, 480)

tracker = Sort(max_age=20, min_hits=20, iou_threshold=0.4) 
damage_counts = {cls_name: 0 for cls_name in class_names}
damage_trackers = {cls_name: {} for cls_name in class_names}
damage_threshold = 0.8

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    detections = np.empty((0, 5))
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            current_class = class_names[cls]
            
            if conf > 0.2:
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f'{current_class} {conf}', (max(0, x1), max(35, y1 - 15)), scale=1, thickness=1)
                current_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_array))
                
                # Update damage tracker
                if current_class in damage_trackers:
                    if (x1, y1, x2, y2) in damage_trackers[current_class]:
                        damage_trackers[current_class][(x1, y1, x2, y2)] += 1
                    else:
                        damage_trackers[current_class][(x1, y1, x2, y2)] = 1
    
    results = tracker.update(detections)
    
    for result in results:
        x1, y1, x2, y2, _ = result 
        print(result)
    
    # Count damages that stayed for the specified number of frames
    for cls_name in damage_trackers:
        for bbox, count in damage_trackers[cls_name].items():
            if count >= damage_threshold:
                damage_counts[cls_name] += 1
        damage_trackers[cls_name] = {bbox: count for bbox, count in damage_trackers[cls_name].items() if count < damage_threshold}
    
    print("Damage Counts:")
    print("Alligator Cracks:", damage_counts['Alligator Cracks'])
    print("Damaged Crosswalk:", damage_counts['Damaged crosswalk'])
    print("Damaged Paint:", int(damage_counts['Damaged paint']/4))
    print("Longitudinal Cracks:", damage_counts['Longitudinal Cracks'])
    print("Manhole Cover:", damage_counts['Manhole cover'])
    print("Potholes:", damage_counts['Potholes'])
    print("Transverse Cracks:", damage_counts['Transverse Cracks'])
    
    cv2.imshow('Image', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
