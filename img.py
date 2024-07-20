from ultralytics import YOLO
import cv2

model = YOLO('best.pt')

results = model(r'images\7.jpg', show=True)

cv2.waitKey(0) 
cv2.destroyAllWindows()
