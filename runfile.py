import sys
sys.path.insert(0, 'D:\\Thesis_HCMUS\\Code\\ultralytics')

from ultralytics import YOLO

model = YOLO('D:\\Thesis_HCMUS\\Code\\yolov8_custom.yaml')
print(model)