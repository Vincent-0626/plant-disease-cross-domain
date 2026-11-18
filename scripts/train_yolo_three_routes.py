from ultralytics import YOLO
from pathlib import Path
p=Path(__file__).resolve().parents[1] / 'runs/three_routes_v1'
model=YOLO('yolo11n.pt')
model.train(data=str(p/'yolo/dataset.yaml'),epochs=40,patience=8,imgsz=640,batch=16,workers=4,device=0,seed=42,deterministic=True,project=str(p),name='leaf_detector',exist_ok=False,pretrained=True,plots=False,cache=False)
(p/'YOLO_COMPLETE').write_text('completed\n')
