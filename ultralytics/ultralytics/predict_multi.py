
import json, time
from ultralytics import YOLO

W = r"best.pt"
IMG = r"dataset/images/val/sample.jpg"

m = YOLO(W)
r = m.predict(IMG, imgsz=512, conf=0.25, device=0)[0]

dets = []
for b in r.boxes:
    x1,y1,x2,y2 = b.xyxy[0].tolist()
    cls = int(b.cls[0].item())
    depth = float(getattr(b, 'depth', 0.0))
    tau   = float(getattr(b, 'tau', 0.0))
    phi   = float(getattr(b, 'phi', 0.0))
    mu    = float(getattr(b, 'mu', 0.0))
    sigma = float(getattr(b, 'sigma', 0.0))
    dets.append({"cls":cls,"xyxy":[x1,y1,x2,y2],
                 "d":depth,"tau":tau,"phi":phi,"mu":mu,"sigma":sigma})

pkt = {"frame_id":0,"ts":time.time(),"dets":dets,
       "meta":{"depth_min":0.2,"depth_max":20.0,"imgsz":[r.orig_shape[1], r.orig_shape[0]]}}
print(json.dumps(pkt, ensure_ascii=False))

