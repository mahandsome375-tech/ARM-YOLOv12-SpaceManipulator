
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO


W   = r"best.pt"
IMG = r"dataset\images\train\00000.jpg"

SAVE_DIR = Path(r"runs\detect\train_82\predict_depth")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def colorize_depth(d: np.ndarray) -> np.ndarray:
    d = d.astype(np.float32)
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    d = (d - lo) / (hi - lo + 1e-6)
    d = np.clip(d, 0, 1)



    color_map = cv2.COLORMAP_TURBO

    vis = cv2.applyColorMap((d * 255).astype(np.uint8), color_map)


    vis = np.clip((vis / 255.0) ** 0.8 * 255, 0, 255).astype(np.uint8)


    vis = cv2.convertScaleAbs(vis, alpha=1.2, beta=-15)

    return vis


model = YOLO(W)


img0 = cv2.imread(IMG)
if img0 is None:
    raise FileNotFoundError(f"Image not found: {IMG}")
img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
im  = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
im  = im.unsqueeze(0).to(next(model.model.parameters()).device)


with torch.no_grad():
    _ = model.model(im)


head = model.model.model[-1]
depth_tensor = None


if hasattr(head, "last_depth") and isinstance(head.last_depth, torch.Tensor) and head.last_depth.ndim == 4:
    depth_tensor = head.last_depth

elif hasattr(head, "depth_out") and isinstance(head.depth_out, torch.Tensor) and head.depth_out.ndim == 4:
    depth_tensor = head.depth_out
elif hasattr(head, "depth") and isinstance(head.depth, torch.Tensor) and head.depth.ndim == 4:
    depth_tensor = head.depth

else:
    with torch.no_grad():
        out_raw = model.model(im)
    if isinstance(out_raw, (list, tuple)) and len(out_raw) >= 2:
        maybe = out_raw[1]
        if isinstance(maybe, torch.Tensor) and maybe.ndim == 4:
            depth_tensor = maybe

if depth_tensor is None:
    raise RuntimeError("Depth output is still unavailable. Please confirm that `self.last_depth = depth_fused` is set in the DDetect.forward inference branch.")


depth_map = depth_tensor[0, 0].detach().cpu().numpy()

d_min, d_mean, d_max = float(depth_map.min()), float(depth_map.mean()), float(depth_map.max())
print(f"[Depth map stats] min={d_min:.4f}, mean={d_mean:.4f}, max={d_max:.4f}")




res_det = model.predict(source=img, save=False, imgsz=640, conf=0.25, iou=0.45, verbose=False)
boxes_xyxy = res_det[0].boxes.xyxy.cpu().numpy() if len(res_det) else []



H_d, W_d = depth_map.shape
stride_x = 640 / W_d
stride_y = 640 / H_d

import csv
csv_path = SAVE_DIR / "detections_xywhd.csv"
rows = []

vis_img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR).copy()

for i, box in enumerate(boxes_xyxy):
    x1, y1, x2, y2 = box.astype(int)

    xs1, ys1 = max(int(x1 / stride_x), 0), max(int(y1 / stride_y), 0)
    xs2, ys2 = min(int(x2 / stride_x), W_d), min(int(y2 / stride_y), H_d)
    roi = depth_map[ys1:ys2, xs1:xs2]
    d_val = float(roi.mean()) if roi.size else float("nan")


    x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)
    rows.append([i, x, y, w, h, d_val])


    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 128, 0), 2)

    cv2.putText(vis_img, f"d={d_val:.3f}", (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis_img, f"d={d_val:.3f}", (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


cv2.imwrite(str(SAVE_DIR / "rgb_with_d.jpg"), vis_img)
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    csvw = csv.writer(f)
    csvw.writerow(["id", "x", "y", "w", "h", "d"])
    csvw.writerows(rows)

print(f"Detection image with d saved: {SAVE_DIR / 'rgb_with_d.jpg'}")
print(f"Numeric table saved: {csv_path}")


np.save(str(SAVE_DIR / "depth_map_raw.npy"), depth_map)
print(f"Raw depth matrix saved as: {SAVE_DIR / 'depth_map_raw.npy'}")

depth_vis = colorize_depth(depth_map)
out_path  = SAVE_DIR / "depth_vis.jpg"
cv2.imwrite(str(out_path), depth_vis)
print("Depth visualization saved:", out_path)
