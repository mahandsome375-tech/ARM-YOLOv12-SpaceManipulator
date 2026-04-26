
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO


W   = r"best.pt"
IMG = r"dataset\images\train\00001.jpg"

SAVE_DIR = Path(r"runs\detect\train_82\predict_depth")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def colorize_depth(d: np.ndarray) -> np.ndarray:
    d = d.astype(np.float32)
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    d = (d - lo) / (hi - lo + 1e-6)
    d = np.clip(d, 0, 1)
    return cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)


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
depth_vis = colorize_depth(depth_map)
out_path  = SAVE_DIR / "depth_vis.jpg"
cv2.imwrite(str(out_path), depth_vis)
print("Depth visualization saved:", out_path)
