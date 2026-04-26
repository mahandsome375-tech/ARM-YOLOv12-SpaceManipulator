




import argparse, socket, json, time, math
from ultralytics import YOLO
import cv2


import numpy as np
import torch
import torch.nn.functional as F



import os, sys      
DEFAULT_WEIGHTS = r"best.pt"

def fetch_aux_maps(model, H, W):
    """
    Return a dictionary:
      {
        "tau":     np.ndarray(H, W) or None,
        "phi_sin": np.ndarray(H, W) or None,
        "phi_cos": np.ndarray(H, W) or None,
        "logvar":  np.ndarray(H, W) or None
      }
    """
    out = {"tau": None, "phi_sin": None, "phi_cos": None, "logvar": None}
    try:
        head = model.model[-1]

        tau     = getattr(head, "last_tau",     None)
        phi     = getattr(head, "last_phi",     None)
        logvar  = getattr(head, "last_logvar",  None)

        def up_to_numpy(t):

            t = F.interpolate(t.detach().float(), size=(H, W),
                              mode="bilinear", align_corners=False)[0].cpu()
            return t

        if tau is not None and tau.ndim == 4:
            out["tau"] = up_to_numpy(tau).squeeze(0).numpy()

        if phi is not None and phi.ndim == 4 and phi.shape[1] >= 2:
            phi_up = up_to_numpy(phi)
            out["phi_sin"] = phi_up[0].numpy()
            out["phi_cos"] = phi_up[1].numpy()

        if logvar is not None and logvar.ndim == 4:
            out["logvar"] = up_to_numpy(logvar).squeeze(0).numpy()

    except Exception:

        pass

    return out


def extract_det(res, maps=None):
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return None

    i = int(boxes.conf.argmax().item())
    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1), (y2 - y1)
    conf = float(boxes.conf[i].item())

    d = None
    try:
        arr = boxes.data[i].tolist()
        if len(arr) > 6:
            d = float(arr[-1])
    except Exception:
        pass
    if d is None:
        area = max(w * h, 1.0)
        d = 1.0 / math.sqrt(area)


    tau_val = None
    sin_phi = None
    cos_phi = None
    logvar_val = None
    try:
        if maps:
            H = maps.get("tau", None).shape[0] if isinstance(maps.get("tau", None), np.ndarray) else \
                (maps.get("phi_sin", None).shape[0] if isinstance(maps.get("phi_sin", None), np.ndarray) else None)
            W = maps.get("tau", None).shape[1] if isinstance(maps.get("tau", None), np.ndarray) else \
                (maps.get("phi_sin", None).shape[1] if isinstance(maps.get("phi_sin", None), np.ndarray) else None)

            if H is not None and W is not None:
                ix = int(np.clip(round(cx), 0, W - 1))
                iy = int(np.clip(round(cy), 0, H - 1))

                if isinstance(maps.get("tau", None), np.ndarray):
                    tau_val = float(maps["tau"][iy, ix])
                if isinstance(maps.get("phi_sin", None), np.ndarray):
                    sin_phi = float(maps["phi_sin"][iy, ix])
                if isinstance(maps.get("phi_cos", None), np.ndarray):
                    cos_phi = float(maps["phi_cos"][iy, ix])
                if isinstance(maps.get("logvar", None), np.ndarray):
                    logvar_val = float(maps["logvar"][iy, ix])
    except Exception:
        pass


    return dict(
        id=None,
        cx=float(cx), cy=float(cy), w=float(w), h=float(h),
        d=float(d), conf=conf, sigma_d=None, yaw=None,
        ttc_tau=float(tau_val if tau_val is not None else 0.0),
        orient_sin=float(sin_phi if sin_phi is not None else 0.0),
        orient_cos=float(cos_phi if cos_phi is not None else 0.0),
        unc_logvar=float(logvar_val if logvar_val is not None else 0.0),
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    ap.add_argument("--src", type=str, default="0")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()


    if not os.path.exists(a.weights):
        print(f"[vbc_sender] ERROR: weights not found:\n  {a.weights}")
        print(f"[vbc_sender] weights = {a.weights}")
        sys.exit(1)

    cap = cv2.VideoCapture(0 if (a.src.isdigit() and len(a.src) == 1) else a.src)
    if not cap.isOpened():
        raise SystemExit(f"Unable to open video source: {a.src}")

    model = YOLO(a.weights)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (a.host, a.port)

    print(f"[vbc_sender] udp://{a.host}:{a.port}  imgsz={a.imgsz}  fps={a.fps}")

    interval = 1.0 / max(a.fps, 1)
    ok, frame = cap.read()
    H, W = frame.shape[:2]

    try:
        frame_id = 0

        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                break


            res = model.predict(frame, imgsz=a.imgsz, conf=0.25, verbose=False)[0]


            maps = fetch_aux_maps(model, H, W)


            det = extract_det(res, maps)


            msg = {
                "ver": 1,
                "t": time.time(),
                "cam": {"W": int(W), "H": int(H)},
                "det": [{"x": 100, "y": 200, "w": 50, "h": 60}],

            }
            n = sock.sendto(json.dumps(msg).encode("utf-8"), addr)
            frame_id += 1
            if frame_id % 10 == 0:
                print(f"[vbc_sender] sent {n} bytes to {a.host}:{a.port}, frame #{frame_id}")

            dt = time.time() - t0
            if dt < interval:
                time.sleep(interval - dt)
    except KeyboardInterrupt:
        print(f"\n[vbc_sender] stopped by user. total frames sent: {frame_id}")
    finally:
        cap.release()
        sock.close()


if __name__ == "__main__":
    main()
