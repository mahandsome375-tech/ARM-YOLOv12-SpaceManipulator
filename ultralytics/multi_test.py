# -*- coding: utf-8 -*-
import argparse, os
import cv2, numpy as np, torch
from ultralytics import YOLO

def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    return x

def save_heatmap(base_img, hm, out_path, alpha=0.45):
    """hm: [H,W] or [1,H,W] float, automatic normalization + pseudo-color overlay"""
    if hm is None:
        return
    if hm.ndim == 3 and hm.shape[0] == 1:
        hm = hm[0]
    h, w = base_img.shape[:2]
    m = hm.copy()
    if np.isnan(m).any():
        m = np.nan_to_num(m, nan=0.0)

    m = m - m.min() if m.max() > m.min() else m
    m = m / (m.max() + 1e-8)
    m = (m * 255).astype(np.uint8)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap(m, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(base_img, 1.0, color, alpha, 0)
    cv2.imwrite(out_path, blended)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="path to .pt")
    ap.add_argument("--img", type=str, required=True, help="a test image path")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--outdir", type=str, default="runs/arm_vis")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)


    img = cv2.imread(args.img)
    assert img is not None, f"Unable to read image: {args.img}"


    m = YOLO(args.weights)


    _ = m.predict(img, imgsz=args.imgsz, conf=0.25, verbose=False)



    extra = m.model.model[-1](m.model.model[-2], return_extra=True)

    depth, tau, sin_phi, cos_phi, logvar = [ _to_numpy(x) for x in extra ]


    bn = os.path.splitext(os.path.basename(args.img))[0]
    save_heatmap(img, depth,  os.path.join(args.outdir, f"{bn}_depth.jpg"))
    save_heatmap(img, tau,    os.path.join(args.outdir, f"{bn}_tau.jpg"))

    save_heatmap(img, sin_phi, os.path.join(args.outdir, f"{bn}_phi_sin.jpg"))
    save_heatmap(img, cos_phi, os.path.join(args.outdir, f"{bn}_phi_cos.jpg"))
    save_heatmap(img, logvar, os.path.join(args.outdir, f"{bn}_logvar.jpg"))


    if sin_phi is not None and cos_phi is not None:

        ang = np.arctan2(sin_phi, cos_phi)
        ang = (ang + np.pi) / (2*np.pi)
        save_heatmap(img, ang, os.path.join(args.outdir, f"{bn}_phi_angle.jpg"))

    print(f"Visualization saved to: {args.outdir}")

if __name__ == "__main__":
    main()
