# -*- coding: utf-8 -*-
"""
test_yolo_video_fourmods.py

Final revised version for the continuously approaching target scenario.

Function:
1) Use YOLO to detect the target and output the bbox.
2) Use ROI segmentation and PCA principal-axis estimation to estimate the target 2D apparent pose phi.
3) Estimate depth d using rotation-compensated scale.
4) Apply a continuously approaching monotonic constraint to d to remove pose-induced false rebound.
5) Estimate tau = d / (-d_dot) from the smoothed d.
6) Construct sigma from confidence, depth residual, and pose residual.
7) Output and save [t, cx, cy, d, tau, phi, sigma].

Notes:
- Keep the definitions of the four variables unchanged:
    d     : distance/depth, smaller when closer
    tau   : time to contact
    phi   : relative pose angle
    sigma : uncertainty
- This version is suitable for the current video where the target continuously approaches.
"""

import os
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO





WEIGHT_PATH = r"best.pt"
VIDEO_PATH = r"satellite_motion.mp4"

SAVE_TRAJ_PATH = r"bridge\station_traj_fourmods.npy"
SAVE_VIDEO_PATH = r"bridge\station_detected_fourmods.mp4"





CONF_THRES = 0.4

TARGET_CLASS_NAME = None
TARGET_CLASS_ID = None


K_DEPTH = 220.0


ALPHA_D = 0.25


TAU_REG_WINDOW = 10
TAU_WARMUP_FRAMES = 6
TAU_MAX = 99.0
TAU_MIN_APPROACH_RATE = 1e-3
ALPHA_TAU = 0.25


ALPHA_PHI = 0.22
ALPHA_SIGMA = 0.28


MIN_CONTOUR_AREA = 15


SIGMA_W_CONF = 0.50
SIGMA_W_DRES = 0.30
SIGMA_W_PHIRES = 0.20
SIGMA_W_SHAPERES = 0.15

WINDOW_NAME = "YOLO Station Detection (4 modules - corrected)"
FONT = cv2.FONT_HERSHEY_SIMPLEX





def wrap_to_pi(angle: float) -> float:
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return float(angle)


def lowpass(prev_val: Optional[float], new_val: float, alpha: float = 0.25) -> float:
    if prev_val is None:
        return float(new_val)
    return float(alpha * new_val + (1.0 - alpha) * prev_val)


def angle_smooth(prev_angle: Optional[float], new_angle: float, alpha: float = 0.25) -> float:
    if prev_angle is None:
        return float(new_angle)

    cands = [new_angle - 2.0 * np.pi, new_angle, new_angle + 2.0 * np.pi]
    aligned = min(cands, key=lambda a: abs(a - prev_angle))
    return float(alpha * aligned + (1.0 - alpha) * prev_angle)


def draw_text(frame, text, org, color=(0, 255, 0), scale=0.58, thickness=2):
    cv2.putText(
        frame,
        text,
        org,
        FONT,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def fit_line_slope(t_hist, y_hist):
    if len(t_hist) < 3:
        return None

    t = np.asarray(t_hist, dtype=np.float64)
    y = np.asarray(y_hist, dtype=np.float64)

    t0 = t.mean()
    y0 = y.mean()
    tt = t - t0
    yy = y - y0

    denom = np.sum(tt * tt)
    if denom < 1e-12:
        return None

    slope = np.sum(tt * yy) / denom
    return float(slope)





def get_roi(frame, x1, y1, x2, y2):
    h_img, w_img = frame.shape[:2]
    x1i = max(0, int(x1))
    y1i = max(0, int(y1))
    x2i = min(w_img - 1, int(x2))
    y2i = min(h_img - 1, int(y2))

    if x2i <= x1i or y2i <= y1i:
        return None, (x1i, y1i, x2i, y2i)

    roi = frame[y1i:y2i, x1i:x2i]
    return roi, (x1i, y1i, x2i, y2i)


def segment_target_in_roi(roi):
    if roi is None or roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < MIN_CONTOUR_AREA:
        return None

    mask = np.zeros_like(bw)
    cv2.drawContours(mask, [c], -1, 255, thickness=-1)
    return mask


def pca_orientation_and_scale(mask) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[np.ndarray]]:
    """
    Output:
    - phi_raw : principal-axis direction angle
    - scale   : rotation-compensated scale
    - resid   : orthogonal residual
    - axis    : principal-axis direction vector
    """
    if mask is None:
        return None, None, None, None

    ys, xs = np.where(mask > 0)
    if len(xs) < 6:
        return None, None, None, None

    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    mean = pts.mean(axis=0, keepdims=True)
    pts_c = pts - mean

    cov = (pts_c.T @ pts_c) / max(len(pts_c) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    main_axis = eigvecs[:, 0]
    sec_axis = eigvecs[:, 1]

    vx, vy = main_axis[0], main_axis[1]
    phi_raw = np.arctan2(vy, vx)

    proj_main = pts_c @ main_axis
    proj_sec = pts_c @ sec_axis

    len_main = float(proj_main.max() - proj_main.min() + 1e-6)
    len_sec = float(proj_sec.max() - proj_sec.min() + 1e-6)

    scale = np.sqrt(len_main * len_sec)
    resid = float(np.mean(np.abs(proj_sec)))

    return float(phi_raw), float(scale), float(resid), main_axis.astype(np.float64)





def estimate_depth_from_scale(scale, k_depth=220.0):
    """
    d: distance/depth, smaller when closer
    """
    scale = max(scale, 1e-6)
    return float(k_depth / scale)


def enforce_monotonic_approach(prev_d: Optional[float], d_candidate: float) -> float:
    """
    For the current video scenario, the target continuously approaches.
    Therefore, d should be monotonically non-increasing, allowing smaller values but not sudden increases.
    """
    if prev_d is None:
        return float(d_candidate)
    return float(min(prev_d, d_candidate))


def estimate_tau_from_history(t_hist, d_hist, tau_max=99.0, min_approach_rate=1e-3):
    """
    tau = d / (-d_dot)
    Valid only when d_dot < 0.
    """
    if len(t_hist) < 3 or len(d_hist) < 3:
        return float(tau_max), None

    d_curr = float(d_hist[-1])
    d_dot = fit_line_slope(t_hist, d_hist)

    if d_dot is None:
        return float(tau_max), None

    if d_dot < -min_approach_rate:
        tau_val = d_curr / (-d_dot)
        tau_val = min(max(tau_val, 0.0), tau_max)
    else:
        tau_val = tau_max

    return float(tau_val), float(d_dot)


def estimate_sigma(conf, d_meas, d_smooth, phi_meas, phi_smooth, contour_resid,
                   prev_sigma=None, alpha=0.28):
    conf_term = max(0.0, 1.0 - float(conf))

    d_res = 0.0 if d_smooth is None else abs(float(d_meas) - float(d_smooth))
    phi_res = 0.0 if phi_smooth is None else abs(wrap_to_pi(float(phi_meas) - float(phi_smooth)))
    shape_res = 0.0 if contour_resid is None else float(contour_resid)

    d_res_n = np.tanh(d_res)
    phi_res_n = np.tanh(phi_res)
    shape_res_n = np.tanh(0.05 * shape_res)

    sigma_raw = (
        SIGMA_W_CONF * conf_term
        + SIGMA_W_DRES * d_res_n
        + SIGMA_W_PHIRES * phi_res_n
        + SIGMA_W_SHAPERES * shape_res_n
    )

    sigma_raw = max(0.01, min(float(sigma_raw), 5.0))
    sigma_val = lowpass(prev_sigma, sigma_raw, alpha=alpha)
    return float(sigma_val)





def main():
    global TARGET_CLASS_ID

    if not os.path.exists(WEIGHT_PATH):
        raise FileNotFoundError("Weight file not found: {}".format(WEIGHT_PATH))

    print("[INFO] Loading YOLO model: {}".format(WEIGHT_PATH))
    model = YOLO(WEIGHT_PATH)

    try:
        class_names = model.model.names
    except AttributeError:
        class_names = model.names

    print("[INFO] Model class indices:")
    for cid, name in class_names.items():
        print("  id={}: {}".format(cid, name))

    if TARGET_CLASS_ID is None and TARGET_CLASS_NAME is not None:
        for cid, name in class_names.items():
            if str(name).lower() == str(TARGET_CLASS_NAME).lower():
                TARGET_CLASS_ID = cid
                print("[INFO] Matched the target class by name: id={}, name={}".format(cid, name))
                break
        if TARGET_CLASS_ID is None:
            print("[WARN] Class name {} was not found. The highest-confidence box will be used by default.".format(TARGET_CLASS_NAME))

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError("Video file not found: {}".format(VIDEO_PATH))

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video: {}".format(VIDEO_PATH))

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video <= 0:
        fps_video = 30.0
    dt_video = 1.0 / fps_video

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_writer = cv2.VideoWriter(SAVE_VIDEO_PATH, fourcc, fps_video, (width, height))

    print("[INFO] Video resolution: {} x {}, fps={:.2f}".format(width, height, fps_video))

    traj_list = []


    prev_d_smooth = None
    prev_phi_smooth = None
    prev_sigma_smooth = None
    prev_tau_smooth = None
    prev_axis = None

    d_hist = deque(maxlen=TAU_REG_WINDOW)
    t_hist = deque(maxlen=TAU_REG_WINDOW)

    frame_idx = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video reading completed.")
            break

        frame_idx += 1
        t = frame_idx * dt_video

        results = model(frame, verbose=False)[0]

        best_box = None
        best_conf = 0.0

        for box in results.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())

            if conf < CONF_THRES:
                continue

            if TARGET_CLASS_ID is not None and cls_id != TARGET_CLASS_ID:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

            if conf > best_conf:
                best_conf = conf
                best_box = (x1, y1, x2, y2, cx, cy, cls_id, conf)

        if best_box is not None:
            x1, y1, x2, y2, cx, cy, cls_id, conf = best_box

            roi, _ = get_roi(frame, x1, y1, x2, y2)
            mask = segment_target_in_roi(roi)
            phi_meas, scale_meas, contour_resid, axis = pca_orientation_and_scale(mask)


            if scale_meas is None:
                w = max(x2 - x1, 1e-6)
                h = max(y2 - y1, 1e-6)
                scale_meas = np.sqrt(w * h)
                contour_resid = 0.0

            if phi_meas is None:
                phi_meas = np.arctan2((y2 - y1), (x2 - x1))


            if axis is not None and prev_axis is not None:
                if float(np.dot(axis, prev_axis)) < 0.0:
                    axis = -axis
                    phi_meas = np.arctan2(axis[1], axis[0])

            phi_meas = wrap_to_pi(phi_meas)
            phi_val = angle_smooth(prev_phi_smooth, phi_meas, alpha=ALPHA_PHI)

            if axis is not None:
                prev_axis = axis.copy()


            d_meas = estimate_depth_from_scale(scale_meas, k_depth=K_DEPTH)
            d_lp = lowpass(prev_d_smooth, d_meas, alpha=ALPHA_D)
            d_val = enforce_monotonic_approach(prev_d_smooth, d_lp)


            t_hist.append(float(t))
            d_hist.append(float(d_val))

            tau_raw, d_dot = estimate_tau_from_history(
                t_hist,
                d_hist,
                tau_max=TAU_MAX,
                min_approach_rate=TAU_MIN_APPROACH_RATE,
            )

            if frame_idx < TAU_WARMUP_FRAMES:
                tau_raw = TAU_MAX

            tau_val = lowpass(prev_tau_smooth, tau_raw, alpha=ALPHA_TAU)
            prev_tau_smooth = tau_val

            sigma_val = estimate_sigma(
                conf=conf,
                d_meas=d_meas,
                d_smooth=d_val,
                phi_meas=phi_meas,
                phi_smooth=phi_val,
                contour_resid=contour_resid,
                prev_sigma=prev_sigma_smooth,
                alpha=ALPHA_SIGMA,
            )

            prev_d_smooth = d_val
            prev_phi_smooth = phi_val
            prev_sigma_smooth = sigma_val


            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            label_main = "{} {:.2f}".format(class_names[cls_id], conf)
            draw_text(frame, label_main, (int(x1), int(y1) - 5), color=(0, 255, 0), scale=0.70, thickness=2)

            extra_text = "d={:.2f}   tau={:.2f}   phi={:.2f}   sigma={:.2f}".format(
                d_val, tau_val, phi_val, sigma_val
            )
            draw_text(frame, extra_text, (int(x1), int(y2) + 18), color=(0, 255, 0), scale=0.58, thickness=2)

            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)

            traj_list.append([
                float(t),
                float(cx),
                float(cy),
                float(d_val),
                float(tau_val),
                float(phi_val),
                float(sigma_val),
            ])

        elapsed = time.time() - t_start
        fps_est = frame_idx / max(elapsed, 1e-6)
        draw_text(
            frame,
            "t={:.2f}s   FPS={:.1f}".format(t, fps_est),
            (10, 30),
            color=(255, 255, 255),
            scale=0.80,
            thickness=2,
        )

        cv2.imshow(WINDOW_NAME, frame)
        out_writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            print("[INFO] Interrupted by user.")
            break

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()

    if len(traj_list) > 0:
        traj_arr = np.array(traj_list, dtype=np.float32)
        np.save(SAVE_TRAJ_PATH, traj_arr)
        print("[INFO] Trajectory saved to: {}".format(SAVE_TRAJ_PATH))
        print("      Each row format: [t, cx, cy, d, tau, phi, sigma]")
        print("[INFO] Trajectory shape = {}".format(traj_arr.shape))
    else:
        print("[WARN] No target was detected, so the trajectory was not saved.")


if __name__ == "__main__":
    main()
