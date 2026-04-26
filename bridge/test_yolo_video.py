"""
test_yolo_video.py

Function:
1) Use the trained YOLO weights to run frame-by-frame inference on the input video.
2) Draw bounding boxes, class names, and confidence scores in the window to check whether the base station is detected stably.
3) Save the center trajectory [cx, cy] and timestamp t of the base-station target to a .npy file,
   which can later be used directly as visual input for RL or as a fitted trajectory.

Before use, modify the following two paths:
    WEIGHT_PATH  = your best.pt
    VIDEO_PATH   = your video path
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import os




WEIGHT_PATH = r"best.pt"
VIDEO_PATH  = r"satellite_motion.mp4"


SAVE_TRAJ_PATH = r"bridge\station_traj.npy"
SAVE_VIDEO_PATH = r"bridge\station_detected.mp4"






CONF_THRES = 0.4



TARGET_CLASS_NAME = None


TARGET_CLASS_ID = None


def main():

    if not os.path.exists(WEIGHT_PATH):
        raise FileNotFoundError(f"Weight file not found: {WEIGHT_PATH}")

    print(f"[INFO] Loading YOLO model: {WEIGHT_PATH}")
    model = YOLO(WEIGHT_PATH)


    try:
        class_names = model.model.names
    except AttributeError:
        class_names = model.names

    print("[INFO] Model class indices:")
    for cid, name in class_names.items():
        print(f"  id={cid}: {name}")


    global TARGET_CLASS_ID
    if TARGET_CLASS_ID is None and TARGET_CLASS_NAME is not None:

        for cid, name in class_names.items():
            if str(name).lower() == str(TARGET_CLASS_NAME).lower():
                TARGET_CLASS_ID = cid
                print(f"[INFO] Matched the base-station class by name: id={cid}, name={name}")
                break
        if TARGET_CLASS_ID is None:
            print("[WARN] No class named '{}' was found. Please check the dataset class definitions."
                  .format(TARGET_CLASS_NAME))


    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video: {}".format(VIDEO_PATH))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    dt = 1.0 / fps


    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_writer = cv2.VideoWriter(SAVE_VIDEO_PATH, fourcc, fps, (width, height))

    print(f"[INFO] Video resolution: {width} x {height}, fps={fps:.2f}")



    traj_list = []

    frame_idx = 0
    t = 0.0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video reading completed.")
            break

        frame_idx += 1
        t = frame_idx * dt



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


            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)


            if conf > best_conf:
                best_conf = conf
                best_box = (x1, y1, x2, y2, cx, cy, cls_id, conf)


        if best_box is not None:
            x1, y1, x2, y2, cx, cy, cls_id, conf = best_box


            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            label = f"{class_names[cls_id]} {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )


            cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)


            traj_list.append([t, cx, cy])


        elapsed = time.time() - t0
        fps_est = frame_idx / max(elapsed, 1e-6)
        cv2.putText(
            frame,
            f"t={t:.2f}s  FPS={fps_est:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("YOLO Station Detection", frame)
        out_writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            print("[INFO] Interrupted by user.")
            break

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()


    if len(traj_list) > 0:
        traj_arr = np.array(traj_list, dtype=np.float32)
        np.save(SAVE_TRAJ_PATH, traj_arr)
        print(f"[INFO] Trajectory saved to: {SAVE_TRAJ_PATH}")
        print("      Trajectory format is [t, cx, cy], with units of seconds and pixel coordinates")
    else:
        print("[WARN] No base-station target was detected, so the trajectory was not saved.")


if __name__ == "__main__":
    main()
