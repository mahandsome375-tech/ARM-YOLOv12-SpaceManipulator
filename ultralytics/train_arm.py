# -*- coding: utf-8 -*-
"""
ARM-YOLOv12 fully automatic training script (two stages)
Path: train_arm.py
"""

from ultralytics import YOLO
import os


def main():

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.chdir(r".")


    DATA_PATH = r"ultralytics/cfg/datasets/arm_dataset.yaml"
    MODEL_CFG = r"ultralytics/cfg/models/12/yolo12.yaml"
    DEVICE = 0
    IMGSZ = 640
    BATCH = 1
    EPOCHS_STAGE0 = 300
    EPOCHS_STAGE1 = 120


    print("\n==============================")
    print("Stage 0: training YOLO + depth module ...")
    print("==============================")

    model = YOLO(MODEL_CFG)
    model.train(
        data=DATA_PATH,
        imgsz=IMGSZ,
        epochs=EPOCHS_STAGE0,
        batch=BATCH,
        device=DEVICE,
        project="runs/detect",
        name="stage0_depth22",

        loss_phi=0.0,
        loss_tau=0.0,
        loss_sigma=0.0,

    )


    print("\n==============================")
    print("Stage 1: loading stage-0 weights and training phi / tau / sigma modules ...")
    print("==============================")

    model = YOLO(r"best.pt")
    model.train(
        data=DATA_PATH,
        imgsz=IMGSZ,
        epochs=EPOCHS_STAGE1,
        batch=BATCH,
        device=DEVICE,
        project="runs/detect",
        name="stage1_pose_uncert",

        loss_phi=0.5,
        loss_tau=1.0,
        loss_sigma=0.25,

    )


    print("\n==============================")
    print("Validating stage-1 results ...")
    print("==============================")

    model.val(data=DATA_PATH, imgsz=IMGSZ, device=DEVICE)

    print("\nTraining and validation completed!")
    print("Weight path: best.pt")


if __name__ == "__main__":
    main()
