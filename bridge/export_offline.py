
import time, json, argparse
import numpy as np
from scipy.io import savemat





def run_export(num_frames=1000, out='bridge/offline_det.mat'):
    cam = {"fw": 640, "fh": 480, "hfov": 60}
    det_list = []

    for k in range(num_frames):

        det = {"x": 100, "y": 200, "w": 50, "h": 60, "d": 1.8, "tau": 0.32, "phi": 0.12, "sigma": 0.08}
        ts = time.time()

        det_list.append({
            "tt": ts,
            "cam": cam,
            "det": [det]
        })


    MEM = {
        "len": len(det_list),
        "cam": cam,



        "obs": np.array([[d["det"][0]["x"], d["det"][0]["y"], d["det"][0]["w"], d["det"][0]["h"],
                          d["det"][0]["d"], d["det"][0]["tau"], d["det"][0]["phi"], d["det"][0]["sigma"]]
                         for d in det_list], dtype=np.float64)
    }
    savemat(out, {"MEM": MEM})
    print(f"saved to {out}, frames={MEM['len']}, obs.shape={MEM['obs'].shape}")

if __name__ == "__main__":
    import os

    OUT_DIR = os.path.dirname(__file__)
    OUT_FN = os.path.join(OUT_DIR, 'offline_det.mat')
    run_export(num_frames=1500, out=OUT_FN)

