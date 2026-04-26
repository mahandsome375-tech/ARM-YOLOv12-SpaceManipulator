
import argparse, socket, json, time, math
ap = argparse.ArgumentParser()
ap.add_argument("--W", type=int, default=1280)
ap.add_argument("--H", type=int, default=720)
ap.add_argument("--host", type=str, default="127.0.0.1")
ap.add_argument("--port", type=int, default=9000)
ap.add_argument("--fps", type=int, default=30)
a = ap.parse_args()
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); addr=(a.host,a.port)
t0 = time.time()
print(f"[fake_sender] udp://{a.host}:{a.port} {a.W}x{a.H} fps={a.fps}")
try:
    while True:
        t=time.time()-t0
        cx=a.W/2+200*math.sin(2*math.pi*0.2*t); cy=a.H/2+120*math.cos(2*math.pi*0.15*t)
        w=160+40*math.sin(2*math.pi*0.3*t);    h=120+30*math.cos(2*math.pi*0.27*t)
        d=1.2+0.2*math.sin(2*math.pi*0.1*t)
        msg={"ver":1,"t":time.time(),"cam":{"W":a.W,"H":a.H},
             "det":[{"id":None,"cx":float(cx),"cy":float(cy),"w":float(w),"h":float(h),
                     "d":float(d),"conf":0.9,"sigma_d":None,"yaw":None}]}
        sock.sendto(json.dumps(msg).encode("utf-8"), addr)
        time.sleep(1.0/max(a.fps,1))
finally:
    sock.close()
