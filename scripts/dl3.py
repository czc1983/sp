#!/usr/bin/env python3
import os, sys, time, threading, requests

JOBS = [
    ("https://hf-mirror.com/t8star/minimax-h3-4step-turbo-loras-comfyui-exp/resolve/main/minimax_h3_turbo_4%E6%AD%A5%E5%8A%A0%E9%80%9F_comfyui.safetensors",
     "/root/ComfyUI-H3/models/loras/minimax_h3_turbo_4step_comfyui.safetensors"),
    ("https://hf-mirror.com/t8star/minimax-h3-4step-turbo-loras-comfyui-exp/resolve/main/minimax_h3_turbo_4%E6%AD%A5%E5%8A%A0%E9%80%9Fema_comfyui.safetensors",
     "/root/ComfyUI-H3/models/loras/minimax_h3_turbo_4step_ema_comfyui.safetensors"),
    ("https://hf-mirror.com/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors",
     "/root/ComfyUI-H3/models/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors"),
]
THREADS = 12
CHUNK = 4 * 1024 * 1024

def fetch_size(url):
    r = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60)
    cr = r.headers.get("content-range", "")
    r.close()
    return int(cr.split("/")[-1])

def dl_seg(url, fd, start, end, idx, progress, lock):
    want = end - start + 1
    for attempt in range(8):
        try:
            got = progress[idx] if progress[idx] > 0 else 0
            with requests.get(url, headers={"Range": f"bytes={start+got}-{end}"}, stream=True, timeout=120) as r:
                r.raise_for_status()
                off = start + got
                for chunk in r.iter_content(CHUNK):
                    os.pwrite(fd, chunk, off)
                    off += len(chunk)
                    with lock:
                        progress[idx] = off - start
            if off - start != want:
                raise IOError(f"short read {off-start} != {want}")
            return
        except Exception as e:
            print(f"seg{idx} retry{attempt}: {e}", flush=True)
            time.sleep(3 * (attempt + 1))
    raise SystemExit(f"seg{idx} FAILED permanently")

for url, dest in JOBS:
    name = dest.split("/")[-1]
    print(f"== start {name} {time.strftime('%T')} ==", flush=True)
    total = fetch_size(url)
    print(f"size {total/1e9:.2f}GB", flush=True)
    tmp = dest + ".part"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT)
    os.ftruncate(fd, total)
    seg = (total + THREADS - 1) // THREADS
    progress = [0] * THREADS
    lock = threading.Lock()
    threads = []
    t0 = time.time()
    for i in range(THREADS):
        s = i * seg
        e = min(total - 1, s + seg - 1)
        if s > e:
            progress[i] = -1
            continue
        t = threading.Thread(target=dl_seg, args=(url, fd, s, e, i, progress, lock), daemon=True)
        t.start()
        threads.append(t)
    while any(t.is_alive() for t in threads):
        time.sleep(30)
        with lock:
            done = sum(p for p in progress if p > 0)
        el = max(time.time() - t0, 0.1)
        print(f"{name}: {done/1e9:.2f}/{total/1e9:.2f}GB {done/el/1e6:.1f}MB/s", flush=True)
    for t in threads:
        t.join()
    os.close(fd)
    os.rename(tmp, dest)
    print(f"== done {name} {time.strftime('%T')} ==", flush=True)
print("ALL_DOWNLOADS_DONE", flush=True)
