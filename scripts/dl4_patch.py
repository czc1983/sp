#!/usr/bin/env python3
"""扫描 .part 稀疏文件的零块找出缺失区间，分段补齐后转正。

原理: 每个下载线程在各自固定段内顺序写入，未写到的地方是稀疏零。
段内找到第一个全零 4MB 块即视为缺失起点（回退 8MB 保险），其后全部重下。
"""
import os, sys, time, threading, requests

URL = "https://hf-mirror.com/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors"
PART = "/root/ComfyUI-H3/models/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors.part"
DEST = PART[:-5]
THREADS = 12
SCAN = 256 * 1024  # 细粒度扫零：能捕捉被杀进程留下的半写 4MB 块尾部零区
BACKOFF = 8 * 1024 * 1024

total = os.path.getsize(PART)
seg = (total + THREADS - 1) // THREADS
fd = os.open(PART, os.O_RDWR)

def block_zero(off, size):
    data = os.pread(fd, size, off)
    return data.count(0) == len(data)

holes = []  # (start, end) inclusive
for i in range(THREADS):
    s = i * seg
    e = min(total, s + seg)
    if s >= e:
        continue
    # 段内前向扫第一个全零块
    first_zero = None
    off = s
    while off < e:
        size = min(SCAN, e - off)
        if size == SCAN and block_zero(off, size):
            first_zero = off
            break
        off += SCAN
    if first_zero is None:
        print(f"seg{i}: complete", flush=True)
        continue
    start = max(s, first_zero - BACKOFF)
    holes.append((start, e - 1))
    print(f"seg{i}: hole {start}-({e-1}) = {(e-start)/1e6:.0f}MB", flush=True)

if not holes:
    os.close(fd)
    os.rename(PART, DEST)
    print("NO_HOLES -> renamed, PATCH_DONE", flush=True)
    sys.exit(0)

def dl(start, end, idx):
    for attempt in range(10):
        try:
            with requests.get(URL, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(30, 60)) as r:
                r.raise_for_status()
                off = start
                for chunk in r.iter_content(4 * 1024 * 1024):
                    os.pwrite(fd, chunk, off)
                    off += len(chunk)
            if off - start != end - start + 1:
                raise IOError(f"short {off-start}")
            print(f"hole{idx} filled {(end-start+1)/1e6:.0f}MB", flush=True)
            return True
        except Exception as ex:
            print(f"hole{idx} retry{attempt}: {ex}", flush=True)
            time.sleep(2 * (attempt + 1))
    return False

t0 = time.time()
# 把每个洞再切成 32MB 子块并行下（单流会被 hf-mirror 限速到 0.6MB/s）
SUB = 32 * 1024 * 1024
subtasks = []
for idx, (s, e) in enumerate(holes):
    off = s
    while off <= e:
        subtasks.append((off, min(e, off + SUB - 1), idx))
        off += SUB
print(f"{len(subtasks)} sub-ranges to fill", flush=True)
threads = []
for (s, e, idx) in subtasks:
    t = threading.Thread(target=dl, args=(s, e, idx), daemon=True)
    t.start()
    threads.append(t)
for t in threads:
    t.join()

# 复检: 剩余零块向源站再要一次，源站也是零才算真数据
bad = 0
for i in range(THREADS):
    s = i * seg
    e = min(total, s + seg)
    off = s
    while off < e:
        size = min(SCAN, e - off)
        if size == SCAN and block_zero(off, size):
            with requests.get(URL, headers={"Range": f"bytes={off}-{off+size-1}"}, timeout=(30, 120)) as r:
                src = r.content
            if len(src) != size:
                print(f"verify fetch mismatch at {off}", flush=True)
                bad += 1
            elif src.count(0) == len(src):
                pass  # 源站也是零，真数据
            else:
                os.pwrite(fd, src, off)
                print(f"verify: fixed block at {off}", flush=True)
        off += SCAN
os.close(fd)
if bad:
    print("PATCH_FAILED", flush=True)
    sys.exit(1)
os.rename(PART, DEST)
print(f"PATCH_DONE in {time.time()-t0:.0f}s", flush=True)
