import time
import av
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

VIDEO = "/root/ComfyUI-H3/input/h3_test.mp4"
MODEL = "/root/Qwen-VL/models/models/Qwen--Qwen2.5-VL-7B-Instruct/snapshots/master"

def extract_frames(path, times):
    container = av.open(path)
    frames = {}
    targets = sorted(times)
    for frame in container.decode(video=0):
        t = round(frame.time, 2)
        if targets and t >= targets[0]:
            frames[targets.pop(0)] = frame.to_image()
        if not targets:
            break
    container.close()
    return [frames[t] for t in sorted(frames)]

TIMES = [0.0, 1.0, 2.0, 3.0, 4.0, 4.5]
print("extracting frames...", flush=True)
imgs = extract_frames(VIDEO, TIMES)
print("got", len(imgs), "frames", flush=True)

print("loading model...", flush=True)
t0 = time.time()
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
)
processor = AutoProcessor.from_pretrained(MODEL)
print("model loaded in", round(time.time() - t0, 1), "s", flush=True)

INSTR = """你是一名视频反推分析师。这 6 帧来自同一段约 5 秒的短剧视频（时间分别为 0s/1s/2s/3s/4s/4.5s）。请按秒级时间轴反推画面内容，供视频生成模型复刻使用。严格按以下结构输出，不要遗漏：

【场景与构图】场景类型、空间结构、所有关键物体（特别注意：是否存在镜子/反射面/屏幕等"画中画"元素？如有，说明其位置、镜中内容与镜外内容分别是什么）、光线方向与色调。
【人物】人数、各自位置（左侧/右侧/前景/背景/镜中/镜外）、身份特征、穿着。
【时间轴】按 [0s-1s] [1s-2s] [2s-3s] [3s-4s] [4s-5s] 五拍，分别描述：人物姿态、动作变化、头部朝向、视线落点、表情与情绪、手部位置、遮挡关系。
【镜头】景别、机位、是否运动、景深与虚化情况。
【光影】主光源方向、明暗对比、氛围。"""

messages = [
    {"role": "user", "content": [{"type": "image", "image": img} for img in imgs]
     + [{"type": "text", "text": INSTR}]}
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                   padding=True, return_tensors="pt").to("cuda")

t0 = time.time()
out = model.generate(**inputs, max_new_tokens=1500)
trimmed = out[:, inputs.input_ids.shape[1]:]
result = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
print("inference", round(time.time() - t0, 1), "s", flush=True)
print("=====REVERSE RESULT=====", flush=True)
print(result, flush=True)
