import os
from diffusers import StableAudioPipeline
import torch

# Ова автоматски ќе го повлече токенот што го внесе во Settings
hf_token = os.getenv("HF_TOKEN")

model_id = "stabilityai/stable-audio-open-1.0"

# Додај го 'token' параметарот тука
pipe = StableAudioPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    token=hf_token,
    low_cpu_mem_usage=True,  # Штеди системска меморија
    device_map=None         # Важно за ZeroGPU
)