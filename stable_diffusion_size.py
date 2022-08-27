import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16", use_auth_token=True)

for name in dir(pipe):
    model = getattr(pipe, name)
    if not isinstance(model, torch.nn.Module):
        continue

    model_size = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            model_size += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            model_size += param.numel() * torch.iinfo(param.data.dtype).bits

    print(f"{name} size: {model_size} / bit | {model_size / 8e6:.2f} / MB")
