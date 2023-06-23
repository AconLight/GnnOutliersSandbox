from PIL.Image import Image
from huggingface_hub import login
from diffusers import DiffusionPipeline
import torch
import os
from dotenv import load_dotenv
load_dotenv()

prompt = "Galactus playing soccer with planet earth instead of a ball."

login(os.environ['access_token'])
pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
generator = torch.Generator(device="cuda").manual_seed(8)
image = pipeline(prompt, generator=generator).images[0]
image.show()


