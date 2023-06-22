
# import diffusion model
from diffusers import UNet2DConditionModel, PNDMScheduler, AutoencoderKL

# import CLIP model
from transformers import CLIPTokenizer, CLIPTextModel, CLIPProcessor, CLIPModel

# import base packages
from typing import Any



# global variables for the unet i.e. the actual diffusion
unet: Any = None
scheduler: Any = None

# global variables for the VAE model
vae: Any = None

# global variables for the CLIP model
clip_tokenizer: Any = None
clip_text_encoder: Any = None
clip_processor: Any = None
clip_model: Any = None

# function that loads the model unless it has already been loaded
loaded_model: bool = False
def load_model() -> None:
    # set loaded_model to global
    global loaded_model
    
    # set the global variables to global
    global unet
    global scheduler

    global vae

    global clip_tokenizer
    global clip_text_encoder

    global clip_processor
    global clip_model

    # check if the global variables have been initialized and if not then initialize them
    if not loaded_model:
        print("Loading diffusion model... (this should only happen once!)")

        diffusion_version = "CompVis/stable-diffusion-v1-4"
        clip_version = "openai/clip-vit-large-patch14"
        
        # initialize the diffusion module
        unet = UNet2DConditionModel.from_pretrained(diffusion_version, subfolder="unet")
        unet.to("cuda")
        scheduler = PNDMScheduler.from_pretrained(diffusion_version, subfolder="scheduler")
        
        # initialize the VAE model
        vae = AutoencoderKL.from_pretrained(diffusion_version, subfolder="vae")
        vae.to("cuda")

        # initialize the CLIP model
        clip_tokenizer = CLIPTokenizer.from_pretrained(clip_version)
        clip_text_encoder = CLIPTextModel.from_pretrained(clip_version)
        clip_text_encoder.to("cuda")

        clip_processor = CLIPProcessor.from_pretrained(clip_version)
        clip_model = CLIPModel.from_pretrained(clip_version)
        clip_model.to("cuda")

        # set loaded model
        loaded_model = True