
# import the VAE modules
import models.vae as vae

# import base packages
import torch
from PIL import Image
from typing import Optional, Any



# a class that represents a image and/or noise
class ImageNoise:
    # human representations i.e. the human understandable types of latents
    seed: Optional[int]
    image: Image.Image

    # machine representations not relevant as no ideas on how to manipulate the latents (in useful ways)
    latents: Any

    def __init__(self, seed: Optional[int] = None, latents: Any = None) -> None:
        if seed != None and latents != None:
            raise Exception("Seed and latents can't both be set as conflicting representations")
        elif latents != None:
            # get the image
            image = vae.get_PIL_image(latents)

            # set the attributes
            self.seed = None
            self.image = image
            self.latents = latents
        else:
            # noise generation parameters
            height = 512    # default height of Stable Diffusion
            width = 512     # default width of Stable Diffusion
            generator = None
            if seed != None:
                generator = torch.manual_seed(seed)
            
            # get the image
            noise_latents = vae.get_noise_latents(height, width, generator)

            # set the attributes
            self.seed = seed
            self.image = None
            self.latents = noise_latents