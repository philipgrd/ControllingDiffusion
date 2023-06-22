
# import models
import models.models_pipelines as models_pipelines

# import base packages
from PIL import Image
import torch
from typing import Any



# function that generates the noise latents
def get_noise_latents(height: int, width: int, generator: Any) -> Any:
    # check if VAE model is not None and if then create it
    models_pipelines.load_model()
    
    # get the noise latents
    noise_latents = torch.randn((1, models_pipelines.unet.in_channels, height // 8, width // 8), generator=generator)

    # move the text noise latents to the GPU
    noise_latents = noise_latents.to("cuda")
    
    return noise_latents

def get_PIL_image(latents: Any) -> Any:
    # check if VAE model is not None and if then create it
    models_pipelines.load_model()
    
    # scale the latents for the VAE model
    latents = 1 / 0.18215 * latents

    # get the image from the latents
    with torch.no_grad():
        image_tensor = models_pipelines.vae.decode(latents).sample
    
    # convert the image to a PIL image and return it
    image = (image_tensor / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images[0]