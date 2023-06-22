
# import models
import models.models_pipelines as models_pipelines
import models.clip as clip

# import base packages
from tqdm.auto import tqdm
import torch
from representations.prompt import Prompt
from representations.image_noise import ImageNoise



# a function that preforms diffusion steps
def diffusion_steps(prompt: Prompt, imageNoise: ImageNoise, num_steps: int = 250, guidance_scale: float = 7.5) -> ImageNoise:
    # check if the global variables have been initialized and if not then initialize them
    models_pipelines.load_model()

    # add the unconditional to the embeddings
    embeddings = clip.add_unconditional_embeddings(prompt.embeddings)

    # prep the scheduler
    models_pipelines.scheduler.set_timesteps(num_steps)
    latents = imageNoise.latents * models_pipelines.scheduler.init_noise_sigma

    # preform the diffusion steps
    for t in tqdm(models_pipelines.scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = models_pipelines.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = models_pipelines.unet(latent_model_input, t, encoder_hidden_states=embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # compute the previous noisy sample
        latents = models_pipelines.scheduler.step(noise_pred, t, latents).prev_sample
    
    # return the noise image obj
    return ImageNoise(latents=latents)