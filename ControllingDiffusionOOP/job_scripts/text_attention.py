
# import stable diffusion
import models.stable_diffusion as stable_diffusion
from representations.prompt import Prompt
from representations.image_noise import ImageNoise

# import attention map methods
import other.attention_plots as attention_plots

# import base packages
import copy
from typing import List



"""
Prompt i.e. Human input -> Embeddings i.e. Machine input
->
Input-to-output mapping: Attention map
->
Control: Manipulation of embeddings
"""



def run_attention_plots() -> None:
    """Plot attention map"""
    # set prompt
    prompt: Prompt = Prompt("a photograph of an astronaut riding a horse")

    # plot the full attention layer
    attention_plots.plot_attention_layers(prompt, False)

    # plot the attention layer
    attention_plots.plot_attention_layers(prompt, True)



    """Plot attention score"""
    # plot the attention score for every layer
    attention_plots.plot_attention_score(prompt, plot_every_layer = True)

    # plot only the attention a few prompts
    prompt = Prompt("cute dog sitting in a movie theater eating popcorn")
    attention_plots.plot_attention_score(prompt)

    prompt = Prompt("a beautiful victorian raven digital painting")
    attention_plots.plot_attention_score(prompt)

    prompt = Prompt("a portrait of a cyborg in a golden suit, concept art")
    attention_plots.plot_attention_score(prompt)

    prompt = Prompt("a photograph of a cat sitting on a chair")
    attention_plots.plot_attention_score(prompt)



def run_manipulate_embeddings() -> None:
    manipulate_embeddings("a photograph of an astronaut riding a horse", 32, ["astronaut", "photograph"])
    
    manipulate_embeddings("a portrait of a cyborg in a golden suit, concept art", 0, ["suit,"])

def manipulate_embeddings(prompt_str: str, seed_int: int, adjust_word_list: List[str]) -> None:
    """Prep"""
    # set prompt
    prompt: Prompt = Prompt(prompt_str)
    seed: ImageNoise = ImageNoise(seed=seed_int)
    base_img_name: str = "img_" + adjust_word_list[0] + "_" + str(seed_int)



    """Non-manipulated embeddings"""
    # non adjusted prompt (for comparison)
    non_adjusted_img = stable_diffusion.diffusion_steps(prompt, seed)
    non_adjusted_img.image.save(base_img_name + ".png")



    """Manipulate embeddings"""
    for adjust_word in adjust_word_list:
        # set epsilon
        epsilon_list = [-0.6, -0.4, -0.2, 0.2, 0.4, 0.6]

        # (multiplicative) adjust the prompt
        for epsilon in epsilon_list:
            adjusted_prompt = copy.deepcopy(prompt)
            adjusted_prompt.adjust(adjust_word, epsilon=epsilon)
            adjusted_img = stable_diffusion.diffusion_steps(adjusted_prompt, seed)
            adjusted_img.image.save(base_img_name + "_" + adjust_word + "_multiplicative_" + str(epsilon) + ".png")