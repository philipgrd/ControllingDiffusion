
# import stable diffusion
import models.stable_diffusion as stable_diffusion

# import base packages
from PIL import Image, ImageChops
import copy
import numpy as np
from typing import Any, List
from representations.prompt import Prompt
from representations.image_noise import ImageNoise
from representations.selection import Selection



# a function that preforms perturbation to select image elements
def select(prompt: Prompt, seed: ImageNoise, epsilon: float = 0.1, type: str = "multiplicative") -> Selection:
    # get number of words in prompt
    words = prompt.text.split()

    # get primary image i.e. the subject of the selection
    subject: ImageNoise = stable_diffusion.diffusion_steps(prompt, seed, num_steps=20)

    # perturbation results
    selections: List[Image.Image] = []

    # preform perturbation on each word
    for word in words:
        # get specific selection for word
        if type == "gradient":
            a_selection = finite_difference(prompt, seed, subject.image, word, epsilon)
        elif type == "multiplicative":
            a_selection = perturbation(prompt, seed, subject.image, word, epsilon)
        else:
            raise Exception("Invalid type: " + str(type))
    
        # save perturbation result
        selections.append(a_selection)
    
    # construct the selection
    selection: Selection = Selection(subject.image, selections, prompt.text)

    # return the selection
    return selection

def finite_difference(prompt: Prompt, seed: ImageNoise, subject_img: Image.Image, word: str, epsilon: float) -> Image.Image:
    # preform perturbation on the copy
    perturbation_prompt: Prompt = copy.deepcopy(prompt)
    epsilon_size = perturbation_prompt.adjust(word, epsilon)

    # get the perturbation image
    perturbation_output: ImageNoise = stable_diffusion.diffusion_steps(perturbation_prompt, seed, num_steps=20)

    # convert subject and perturbation to array
    perturbation_tensor: Any = np.array(perturbation_output.image)
    subject_tensor: Any = np.array(subject_img)

    # preform finite difference approximation
    gradient = (perturbation_tensor - subject_tensor) / epsilon_size

    # calculate the magnitude of the gradient vector at each pixel
    gradient_magnitude = np.linalg.norm(gradient, axis=2, keepdims=True)

    # remove the extra dimension
    gradient_magnitude = np.squeeze(gradient_magnitude)
    
    # normalize gradient to be between 0 and 255
    gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) * (255 / (gradient_magnitude.max() - gradient_magnitude.min()))

    # convert gradient to a image
    gradient_pil: Image.Image = Image.fromarray(gradient_magnitude.astype(np.uint8))

    # return the gradient
    return gradient_pil

def perturbation(prompt: Prompt, seed: ImageNoise, subject_img: Image.Image, word: str, epsilon: float) -> Image.Image:
    # preform perturbation on the copy
    perturbation_prompt: Prompt = copy.deepcopy(prompt)
    perturbation_prompt.adjust(word, epsilon)

    # get the perturbation image
    perturbation_output: ImageNoise = stable_diffusion.diffusion_steps(perturbation_prompt, seed, num_steps=20)

    # get the difference between the subject and the perturbation
    perturbation_diff: Image.Image = get_difference_in_img(subject_img, perturbation_output.image)

    # return the perturbation
    return perturbation_diff

# function that takes two PIL images and returns a new PIL image that represents the difference between the two images
def get_difference_in_img(image1: Image.Image, image2: Image.Image) -> Image.Image:
    # get difference image
    difference_image = ImageChops.difference(image1, image2)

    # convert to black and white
    difference_image = difference_image.convert("L")

    return difference_image