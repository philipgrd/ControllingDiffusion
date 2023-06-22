
# import the stable diffusion model
import models.stable_diffusion as stable_diffusion

# import base libraries
import numpy as np
import torch
from typing import List
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from representations.prompt import Prompt
from representations.image_noise import ImageNoise



"""
Embeddings i.e. Input + Similar inputs i.e. "Lighthouses"
->
Input-to-output mapping: Selection i.e. Perturbation and Gradient maps
"""



def joystick_prompt_run() -> None:
    main_string = "a picture of a space explorer galloping on a horse"
    strings = [
        "a photograph of an astronaut riding a horse",
        "an image showing an astronaut on horseback",
        "a snapshot of a spaceman riding a steed",
        "a visual capture of an astronaut astride a horse",
        "a snapshot depicting a space explorer riding a majestic steed",
        "a visual representation of an astronaut astride a powerful horse",
        "a picture of a cosmonaut on horseback",
    ]

    # Prepare the prompt and seed
    original_prompt: Prompt = Prompt(main_string)
    moved_prompt: Prompt = Prompt(main_string)
    seed: ImageNoise = ImageNoise(seed=0)
    
    moved_prompt.embeddings = move_towards(original_prompt.embeddings, Prompt(strings[3]).embeddings, 0.6)

    # Generate and plot the PCA and associated images
    generate_and_plot_pca(original_prompt.embeddings, strings, "original")
    original_img = stable_diffusion.diffusion_steps(original_prompt, seed)
    original_img.image.save("original.png")

    generate_and_plot_pca(moved_prompt.embeddings, strings, "moved")
    moved_img = stable_diffusion.diffusion_steps(moved_prompt, seed)
    moved_img.image.save("moved.png")

# a function to move from start towards towards by amount
def move_towards(start: torch.Tensor, towards: torch.Tensor, amount: float) -> torch.Tensor:
    # Check that amount is in the correct range
    if not (0 <= amount <= 1):
        raise ValueError("amount should be between 0 and 1")
    
    # Linear interpolation between start and towards
    result = (1 - amount) * start + amount * towards
    
    return result

# a function to generate and plot the PCA of a list of strings
def generate_and_plot_pca(point: torch.Tensor, list_of_str: List[str], name: str) -> None:
    # Initialize new figure
    plt.figure(figsize=(8,8))

    # Initialize PCA
    pca = PCA(n_components=2)

    # Generate embeddings using the Prompt class and append them to the embeddings list
    embeddings_list = []
    for str_ in list_of_str:
        prompt_obj = Prompt(str_)

        # Remove the embedding from GPU and flatten it
        embedding = prompt_obj.embeddings.detach().cpu().flatten().numpy()

        # Append the embedding to the list
        embeddings_list.append(embedding)
    
    # Stack all embeddings into a numpy array
    embeddings_ndarray = np.vstack(embeddings_list)
    
    # Apply PCA to the embeddings
    pca.fit(embeddings_ndarray)
    transformed_embeddings = pca.transform(embeddings_ndarray)

    # Prepare color map
    cmap = plt.cm.get_cmap('viridis', len(list_of_str))

    # Plot the embeddings, using different colors for each point
    for i in range(len(list_of_str)):
        plt.scatter(transformed_embeddings[i, 0], transformed_embeddings[i, 1], color=cmap(i), label=list_of_str[i], s=36*3.0) 

    # Remove the embedding from GPU and flatten it
    point_embedding = point.detach().cpu().flatten().numpy()

    # Transform the point using the fitted PCA
    transformed_point = pca.transform(point_embedding.reshape(1, -1))

    # Plot the point with a different color
    plt.scatter(transformed_point[:, 0], transformed_point[:, 1], color='red', label='Input/Joystick point', s=36*4.5)

    # Increase the size of the labels
    plt.legend(fontsize=14)
    
    # Set plot details
    plt.title('PCA plot of the relevant embedding space')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')

    # Set the legend location to 'best' to avoid overlapping
    plt.legend(loc='best')

    # Save the figure
    plt.savefig("pca_plot_" + name + ".png")
