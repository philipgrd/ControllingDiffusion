
# import stable diffusion
from representations.prompt import Prompt
from representations.image_noise import ImageNoise
from representations.selection import Selection

# import selection methods
import model_methods.selection_methods as selection_methods

# import image processing
import other.image_processing as image_processing



"""
Embeddings i.e. Input -> Image i.e. Human output
->
Input-to-output mapping: Selection i.e. Perturbation and Gradient maps
"""



def run_perturbation_mapping() -> None:
    """Prep"""
    # set prompt
    prompt: Prompt = Prompt("a photograph of an astronaut riding a horse")
    seed: ImageNoise = ImageNoise(seed=17)


    
    """Perturbation mapping"""
    # preform multiplicative perturbation selection
    selection_type = "multiplicative"
    selection: Selection = selection_methods.select(prompt, seed, epsilon=0.15, type=selection_type)
    image_processing.show_selection(selection, selection_type)

    

    """Gradient mapping"""
    # preform gradient selection
    selection_type = "gradient"
    selection = selection_methods.select(prompt, seed, epsilon=0.01, type=selection_type)
    image_processing.show_selection(selection, selection_type)