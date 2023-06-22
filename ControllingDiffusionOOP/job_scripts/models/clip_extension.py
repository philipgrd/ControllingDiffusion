
# import models
import models.models_pipelines as models_pipelines

# import base packages
from typing import Any
from representations.prompt import Prompt
from representations.image_noise import ImageNoise



# a function that returns the similarity score between a prompt and an image
def get_similarity_score(prompt: Prompt, image: ImageNoise) -> float:
    # check if clip_tokenizer are None and if so then create them
    models_pipelines.load_model()

    # preprocess the prompt and image
    inputs = models_pipelines.clip_processor(text=[prompt.text], images=image.image, return_tensors="pt", padding=True)

    # move the inputs to the GPU
    inputs = inputs.to("cuda")

    # get the image-text similarity score
    model_output = models_pipelines.clip_model(**inputs)
    logits_per_image = model_output.logits_per_image  # i.e. image-text similarity score

    # this converts the similarity score to probabilities (this works badly for comparing the prompt and the generated image as it is always 1 as it's always close enough)
    #probs = logits_per_image.softmax(dim=1)

    output: float = logits_per_image.item()
    
    return output