
# import CLIP modules
import models.clip as clip

# import base packages
from torch import linalg as LA
from typing import Any



# a class that represents a text and/or image prompt
class Prompt:
    # human representations i.e. the human understandable ways a prompt can be represented
    text: str
    # could also be a image representation here i.e. embeddings from an image

    # machine representations i.e. the machine understandable prompt
    embeddings: Any

    # diagnostics representations i.e. the representations used for diagnostics
    attentions: Any
    
    def __init__(self, text: str) -> None:
        # get the embeddings        
        text_embeddings_obj = clip.get_text_embeddings(text)

        # set the attributes
        self.text = text
        self.embeddings = text_embeddings_obj[0]
        self.attentions = text_embeddings_obj[1]
        
    # a function that perturbs the prompt and returns the size of the perturbation
    def adjust(self, word: str, epsilon: float = 0.05) -> float:
        if self.text == None:
            raise Exception("Adjust is not supported for image prompts")
        
        # parse the text prompt
        words = self.text.split()

        # get the index of the word
        index = words.index(word)
        # account for the CLIP start token
        index += 1

        # preform perturbation on the embeddings and save the size of the perturbation
        size_0: float = LA.vector_norm(self.embeddings[0, index]).item()
        self.embeddings[0, index] = (1 + epsilon) * self.embeddings[0, index]
        size_1: float = LA.vector_norm(self.embeddings[0, index]).item()
        
        # return the size of the perturbation
        return (size_1 - size_0)