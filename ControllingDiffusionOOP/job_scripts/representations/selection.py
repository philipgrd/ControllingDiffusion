
# import base packages
from PIL import Image
from typing import List



# a class that represents a image word selection
class Selection:
    # subject i.e. the image that the selections are being made on
    subject: Image.Image

    # selections i.e. the black and white images that represent the selections
    selections: List[Image.Image]
    # labels i.e. the labels for the selections
    labels: List[str]

    def __init__(self, subject: Image.Image, selections: List[Image.Image], prompt: str) -> None:
        # set the attributes
        self.subject = subject
        self.selections = selections
        self.labels = prompt.split()

        if len(self.selections) != len(self.labels):
            raise Exception("Selections and labels must be the same length")
    
    def get_specific_selection(self, label: str) -> Image.Image:
        # get the index of the label
        index = self.labels.index(label)

        # return the selection
        return self.selections[index]