
# import models
import models.models_pipelines as models_pipelines

# import base packages
import torch
from typing import Any, Tuple, Optional



# function that adds the unconditional text embeddings for classifier-free guidance, which are just the embeddings for the padding token (empty text)
def add_unconditional_embeddings(text_embeddings: Any) -> Any:
    # check if CLIP model is not None and if then create it
    models_pipelines.load_model()

    # get the embeddings for the empty text
    empty_text_embeddings = get_text_embeddings("")[0]

    # move the text embeddings to the GPU
    text_embeddings = text_embeddings.to("cuda")
    
    # create and return the text embeddings
    text_embeddings = torch.cat([empty_text_embeddings, text_embeddings])

    return text_embeddings

# function encodes the tokenized text
def get_text_embeddings(prompt: str, attentions_index: Optional[int] = None) -> Tuple[torch.FloatTensor, Any]:
    # check if CLIP model is not None and if then create it
    models_pipelines.load_model()

    tokenized_text = models_pipelines.clip_tokenizer(prompt, padding="max_length", max_length=models_pipelines.clip_tokenizer.model_max_length, truncation=True, return_tensors="pt")

    # move the tokenized text to the GPU
    input_ids = tokenized_text.input_ids.to("cuda")

    # get the embeddings
    with torch.no_grad():
        text_embeddings_obj = models_pipelines.clip_text_encoder(input_ids, output_attentions=True)

    # concatenate attentions across all layers along the head dimension (dimension 1)
    if attentions_index == None:
        all_attentions = torch.cat(text_embeddings_obj.attentions, dim=1)
    else:
        all_attentions = text_embeddings_obj.attentions[attentions_index]
    
    # move the attentions to the CPU
    all_attentions_cpu = all_attentions.cpu()
        
    # return the embeddings and the attentions
    return (text_embeddings_obj[0], all_attentions_cpu)