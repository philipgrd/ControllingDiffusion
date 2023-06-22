
# import base packages
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Any
from representations.prompt import Prompt



# function that plots the attention layer(s)
def plot_attention_layers(prompt: Prompt, word_size: bool) -> None:
    # get the labels and number of words
    labels = prompt.text.split()
    num_words = len(labels)

    # go through each layer and plot the attention layers
    for i in range(prompt.attentions.shape[1]):
        # clear the plot
        plt.clf()
        # start with a fresh figure
        plt.figure()
        
        # convert the attentions tensor to a matrix
        attention_array = attentions_to_matrix(num_words, prompt.attentions[0, i], word_size)
        
        # create the plot
        fig, ax = plt.subplots()
        img = ax.imshow(attention_array)
        
        # display/save the image
        if word_size:
            # adjust whitespace
            fig.subplots_adjust(bottom=0.25)

            # set labels
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            plt.xticks(rotation=90)

            plt.tight_layout()
            plt.savefig("attention_layer_" + str(i) + ".png")
        else:
            plt.tight_layout()
            plt.savefig("attention_layer_" + str(i) + "_full.png")

# function that plots the attention score(s)
def plot_attention_score(prompt: Prompt, plot_every_layer: bool = False) -> None:
    # get the labels and number of words
    labels = prompt.text.split()
    num_words = len(labels)

    # go through each layer and plot the attention score
    for i in range(prompt.attentions.shape[1]):
        # clear the plot
        plt.clf()
        # start with a fresh figure
        plt.figure()

        # convert the attentions tensor to a matrix
        attention_array = attentions_to_matrix(num_words, prompt.attentions[0, i])

        # get the attention score vector
        attention_score = attention_array.sum(axis=0)

        if plot_every_layer:
            # scale the attention score vector
            attention_score /= attention_score.sum()

            # create the layer plot
            plt.bar(labels, attention_score)
            
            # set labels
            plt.xlabel('Labels')
            plt.ylabel('Values')

            # save the figure
            plt.savefig("attention_score_layer_" + str(i) + ".png")
        
        # add the attention score to the total attention score
        if i == 0:
            total_attention_score = attention_score
        else:
            total_attention_score += attention_score
    
    # clear the plot
    plt.clf()
    
    # scale the total attention score vector
    total_attention_score /= total_attention_score.sum()

    # create the total plot
    plt.bar(labels, total_attention_score)

    # set labels
    plt.xlabel('Labels')
    plt.ylabel('Values')

    # save the figure
    plt.savefig("attention_score_" + prompt.text + ".png")

# function that converts the attentions tensor to a matrix
def attentions_to_matrix(num_words: int, attentions: torch.Tensor, word_size: bool = True) -> Any:
    # convert the tensor to a numpy array
    matrix = attentions.numpy()

    if word_size:
        # remove first column and row
        matrix = matrix[1:, 1:]

        # remove all rows and columns after the num_words
        matrix = matrix[:num_words, :num_words]

    # normalize the matrix so all elements add up to 1
    matrix = matrix / matrix.sum()

    return matrix