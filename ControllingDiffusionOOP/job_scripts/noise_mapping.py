
# import stable diffusion
import models.stable_diffusion as stable_diffusion
from representations.prompt import Prompt
from representations.image_noise import ImageNoise

# import the CLIP model (extension)
import models.clip_extension as clip_extension

# import base packages
import matplotlib.pyplot as plt
import pandas as pd



"""
Noise i.e. "Input" -> Image i.e. Human output
->
Input-to-output mapping: Text and image embeddings comparison i.e. Generation difficulty measure
"""



# a function to run the noise impact mapping
# (The name is from the observation that generation difficulty is roughly the same as a "machine common sense" i.e. if it can't be generated it's not known by the model)
def run_common_sense() -> None:
    strings = [
        "a photograph of a astronaut in space",
        "a photograph of an astronaut riding a horse",
        "a photograph of an astronaut riding a elephant",
        "a photograph of an astronaut riding a horse in the desert",
        "a photograph of an astronaut riding a horse in the prism"
    ]
    names = [
        "space",
        "horse",
        "elephant",
        "desert",
        "prism"
    ]
    num_res = 15
    creativity_scores_avg = []
    creativity_scores_std = []

    for index, string in enumerate(strings):
        prompt = Prompt(string)

        string_scores = []
        for i in range(num_res):
            seed = ImageNoise() # Can use "i" as seed

            res = stable_diffusion.diffusion_steps(prompt, seed)
            res.image.save("img_" + names[index] + "_" + str(i) + ".png")

            y = clip_extension.get_similarity_score(prompt, res)

            string_scores.append(y)
        
        # Calculate moving average and standard deviation with a window size of n
        n = num_res
        string_scores_series = pd.Series(string_scores)
        moving_avg = string_scores_series.rolling(window=n).mean().fillna(string_scores_series.expanding().mean()).tolist()
        moving_std = string_scores_series.rolling(window=n).std().fillna(string_scores_series.expanding().std()).tolist()
        creativity_scores_avg.append(moving_avg)
        creativity_scores_std.append(moving_std)

    # Plotting
    fig, ax = plt.subplots()
    
    for idx, (avg_scores, std_scores) in enumerate(zip(creativity_scores_avg, creativity_scores_std)):
        ax.plot(avg_scores, label=strings[idx])
        ax.fill_between(range(len(avg_scores)), 
                        [avg - std for avg, std in zip(avg_scores, std_scores)], 
                        [avg + std for avg, std in zip(avg_scores, std_scores)], 
                        color='b', alpha=0.1)

    ax.set_ylabel('Moving Average of CLIP similarity scores')
    ax.set_xlabel('Number of samples')
    ax.set_title('Moving Average of CLIP similarity scores by Prompt')
    ax.legend()

    fig.tight_layout()
    plt.savefig('img_common_sense_plot.png')
    
    # New figure for standard deviation
    fig_std, ax_std = plt.subplots()
    
    for idx, std_scores in enumerate(creativity_scores_std):
        ax_std.plot(std_scores, label=strings[idx])

    ax_std.set_ylabel('Moving Standard Deviation of CLIP similarity scores')
    ax_std.set_xlabel('Number of samples')
    ax_std.set_title('Moving Standard Deviation of CLIP similarity scores by Prompt')
    ax_std.legend()

    fig_std.tight_layout()
    plt.savefig('img_common_sense_plot_std.png')
    
    print("Average:")
    print(creativity_scores_avg)
    
    print("Standard deviation:")
    print(creativity_scores_std)