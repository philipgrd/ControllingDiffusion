
# import image processing packages
from PIL import Image, ImageDraw, ImageFont
import colorsys

# import base packages
import numpy as np
import copy
from typing import List
from representations.selection import Selection



# function that takes a primary image and a list images representing the selection and combines them into one image
def show_selection(selection: Selection, name: str) -> None:
    """Max selection"""
    # create the selection image
    max_selection = copy.deepcopy(selection)
    selection_image = create_selection_image(max_selection.selections)

    # overlay the selection images on top of the primary image
    selection_img = overlay_image(max_selection.subject, [selection_image])

    # add colored prompt
    selection_img = add_colored_prompt(max_selection.labels, selection_img)

    # display/save the selection image
    selection_img.save("full_selection_" + name + ".png")
    
    # save selections
    for i, the_selection in enumerate(max_selection.selections):
        the_selection.save("selection_" + name + "_" + max_selection.labels[i] + ".png")

    

    """Blend selection"""
    # create the selection image
    blend_selection = copy.deepcopy(selection)
    selection_images = colorize_images(blend_selection.selections)

    # overlay the selection images on top of the primary image
    selection_img = overlay_image(blend_selection.subject, selection_images)

    # add colored prompt
    selection_img = add_colored_prompt(blend_selection.labels, selection_img)

    # display/save the selection image
    selection_img.save("full_selection_" + name + "_blend.png")

# add color to selection images
def colorize_images(images: List[Image.Image]) -> List[Image.Image]:
    # convert all images to RGBA
    images = [img.convert("RGBA") for img in images]
    
    colored_images = []
    for i, image in enumerate(images):
        # get the color for the current image
        color = color_from_index(i, len(images))

        # convert the image to a NumPy array
        data = np.array(image)

        # create a new NumPy array filled with the original data
        colored_data = data.copy()

        # set color channels to the specific color
        colored_data[..., :3] = color

        # calculate the normalized white intensity
        white_intensity = np.mean(data[..., :3], axis=-1, keepdims=True) / 255

        # squeeze white_intensity array to remove the singleton dimension
        white_intensity = np.squeeze(white_intensity)

        # adjust the alpha channel based on the white intensity
        colored_data[..., 3] = (white_intensity * 255).astype(np.uint8)

        # create a new image from the colored data
        colored_image = Image.fromarray(colored_data, "RGBA")

        colored_images.append(colored_image)

    return colored_images

# function that takes a list of PIL images and returns a new image where each pixel is the color of the image with the highest RGB sum
def create_selection_image(images: List[Image.Image]) -> Image.Image:
    # convert all images to RGB
    images = [img.convert("RGBA") for img in images]
    
    # get the size of the first image (assuming all images are the same size)
    width, height = images[0].size

    # convert PIL images to numpy arrays for faster computation
    np_images = [np.array(img) for img in images]

    # calculate RGB sum for each pixel for each image
    rgb_sums = np.array([np_image.sum(axis=2) for np_image in np_images])

    # find the image index with the maximum RGB sum for each pixel
    max_index = np.argmax(rgb_sums, axis=0)

    # create a new image with alpha channel
    new_image = Image.new("RGBA", (width, height))

    # define alpha value
    alpha = 200  # Adjust as needed 0 is fully transparent, 255 is fully opaque.

    # for each pixel, assign the color given by the function `color_from_index`, with the defined alpha value
    for x in range(width):
        for y in range(height):
            r, g, b = color_from_index(max_index[y, x], len(images))
            new_image.putpixel((x, y), (r, g, b, alpha))

    return new_image

# overlay a list of partly transparent PIL images on top of a base image
def overlay_image(base_image: Image.Image, overlay_images: List[Image.Image]) -> Image.Image:
    # convert base image to RGBA
    base_image = base_image.convert("RGBA")
    base_image = adjust_alpha(base_image, 20)
    
    for overlay_image in overlay_images:
        # convert overlay image to RGBA
        overlay_image = overlay_image.convert("RGBA")
        
        # overlay
        base_image = Image.alpha_composite(base_image, overlay_image)

    return base_image

def adjust_alpha(image: Image.Image, alpha: float) -> Image.Image:
    """Adjusts the alpha of the given PIL Image."""
    # adjust alpha to be between 0 and 1
    alpha = alpha / 255

    # ensure image has an alpha channel
    image = image.convert('RGBA')

    # use the split() function to split the image into its respective bands
    r, g, b, a = image.split()

    # use the point() function to modify the alpha channel
    a = a.point(lambda p: p * alpha)

    # merge the channels back
    image = Image.merge('RGBA', (r, g, b, a))

    return image

def add_colored_prompt(labels: List[str], image: Image.Image) -> Image.Image:
    # create an ImageDraw object
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # calculate total text height
    text_height = max([draw.textsize(label, font=font)[1] for label in labels])
    padding = 10  # Padding around the text

    # create a new image that is larger than the original and fill it with white color
    new_image = Image.new("RGB", (image.width, image.height + text_height + 2 * padding), color=(255, 255, 255))

    # paste the original image onto the new image
    new_image.paste(image, (0, 0))

    # create a new ImageDraw object for the new image
    draw = ImageDraw.Draw(new_image)

    # start the x-coordinate for the text
    x = 10

    for i, label in enumerate(labels):
        # measure the size of the text to be drawn
        text_size = draw.textsize(label, font=font)
        
        # draw the word, using a color from the colors list
        draw.text((x, image.height + padding), label, fill=color_from_index(i, len(labels)), font=font)

        # add some space for the next word
        x += text_size[0] + 10  # Add extra 10 for spacing

    return new_image

def color_from_index(index: int, num_colors: int) -> tuple[int, int, int]:
    (h, s, v) = (index / num_colors, 0.5, 0.9)
    
    (r, g, b) = colorsys.hsv_to_rgb(h, s, v)

    return (int(255 * r), int(255 * g), int(255 * b))