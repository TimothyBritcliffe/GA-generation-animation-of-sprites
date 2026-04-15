from typing import List

import numpy as np
from PIL import Image


#Produces a 1xside array of values between 0.1 and 0.9
#https://pillow.readthedocs.io/en/stable/reference/Image.html - kinda neat, and worthwhile to look through
def process_image(image_path:str, side:int) -> np.ndarray:
    img: Image.Image = Image.open(image_path)
    img: Image.Image = img.convert("L") #to grayscale
    img: Image.Image = img.resize((side, side))

    arr: np.ndarray = np.array(img, dtype=np.float32)
    arr: np.ndarray = arr.flatten()

    arr /= 255

    arr: np.ndarray = custom_rounding(arr)

    return arr


#Basic helper function for image processing to round 0, and 1
def custom_rounding(arr) -> np.ndarray:
    arr: np.ndarray = np.asarray(arr)
    result: np.ndarray = arr.copy()

    result[result == 0] = 0.1
    result[result == 1] = 0.9

    return result


#Function that runs through the list of slice (x) histories, then compiles each slices history into a complete vector and creates an image from it returning a list tof said images
def build_full_images_from_histories(slice_histories, side) -> List[Image.Image]:
    max_len: int = max(len(history) for history in slice_histories)
    images: List[Image.Image] = []

    for i in range(max_len):
        current_slices: List[np.ndarray] = []
        for history in slice_histories:
            current_slices.append(history[i] if i < len(history) else history[-1])

        full_vector: np.ndarray = reassemble_vector(current_slices)
        image: Image.Image = vector_to_image(full_vector, side)
        images.append(image)

    return images


#Takes many vectors and smooshes them together
def reassemble_vector(slice_vectors) -> np.ndarray:
    return np.concatenate(slice_vectors)


#Turns a given vector into an image (reverse of image_processing)
def vector_to_image(full_vector, side) -> Image.Image:
    arr: np.ndarray = np.array(full_vector, dtype=np.float32).reshape((side, side))
    arr: np.ndarray = np.clip(arr, 0.0, 1.0)
    arr: np.ndarray = (arr * 255).astype(np.uint8)

    return Image.fromarray(arr, mode="L")


#Saves ALL images together as one .gif file (thus giving the animation)
def generate_animation_gif(images:List[Image], output_path:str="animation.gif", duration:int=120):
    if images:
        images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration, loop=0)


