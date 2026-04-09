import numpy as np
from PIL import Image


#Produces a 1x64 array of values between 0.1 and 0.9
#Commented code is incase we wanted to separate the image array by rows for multithreading processing
#https://pillow.readthedocs.io/en/stable/reference/Image.html - kinda neat, and worthwhile to look through
def process_image(image_path: str, side: int) -> np.ndarray:
    img: Image.Image = Image.open(image_path)
    img.convert("L") #to grayscale
    img.resize((side, side))

    arr: np.ndarray = np.array(img)
    arr.flatten() #may not need if we are just going to separate this...
    #separate by rows:
    #new_arr = arr.tolist()
    arr /= 255
    #new_arr /= 255
    arr = custom_rounding(arr)
    #new_arr = custom_rounding(new_arr)

    return arr
    #return new_arr


def custom_rounding(arr):
    arr = np.asarray(arr)
    result = arr.copy()
    result[result == 0] = 0.1
    result[result == 1] = 0.9

    return result

def generate_animation_gif():
    pass


