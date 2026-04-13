import numpy as np
from PIL import Image
import json_math_calculations as M
import cv2
import os

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

def generate_animation_gif(num, img1, img2, A, lam, r, inc, dimensions):
    print("Generating animation...")
    if not os.path.exists(f"FINAL_GENERATION"):
        os.mkdir("FINAL_GENERATION")
    n = len(img1)
    x = img1
    C = x

    B = A @ img1
    D = A @ img2

    r_initial = r
    inc_initial = inc

    # print(f"Initial r = {r_initial}, initial inc = {inc_initial}")

    for i in range(num):
        F = np.array(M.compute_F(A, x, B, D, lam, r, C))
        J = M.compute_J(A, B, D, x, C, lam)

        try:
            solution = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            continue

        if abs(solution[n]) > 1e-10:
            x = x + solution[:n]
            lam = lam + solution[n]
            r += inc
            #print(f"After iteration {i + 1}: x = {x}, lam = {lam}")
        else:
            #print(f"Converged at iteration {i}")
            break


        img = np.array(x)
        img = img.reshape(dimensions, dimensions)
        final_img = (img * 255).astype(np.uint8)
        cv2.imwrite(f"FINAL_GENERATION/IMG{i+1}.png", final_img)
    frames = [Image.open(f"FINAL_GENERATION/IMG{i + 1}.png") for i in range(30)]
    frames[0].save(
        "animation.gif",
        save_all=True,
        append_images=frames[1:],
        duration=150,
        loop=0,
    )
    print("animation generated succefully!")


