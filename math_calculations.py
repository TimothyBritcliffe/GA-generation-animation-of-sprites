import numpy as np
import random

IMG_1_arr = np.array([0.3])
IMG_2_arr = np.array([0.8])

A_arr = np.array([5])

x = IMG_1_arr

C = IMG_1_arr

B = A_arr @ IMG_1_arr
D = A_arr @ IMG_2_arr

def compute_f1(A, x, B, D, lam):
    Ax = A @ x
    return lam*(Ax - D) + (1-lam)*(Ax - B)

def compute_f2(x, C, lam, r):
    return (x - C)**2 + lam**2 - r**2

def compute_F(A, x, B, D, lam, r, C):
    top = compute_f1(A, x, B, D, lam)
    bottom = compute_f2(x, C, lam, r)[0]
    return np.array([
        float(top),
        float(bottom)
    ])

def compute_J(A, B, D, x, C, lam):
    d = B - D
    top_left = A[0]
    top_right = d
    bottom_left = 2*(x - C)[0]
    bottom_right = 2*lam
    return np.array([
        [top_left, top_right],
        [bottom_left, bottom_right]
    ])

def run_iterations(num: int, img1, img2, A, lam, r, inc):
    x = img1
    C = x

    B = A @ img1
    D = A @ img2

    r_initial = r
    inc_initial = inc

    #print(f"Initial r = {r_initial}, initial inc = {inc_initial}")

    for i in range(num):
        F = np.array(compute_F(A, x, B, D, lam, r, C))
        J = compute_J(A, B, D, x, C, lam)

        solution = np.linalg.solve(J, -F)

        if abs(solution[1]) > 1e-10:
            x = x + solution[0]
            lam = lam + solution[1]
            r += inc
            #print(f"After iteration {i + 1}: x = {x}, lam = {lam}")
        else:
            #print(f"Solution converged at iteration {i}")
            break
    return x, lam, r, inc, r_initial, inc_initial

def find_best(iterations, img1, img2, A):
    while True:
        r_str = ''.join(random.choice('01') for _ in range(16))
        inc_str = ''.join(random.choice('01') for _ in range(16))
        r = int(r_str, 2) / 100000
        inc = int(inc_str, 2) / 1000000
        lam = 0.001

        x_final, lam_final, r_final, inc_final, r_init, inc_init = run_iterations(iterations, img1, img2, A, lam, r, inc)

        if abs(lam_final - 1) >= 0.0005:
            continue
        else:
            print(f"Result: Initial r = {r_init}, initial inc = {inc_init} \n"
                  f"x = {x_final}, lam = {lam_final}")
            break

find_best(30, IMG_1_arr, IMG_2_arr, A_arr)
#
# r_str = ''.join(random.choice('01') for _ in range(16))
# inc_str = ''.join(random.choice('01') for _ in range(16))
# r = int(r_str, 2) / 100000
# inc = int(inc_str, 2) / 1000000
#
# run_iterations(30, IMG_1_arr, IMG_2_arr, A_arr, 0.001, r, inc)