import numpy as np
import random

side = 8
size = side**2

IMG_1_arr = np.round(np.random.uniform(0.1, 0.9, size), 1)
print(IMG_1_arr)
IMG_2_arr = np.round(np.random.uniform(0.1, 0.9, size), 1)
print(IMG_2_arr)
A_arr = np.random.randint(-5, 5, (size, size))
print(A_arr)

x = C = IMG_1_arr

B = A_arr @ IMG_1_arr
D = A_arr @ IMG_2_arr

def compute_f1(A, x, B, D, lam):
    Ax = A @ x
    return lam*(Ax - D) + (1-lam)*(Ax - B)

def compute_f2(x, C, lam, r):
    return (x - C)**2 + lam**2 - r**2

def compute_F(A, x, B, D, lam, r, C):
    n = len(x)
    F = np.zeros(n + 1)

    Ax = A @ x
    d = B - D

    for i in range(0, n):
        F[i] = Ax[i] - B[i] + lam*d[i]

    F[n] = (np.sum((x - C)**2)) + lam**2 - r**2

    return F

def compute_J(A, B, D, x, C, lam):
    n = len(x)
    J = np.zeros((n + 1, n + 1))

    J[:n, :n] = A # Fills in from the top left corner until the second to last row and column
    J[:n, n] = B - D # Fills in the last column from the top row to the second last row
    J[n, :n] = 2 * (x - C) # Fills in the last row from the first column to the second last column
    J[n, n] = 2 * lam # Fills in the last element in the bottom right corner

    return J

def run_iterations(num: int, img1, img2, A, lam, r, inc):
    n = len(img1)
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
    return x, lam, r, inc, r_initial, inc_initial

def find_best(iterations, img1, img2, A):
    while True:
        r_str = ''.join(random.choice('01') for _ in range(16))
        inc_str = ''.join(random.choice('01') for _ in range(16))
        r = int(r_str, 2) / 100000
        inc = int(inc_str, 2) / 1000000
        lam = 0.001

        x_final, lam_final, r_final, inc_final, r_init, inc_init = run_iterations(iterations, img1, img2, A, lam, r, inc)

        if abs(lam_final - 1) >= 0.09:
            #print(f"Result: lam = {lam_final}")
            continue
        else:
            print(f"Result: Initial r = {r_init}, initial inc = {inc_init} \n"
                  f"x = {x_final}, lam = {lam_final}")
            break

find_best(30, IMG_1_arr, IMG_2_arr, A_arr)

# r_str = ''.join(random.choice('01') for _ in range(16))
# inc_str = ''.join(random.choice('01') for _ in range(16))
# r = int(r_str, 2) / 100000
# inc = int(inc_str, 2) / 1000000

# run_iterations(30, IMG_1_arr, IMG_2_arr, A_arr, 0.001, r, inc)