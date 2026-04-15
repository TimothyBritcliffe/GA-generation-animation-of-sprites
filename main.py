import time
from typing import List, Tuple
import numpy as np
from genetic_algorithm import initialize_population, calculate_fitness, selection, crossover, mutate, decode
from math_calculations import run_iterations
from joblib import Parallel, delayed
from image_translation import process_image, build_full_images_from_histories, generate_animation_gif, vector_to_image


if __name__ == '__main__':
    start_time = time.time()
    side = 64
    size = side ** 2

    #The IMG_1/2_arr variables need to be assigned a 1d array of floating points between 1 and 0 (or in this case 0.9 and 0.1)
    IMG_1_arr = process_image("GAtest1.png", side=side)
    IMG_2_arr = process_image("GAtest2.png", side=side)
    # IMG_1_arr = np.round(np.random.uniform(0.1, 0.9, size), 1)
    # IMG_2_arr = np.round(np.random.uniform(0.1, 0.9, size), 1)
    #A_arr = np.random.randint(-5, 5, (size, size))
    # OPTIMIZATION TEST
    A_arr = np.random.randint(-5, 5, size=(16, 16))
    #
    population: List[List[int]] = initialize_population()
    generations: int = 30

    for i in range(0, generations):
        scored_pop: List[Tuple[List[int], float]] = []

        print(f"Analyzing Generation: {i}")
        scored_pop = Parallel(n_jobs=-1)(delayed(calculate_fitness)(individual, IMG_1_arr, IMG_2_arr, A_arr) for individual in population)
        print(f"Finished Analyzing Generation: {i}")

        new_population: List[List[int]] = []
        parents, random_list = selection(scored_pop)
        while len(new_population) < 20:
            for j in range(0, len(parents), 2): #Will throw an Index out of Bounds error if the number of parents selected is not even currently
                parent1 = parents[j]
                parent2 = parents[j+1]

                child1 = crossover(parent1, parent2)
                child2 = crossover(parent1, parent2)

                child1 = mutate(child1, 0.01) #0.01 = 1% mutation rate
                child2 = mutate(child2, 0.01)

                new_population.append(child1)
                new_population.append(child2)

        new_population.extend(random_list)
        population = new_population

        # Rescore the population (as the score gets removed when running selection
    scored_pop = []
    scored_pop = Parallel(n_jobs=-1)(delayed(calculate_fitness)(individual, IMG_1_arr, IMG_2_arr, A_arr) for individual in population)

    #Select the TRUE best
    best_individuals, best_random = selection(scored_pop)
    final_winner = best_individuals[0]
    final_r, final_inc = decode(final_winner)


    #Handles the new slicing functionality (specifically since the A_arr is now 16x16
    slice_histories = []
    chunk_size = A_arr.shape[0]

    for start in range(0, len(IMG_1_arr), chunk_size):
        slice_1 = IMG_1_arr[start:start + chunk_size]
        slice_2 = IMG_2_arr[start:start + chunk_size]

        lam, x_final, x_history = run_iterations(30, slice_1, slice_2, A_arr, 0.001, final_r, final_inc, comments=False, capture_history=True)

        slice_histories.append(x_history)

    #Creates the full vector from the slice history, then turns that into an image (does it for all slice histories, therefore it is a list of "images")
    list_of_images = build_full_images_from_histories(slice_histories, side)

    #The starting and ending images must be included in the animation so we need to convert these from vectors back to images
    start_image = vector_to_image(IMG_1_arr, side)
    end_image = vector_to_image(IMG_2_arr, side)

    #MASH EM
    final_list_of_images = [start_image] + list_of_images + [end_image]

    #Creates the gif based on the list of images and a desired path name for le gif
    generate_animation_gif(final_list_of_images, output_path="translation.gif")

    print(f"Complete. Best Radius: {final_r}, Best Increment: {final_inc}")
    print(f"Best fitness: {scored_pop[0][1]}")
    print("=======")
    print(f"Best individual: \n {final_winner}")
    print("=======")
    print(f"Ending Image: \n {IMG_2_arr}")
    print("=======")
    #x, lam, r, inc, r_initial, inc_initial = run_iterations(30, IMG_1_arr, IMG_2_arr, A_arr, 0.001, final_r, final_inc, comments=False)
    #print(f"Ending image with best individual: \n {x}")
    print(f"Total Time: {time.time() - start_time}")