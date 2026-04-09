from typing import List, Tuple
import numpy as np
from genetic_algorithm import initialize_population, calculate_fitness, selection, crossover, mutate, decode
from image_translation import process_image, generate_animation_gif



if __name__ == '__main__':
    side = 8
    size = side ** 2

    #The IMG_1/2_arr variables need to be assigned a 1d array of floating points between 1 and 0 (or in this case 0.9 and 0.1)
    #IMG_1_arr = process_image("img1.png", size=size) (not implemented)
    #IMG_2_arr = process_image("img2.png", size=size) (not implemented)
    IMG_1_arr = np.round(np.random.uniform(0.1, 0.9, size), 1)
    IMG_2_arr = np.round(np.random.uniform(0.1, 0.9, size), 1)
    A_arr = np.random.randint(-5, 5, (size, size))

    generation: List[List[int]] = initialize_population()
    generations: int = 30

    for i in range(0, generations):
        scored_pop: List[Tuple[List[int], float]] = []

        for individual in generation:
            score = calculate_fitness(individual, IMG_1_arr, IMG_2_arr, A_arr)
            scored_pop.append((individual, score))

        best_ind: List[int] = selection(scored_pop)

        new_generation: List[List[int]] = [best_ind]

        while len(new_generation) < 10:
            p1 = selection(scored_pop)
            p2 = selection(scored_pop)

            child = crossover(p1, p2)
            child = mutate(child, 0.01)  #1% chance of mutation

            new_generation.append(child)

        generation = new_generation

    final_winner = selection(scored_pop)
    final_r, final_inc = decode(final_winner)
    print(f"Complete. Best Radius: {final_r}, Best Increment: {final_inc}")
    #generate_animation_gif() (not implemented)