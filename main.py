import time
from typing import List, Tuple
import numpy as np
from genetic_algorithm import initialize_population, calculate_fitness, selection, crossover, mutate, decode



if __name__ == '__main__':
    start_time = time.time()
    side = 4
    size = side ** 2

    #The IMG_1/2_arr variables need to be assigned a 1d array of floating points between 1 and 0 (or in this case 0.9 and 0.1)
    #IMG_1_arr = process_image("img1.png", side=side) (not implemented)
    #IMG_2_arr = process_image("img2.png", side=side) (not implemented)
    IMG_1_arr = np.round(np.random.uniform(0.1, 0.9, size), 1)
    IMG_2_arr = np.round(np.random.uniform(0.1, 0.9, size), 1)
    A_arr = np.random.randint(-5, 5, (size, size))

    population: List[List[int]] = initialize_population()
    generations: int = 30

    for i in range(0, generations):
        scored_pop: List[Tuple[List[int], float]] = []

        for individual in population:
            score = calculate_fitness(individual, IMG_1_arr, IMG_2_arr, A_arr)
            scored_pop.append((individual, score))

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

                if len(new_population) >= 20:
                    break

        new_population.extend(random_list)
        population = new_population

    #Rescore the population (as the score gets removed when running selection
    scored_pop = []
    for individual in population:
        score = calculate_fitness(individual, IMG_1_arr, IMG_2_arr, A_arr)
        scored_pop.append((individual, score))

    #Select the TRUE best
    best_individuals, best_random = selection(scored_pop)
    final_winner = best_individuals[0]
    final_r, final_inc = decode(final_winner)


    print(f"Complete. Best Radius: {final_r}, Best Increment: {final_inc}")
    print(f"Best fitness: {scored_pop[0][1]}")
    print("=======")
    print(final_winner)
    print("=======")
    print(IMG_1_arr)