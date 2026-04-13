import json
import os.path
import time
from typing import List, Tuple
import numpy as np
from PIL import Image
from json_image_translation import generate_animation_gif
from json_genetic_algorithm import initialize_population, calculate_fitness, selection, crossover, mutate, decode
from json_math_calculations import run_iterations
from multiprocessing import Pool


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

    population: List[List[int]] = initialize_population(IMG_1_arr, IMG_2_arr, A_arr)
    generations: int = 30
    currentGen = 0
    for i in range(0, generations):
        scored_pop: List[Tuple[List[int], float]] = []
        currentGen = i
        '''for individual in population:
            score = calculate_fitness(individual, IMG_1_arr, IMG_2_arr, A_arr)
            scored_pop.append((individual, score))'''
        '''for genomeNum in range(len(population)):
            print(f"checking individual {genomeNum}")
            score, fileRef = calculate_fitness(i, genomeNum)
            print(score, fileRef)
            scored_pop.append((score, fileRef))'''

        filePaths = [f"GENERATION {i}/file{n}.json" for n in range(30)]

        with Pool(8) as p:
            results = p.imap_unordered(calculate_fitness, filePaths)
            for score, fileRef in zip(results, filePaths):
                temp = [score, fileRef]
                print(fileRef)
                scored_pop.append(temp)

        new_population: List[List[int]] = []
        parents, random_list = selection(scored_pop)
        while len(new_population) < 20:
            for j in range(0, len(parents), 2): #Will throw an Index out of Bounds error if the number of parents selected is not even currently
                parent1 = parents[j][0]
                with open(f"GENERATION {i}/{parent1}.json", 'r') as file:
                    data = json.load(file)
                    parent1Genome = data["individual"]

                parent2 = parents[j+1][0]
                with open(f"GENERATION {i}/{parent2}.json", 'r') as file:
                    data = json.load(file)
                    parent2Genome = data["individual"]

                child1 = crossover(parent1Genome, parent2Genome)
                child2 = crossover(parent1Genome, parent2Genome)

                child1 = mutate(child1, 0.01) #0.01 = 1% mutation rate
                child2 = mutate(child2, 0.01)

                new_population.append(child1)
                new_population.append(child2)
#            breakpoint("[poop")
        #write our new population to new folder
        if not os.path.exists(f"GENERATION {i+1}"):
            os.makedirs(f"GENERATION {i+1}")

        for y in range(len(new_population)):
            with open(f"GENERATION {i+1}/file{y}.json", 'w') as file:
                data = {"individual": new_population[y], "img1" : IMG_1_arr.tolist(), "img2" : IMG_2_arr.tolist(), "A_arr" : A_arr.tolist()}
                json.dump(data, file)

        #new_population.extend(random_list)
        #Chuds/random_list are file references
        #must read first
        print("adding chuds")
        for z, chud in enumerate(random_list, start=20):
            print(chud)
            with open(f"GENERATION {i}/{chud[0]}.json", 'r') as oldFile:
                variables = json.load(oldFile)

            with open(f"GENERATION {i+1}/file{z}.json", 'w') as newFile:
                json.dump(variables, newFile)


        population = new_population

    #Rescore the population (as the score gets removed when running selection
    scored_pop = []
    '''for individual in population:
        score = calculate_fitness(individual, IMG_1_arr, IMG_2_arr, A_arr)
        scored_pop.append((individual, score))'''

    '''for genomeNum in range(len(population)):
        print(f"checking individual {genomeNum}")
        score, fileRef = calculate_fitness(currentGen, genomeNum)
        scored_pop.append((score, fileRef))'''
    filePaths = [f"GENERATION {currentGen}/file{n}.json" for n in range(30)]

    with Pool() as p:
        results = p.map(calculate_fitness, filePaths)
        for score, fileRef in zip(results, filePaths):
            temp = [score, fileRef]
            print(fileRef)
            scored_pop.append(temp)
    #Select the TRUE best
    best_individuals, best_random = selection(scored_pop)
    final_winner = best_individuals[0]
    with open(f"GENERATION 30/{final_winner[0]}.json", 'r') as file:
        data = json.load(file)
        final_winner = data["individual"]
    final_r, final_inc = decode(final_winner)




    print(f"Complete. Best Radius: {final_r}, Best Increment: {final_inc}")
    print(f"Best fitness: {scored_pop[0][1]}")
    print("=======")
    print(f"Best individual: \n {final_winner}")
    print("=======")
    print(f"Ending Image: \n {IMG_2_arr}")
    print("=======")
    x, lam, r, inc, r_initial, inc_initial = run_iterations(30, IMG_1_arr, IMG_2_arr, A_arr, 0.001, final_r, final_inc, comments=False)
    print(f"Ending image with best individual: \n {x}")
    print(f"Total Time: {time.time() - start_time}")
    generate_animation_gif(30, IMG_1_arr, IMG_2_arr, A_arr, 0.0001, final_r, final_inc, side)
