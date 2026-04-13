import random
from typing import List, Tuple
import numpy as np
from json_math_calculations import run_iterations
import json
import os

#Initializes by default 10 individuals (32 bit strings) - First 16 bits are the radius, second half are the increment
def initialize_population(img1, img2, A_arr, size:int = 30) -> List[List[int]]:
    individuals_=[]
    if not os.path.exists('GENERATION 0'):
        os.mkdir('GENERATION 0')

    for _ in range(size):
        individuals_.append([random.randint(0, 1) for _ in range(32)])

    img1 = img1.tolist()
    img2 = img2.tolist()
    A_arr = A_arr.tolist()

    for i in range(len(individuals_)):
        print(f"writing individual {i}")
        data = {"individual" : individuals_[i], "img1" : img1, "img2" : img2, "A_arr" : A_arr}
        with open(f'GENERATION 0/file{i}.json', 'w') as outfile:
            json.dump(data, outfile)

    return individuals_


#Takes the individual and breaks it down into radius and increment (and converts into decimal from binary)
def decode(ind:List[int]) -> Tuple[float,float]:
    rad: List[int] = ind[0:16]
    inc: List[int] = ind[16:32]

    radius: str = "".join([str(i) for i in rad])
    increment: str = "".join([str(i) for i in inc])

    radius: int = int(radius, 2)
    increment: int = int(increment, 2)

    radius /= 100000
    increment /= 1000000

    return radius, increment


#Executes the math solver and returns a score ($0.0$ to $1.0$) based on how close lam got to 1.0
#individual:List[int], img1:np.ndarray, img2:np.ndarray, A:np.ndarray
def calculate_fitness(generationNum:int, individualNum:int) :
    filePath = f"GENERATION {generationNum}/file{generationNum}.json"
    with open(filePath, 'r') as json_file:
        data = json.load(json_file)
        individual = data["individual"]
        img1 = data["img1"]
        img1 = np.array(img1)

        img2 = data["img2"]
        img2 = np.array(img2)

        A_arr = data["A_arr"]
        A_arr = np.array(A_arr)


    radius, increment = decode(individual)
    x, lam, r, inc, r_initial, inc_initial = run_iterations(30, img1, img2, A_arr, 0.001, radius, increment)
    fileRef = str(filePath[-5])
    if filePath[-6] is int:
        fileRef += str(filePath[-6])
    # Changed scoring system - it currently makes it so that there's a bunch
    # of 0's and only looking for a needle in a haystack

    # if abs(1 - lam) >= 0.05:
    #     score: float = 0.0
    # else:
    #     #score: float = 1.0 - abs(1-lam)
    #     score: float = lam
    # 
    # return score

    # Improved scoring system
    print(individualNum)
    return f"file{individualNum}", 1 / (1 + abs(1 - lam)) # Makes it so that the score of an individual is relative to 1.0

#Executes the math solver and returns a score ($0.0$ to $1.0$) based on how close lam got to 1.0
#individual:List[int], img1:np.ndarray, img2:np.ndarray, A:np.ndarray
def calculate_fitness(filePath) :
    with open(filePath, 'r') as json_file:
        data = json.load(json_file)
        individual = data["individual"]

        print(f"analysing {filePath}")
        img1 = data["img1"]
        img1 = np.array(img1)

        img2 = data["img2"]
        img2 = np.array(img2)

        A_arr = data["A_arr"]
        A_arr = np.array(A_arr)


    radius, increment = decode(individual)
    x, lam, r, inc, r_initial, inc_initial = run_iterations(30, img1, img2, A_arr, 0.001, radius, increment, True)

    # Changed scoring system - it currently makes it so that there's a bunch
    # of 0's and only looking for a needle in a haystack

    # if abs(1 - lam) >= 0.05:
    #     score: float = 0.0
    # else:
    #     #score: float = 1.0 - abs(1-lam)
    #     score: float = lam
    #
    # return score

    # Improved scoring system
    #print(individualNum)
    fileRef = filePath.split('/')
    fileRef = fileRef[1]
    fileRef = fileRef.split('.')[0]
    #print(f"fileReference: {fileRef}\n")
    return fileRef, 1 / (1 + abs(1 - lam)) # Makes it so that the score of an individual is relative to 1.0

#Sorts the overall population by score (in descending order) takes the best 10 from that list and gets a random set of individuals
#Random individual selection count is determined by the target size (if the # of individuals after selecting the top 10 is under 10, then it just selects the remaining, otherwise by default it is 10)
pop_with_scores:List[Tuple[List[int], float]]
def selection(pop_with_scores) -> Tuple[List[List[int]], List[List[int]]]:
    pop_with_scores.sort(key=lambda x: x[1], reverse=True)

    set_of_individuals = [individual[0] for individual in pop_with_scores]

    best_10 = set_of_individuals[:10]
    candidates = set_of_individuals[10:]

    random_10 = []
    target_size = min(10, len(candidates))

    if target_size == 0:
        print("Warning: Population too small to pick random survivors.")
        return best_10, []
    random_10 = random.sample(candidates, k=target_size) if target_size > 0 else []

    return best_10, random_10


#Creates a new (child) individual from slicing each parent at a random point and reassembling
def crossover(p1:List[int], p2:List[int]) -> List[int]:
    random_slice: int = random.randint(0, 31)
    child: List[int] = p1[:random_slice] + p2[random_slice:]

    return child


#Iterates through the individual checking for a probability hit, in the event of a hit, it switches the bits values (0 to 1 and 1 to 0)
#assumes that rate is given in decimal form, i.e 0.01 = 1%
def mutate(ind:List[int], rate:float) -> List[int]:
    modified_ind: List[int] = []

    for bit in ind:
        if random.random() < rate:
            modified_ind.append(1-bit)
        else:
            modified_ind.append(bit)

    return modified_ind

