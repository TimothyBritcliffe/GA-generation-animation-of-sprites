import random
from typing import List, Tuple
import numpy as np
from math_calculations import run_iterations


#Initializes by default 10 individuals (32 bit strings) - First 16 bits are the radius, second half are the increment
def initialize_population(size:int = 10) -> List[List[int]]:
    individuals_=[]

    for _ in range(size):
        individuals_.append([random.randint(0, 1) for _ in range(32)])

    return individuals_

#Takes the individual and breaks it down into radius and increment (and converts into decimal from binary)
def decode(ind:List[int]) -> Tuple[float,float]:
    rad: List[int] = ind[0:16]
    inc: List[int] = ind[16:32]

    radius: str = "".join([str(i) for i in rad])
    increment: str = "".join([str(i) for i in inc])

    radius: int = int(radius, 2)
    increment: int = int(increment, 2)

    radius /= 1000000
    increment /= 100000

    return radius, increment


#Executes the math solver and returns a score ($0.0$ to $1.0$) based on how close lam got to 1.0
def calculate_fitness(individual:List[int], img1:np.ndarray, img2:np.ndarray, A:np.ndarray) -> float:
    radius, increment = decode(individual)
    x, lam, r, inc, r_initial, inc_initial = run_iterations(30, img1, img2, A, 0.001, radius, increment)

    if lam > 1.1:
        score: float = 0.0
    else:
        score: float = 1.0 - abs(1-lam)

    return score


#Checks the population and returns the winning individual that gets to be a parent
#Winning parent selected by Tournament - if elitism preferred we can just remove the sample so it finds the best overall
def selection(pop_with_scores:List[Tuple[List[int], float]]) -> List[int]:
    sample = random.sample(pop_with_scores, 3)
    best_ind: List[int] = sample[0][0]
    highest_score: float = sample[0][1]

    for ind in sample:
        selected_ind: List[int] = ind[0]
        current_score: float = ind[1]
        if current_score > highest_score:
            highest_score = current_score
            best_ind = selected_ind

    return best_ind


#Creates a new (child) individual from slicing each parent at a random point and reassembling
def crossover(p1:List[int], p2:List[int]) -> List[int]:
    random_slice: int = random.randint(0, 31)
    child: List[int] = p1[:random_slice] + p2[random_slice:]

    return child


#Iterates through the individual checking for a probability hit, in the event of a hit, it switches the bits values (0 to 1 and 1 to 0)
#assumes that rate is given in decimal form, i.e 0.01 = 1%
def mutate(ind:List[int], rate:float) -> List[int]:
    modified_ind: List[int] = []
    rate *= 100

    for i in ind:
        if random.randint(1, 100) in range(1, int(rate)+1):
            if i == 0:
                modified_ind.append(1)
            else:
                modified_ind.append(0)
        else:
            modified_ind.append(i)

    return modified_ind