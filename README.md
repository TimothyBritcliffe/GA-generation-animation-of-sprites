## Animation of Images Using a Genetic Algorithm

This project was made for our Applied AI course (COMP 3710) at Thompson Rivers University. We were tasked with developing a genetic algorithm (GA) to create art. 

The idea we selected was generating animations between two given image (frames).

## Genetic Algorithm

To create our **initial population***, we use 30 individuals made up of 32-bit strings. The first half of the string (first 16-bits) represents the radius of the hypersphere and the last half represents its increment.

Our **fitness function** breaks down a given individual into slices (or chunks) which allows for faster runtimes via parallel processing. it then uses our `run_iterations()` function to calculate the lambda value of a given slice and forms a list containing all of the lambda values for each slice of an individual. It uses the minimum lambda value in the list of slices to represent the individuals final lambda value, and then using this, it generates a score relative to 1 for the individual.

The **selection algorithm** uses an elitist approach. It takes the 10 best individuals by score and then 10 random individuals from the remaining list. 

For our **breeding** method, we take the 10 best individuals and breed them with one another twice to create a total of 20 children. These children are formed using a randomized crossover. Then to add variation to the new generation, we use a 1% mutation rate on each bit of a newly created individual.

To form **the next generation** we take the 20 children, and then add the 10 randomly selected individuals. This ensures we adhere to a fixed-population size of 30, if we didn't include this, there would be a significant drop in genetic diversity, and ultimately would lead to an overgeneralized population, and a significant likelihood of premature convergence. 

## Math Calculations

Our animation process depends on a Homotopy-based hypersphere solver. We view the transition from `IMG_1_arr` to `IMG_2_arr` as a continuous path, defined by the homotopy equation:

$$H(x, \lambda) = \lambda(Ax - D) + (1 - \lambda)(Ax - B)$$

Where $\lambda$ transitions from 0 (start) to 1 (end).

To compute the following frames, we solve the hypersphere equation iteratively using the Newton-Raphson method. Due to the complexity and computational expense of matrix inversion, we treat the Jacobian as a linear system:

$$J\Delta x = -F$$

Solving this linear system instead of performing matrix division allows us to calculate the change in pixels and $\lambda$ efficiently across 30 iterations, which significantly reduces the total runtime of our system.

## How To Use

1. Clone/Download our repository
2. Install any dependencies needed (Pillow, NumPy, joblib)
3. Move two images you would like an animation between into the `GA-generation-animation-of-sprites` directory
4. In `main.py` (lines 16/17) modify the `IMG_1_arr` and `IMG_2_arr` `image_path` arguments to match the file names of your images
5. Run `main.py` and wait for your results
6. View the resulting animation named `translation.gif`