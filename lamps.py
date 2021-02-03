# Function to evaluate the "lamp" problem, where we try to place round-shaped lamps to cover a square space


import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import copy
from random import Random

from matplotlib.patches import Circle, Rectangle


def evaluateLamps(lamps, radius, square, visualize=False):
    globalFitness = 0.0
    individualFitness = [0] * len(lamps)

    # this is a very rough discretization of the space
    discretization = 80  # lower discretization here to speed up computation, increase for increased precision
    discretizationStepX = square[0] / discretization
    discretizationStepY = square[1] / discretization
    totalArea = square[0] * discretization * square[1] * discretization

    # compute coverage of the square, going step by step
    coverage = 0.0
    overlap = 0.0

    for x in np.arange(0.0, square[0], discretizationStepX):
        for y in np.arange(0.0, square[1], discretizationStepY):
            coveredByLamps = 0
            for l in range(0, len(lamps)):

                lamp = lamps[l]

                # if the distance between the point and the center of any lamp is less than
                # the radius of the lamps, then the point is lightened up!
                distance = math.sqrt(math.pow(lamp[0] - x, 2) + math.pow(lamp[1] - y, 2))
                if distance <= radius:
                    coveredByLamps += 1
                    individualFitness[l] += 1

            # now, if the point is covered by at least one lamp, the global fitness increases
            if coveredByLamps > 0:
                coverage += square[0] * square[1]
            # but if it is covered by two or more, there's a 'waste' of light here, an overlap
            if coveredByLamps > 1:
                overlap += 1

    # the global fitness can be computed in different ways
    # globalFitness = coverage / totalArea  # just as total coverage by all lamps
    globalFitness = (coverage - overlap) / totalArea  # or maybe you'd like to also minimize overlap!

    # if the flag "visualize" is true, let's plot the situation
    if visualize:

        figure = plt.figure()
        ax = figure.add_subplot(111, aspect='equal')

        # matplotlib needs a list of "patches", polygons that it is going to render
        for l in lamps:
            ax.add_patch(Circle((l[0], l[1]), radius=radius, color='b', alpha=0.4))
        ax.add_patch(Rectangle((0, 0), square[0], square[1], color='w', alpha=0.4))

        ax.set_title("Lamp coverage of the arena (fitness %.2f)" % globalFitness)
        plt.xlim(0, square[0])
        plt.ylim(0, square[1])
        plt.show()
        plt.close(figure)

    return globalFitness


def mutation(generator, mutation_probability_coord, mutation_probability_size, lamps1, sigma_x, sigma_y, square):
    lamps = copy.deepcopy(lamps1)
    for i in range(0, len(lamps)):
        change_probability_coord = generator.uniform(0, 1)
        if change_probability_coord <= mutation_probability_coord:
            # gauss mutation
            alpha_x = generator.gauss(0, sigma_x)
            alpha_y = generator.gauss(0, sigma_y)
            lamps[i][0] += alpha_x
            lamps[i][1] += alpha_y

    # we can also add / remove a lamp
    change_probability_size = generator.uniform(0, 1)
    if change_probability_size <= mutation_probability_size:
        tail = generator.uniform(0, 1)
        if tail <= 0.5:
            x_max = square[0]
            y_max = square[1]
            lamp_x = generator.uniform(0, x_max)
            lamp_y = generator.uniform(0, y_max)
            lamps.append([lamp_x, lamp_y])
        elif tail > 0.5 and len(lamps) != 1:
            position = generator.randint(0, len(lamps) - 1)
            del lamps[position]

    return lamps


def reproduction(generator, species1, offspring_size, crossover_probability, tournament_size):

    # we first copy the best specie so we don't lose it
    species = copy.deepcopy(species1)
    species.append(species[0].copy())
    species_nb = len(species)

    for i in range(offspring_size):
        # we make sure to only keep the previous generation as parents
        parent1 = tournament(generator, species[0:species_nb], tournament_size)

        # crossover
        test = generator.uniform(0, 1)
        if test <= crossover_probability:
            parent2 = tournament(generator, species[0:species_nb], tournament_size)

            # croisement barycentrique
            child = [[], 0]
            parent1_size = len(parent1[0])
            parent2_size = len(parent2[0])
            long_parent_size = max(parent1_size, parent2_size)
            short_parent_size = min(parent1_size, parent2_size)
            # croisement barycentrique sur les premiers élements
            for i in range(short_parent_size):
                alpha = generator.uniform(0, 1)
                child_gene_x = alpha * parent1[0][i][0] + (1 - alpha) * parent2[0][i][0]
                child_gene_y = alpha * parent1[0][i][1] + (1 - alpha) * parent2[0][i][1]
                child[0].append([child_gene_x, child_gene_y])

            # on ajoute au hasard la moitié des lampes du parent le plus long
            if long_parent_size > short_parent_size:
                nb_of_lamps = (long_parent_size - short_parent_size) // 2

                # finding which parent is longer
                long_parent = parent1
                if long_parent_size != len(long_parent[0]):
                    long_parent = parent2

                # adding random lamps to the child from the longest parent
                lamps_to_add = long_parent[0][short_parent_size:long_parent_size].copy()
                for j in range(nb_of_lamps):
                    position = generator.randint(0, len(lamps_to_add) - 1)
                    lamp = lamps_to_add.pop(position)
                    child[0].append(lamp)

        else:
            child = parent1.copy()

        species.append(child)

    return species


def generate_species(square, nb_of_species, generator, nb_of_lamps, radius):
    species = []
    for specie in range(nb_of_species):

        # generating the first generation of lamps
        lamps = []
        for lamp in range(nb_of_lamps):
            x_max = square[0]
            y_max = square[1]
            lamp_x = generator.uniform(0, x_max)
            lamp_y = generator.uniform(0, y_max)
            lamps.append([lamp_x, lamp_y])

        fitness = evaluateLamps(lamps, radius, square)
        species.append([lamps, fitness])
    species = sorted(species, key=lambda x: x[1], reverse=True)
    return species


def tournament(generator, species, tournament_size):
    contestants = [species[generator.randint(0, len(species) - 1)] for i in range(tournament_size)]
    contestants = sorted(contestants, key=lambda x: x[1], reverse=True)
    return contestants[0]


def do_stats(stats, species, generation, verb=True):
    best_fitness = species[0][1]
    worst_fitness = species[-1][1]
    best_length = len(species[0][0])

    avg_fitness = 0
    for i in range(len(species)):
        avg_fitness += species[i][1]
    avg_fitness = avg_fitness / len(species)

    generation_stats = {'best_fitness': best_fitness,
                        'worst_fitness': worst_fitness,
                        'avg_fitness': avg_fitness,
                        'length': best_length,
                        'generation': generation
                        }

    if verb:
        print("generation", generation,
              "best", best_fitness,
              "worst", worst_fitness,
              "avg", avg_fitness,
              "length", best_length)

    stats.append(generation_stats)

    return


def show_stats(stats):
    best_fitness_list = []
    worst_fitness_list = []
    avg_fitness_list = []
    generations_list = []
    length_list = []
    for generation in range(len(stats)):
        generation_stats = stats[generation]
        best_fitness_list.append(generation_stats['best_fitness'])
        worst_fitness_list.append(generation_stats['worst_fitness'])
        avg_fitness_list.append(generation_stats['avg_fitness'])
        length_list.append(generation_stats['length'])
        generations_list.append(generation_stats['generation'])

    plt.figure()
    plt.plot(generations_list, best_fitness_list, 'b-', label='Best fitness')
    plt.plot(generations_list, worst_fitness_list, 'r-', label='Worst fitness')
    plt.plot(generations_list, avg_fitness_list, 'g-', label='Average fitness')
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.show()
    plt.figure()
    plt.plot(generations_list, length_list, 'b-', label='Length')
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Length of the best specie")
    plt.show()

    return


def simulation(generations, nb_of_species, nb_of_lamps, generator, offspring_size, mutation_probability_coord,
               mutation_probability_size, crossover_probability, tournament_size, radius, square, sigma_x, sigma_y,
               visualize_result=False, visualize_stats=False, verb=True):
    # for keeping stats
    stats = []

    # simulation

    # generating all the species
    species = generate_species(square, nb_of_species, generator, nb_of_lamps, radius)
    do_stats(stats, species, 0)


    for generation in range(1, generations + 1):

        species = copy.deepcopy(reproduction(generator, species, offspring_size, crossover_probability, tournament_size))

        # mutations
        for specie in species[1:len(species)]:  # we don't want to mutate the best specie
            # mutating
            specie[0] = copy.deepcopy(mutation(generator, mutation_probability_coord, mutation_probability_size, specie[0], sigma_x, sigma_y,
                     square))

            # computing and actualizing fitness
            specie[1] = evaluateLamps(specie[0], radius, square)
            
        for i in range(len(species)):
            if  species[i][1] != evaluateLamps(species[i][0], radius, square):
                print(i)

        # sorting the species based on entry [1], which is the fitness value
        species = sorted(species, key=lambda x: x[1], reverse=True)
        species = species[0:nb_of_species - 1]

        do_stats(stats, species, generation, verb)

    # calling the function; the argument "visualize=True" makes it plot the current situation
    bestLamps = species[0][0]
    fitness = evaluateLamps(bestLamps, radius, square, visualize=visualize_result)


    if visualize_stats:
        show_stats(stats)

    return stats


# this main is just here to try the function, and give you an idea of how it works
def main():
    generator = Random()  # create instance of pseudo-random number generator
    generator.seed(42)

    # sides of the square, [width, height]
    square = [1, 1]

    # radius of the lamps
    radius = 0.3

    # probabilities
    mutation_probability_coord = 0.15
    mutation_probability_size = 0.6
    crossover_probability = 0.9
    sigma_x = square[0] / 30
    sigma_y = square[1] / 30

    # offspring size
    offspring_size = 20

    nb_of_lamps = 1  # number of lamps per specie during the first generation
    nb_of_species = 20

    # number of generations
    generations = 30
    # tournament size
    tournament_size = 4

    stats = simulation(generations, nb_of_species, nb_of_lamps, generator, offspring_size, mutation_probability_coord,
                       mutation_probability_size, crossover_probability, tournament_size, radius, square, sigma_x,
                       sigma_y, True, True)

    return


if __name__ == "__main__":
    sys.exit(main())
