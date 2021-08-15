from random import randint, uniform
from algorithm.algorithm_base import *

class GA(AlgorithmBase):

    def __init__(self, dimension, population_size, crossover_rate, mutation_rate):
        super().__init__()
        self.dimension = dimension
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = []
        self.fitness_value_sum = 0
        self.best_fitness = float("inf")
        self.best_solution = None

    def evaluatePopulation(self, population):
        for i in range(len(population)):
            self.population[i].fitness, self.population[i].pruned, self.population[i].acc = self.evaluate(population[i].position)
            self.updateRatioBest(self.population[i])
            if self.population[i].fitness < self.best_fitness:
                self.best_fitness = self.population[i].fitness
                self.best_solution = population[i].position

    def crossover(self, father, mother):
        # NOTE: Single point crossover
        # pivot = randint(0, self.dimension - 1)
        # child1 = father.position[:pivot] + mother.position[pivot:]
        # child2 = mother.position[:pivot] + father.position[pivot:]
        # NOTE: Uniform crossover
        child1 = Solution(self.dimension)
        child2 = Solution(self.dimension)
        for i in range(len(father.position)):
            if uniform(0, 1) < 0.5:
                child1.position[i] = father.position[i]
                child2.position[i] = mother.position[i]
            else:
                child1.position[i] = mother.position[i]
                child2.position[i] = father.position[i]

        return child1, child2

    def rouletteWheel(self):
        raise NotImplementedError("Roulette Wheel selection is not implemented.")

    def tournament(self):
        i = randint(0, self.population_size - 1)
        j = randint(0, self.population_size - 1)
        if self.population[i].fitness < self.population[j].fitness:
            return self.population[i]
        return self.population[j]

    def mutation(self, target):
        for i in range(len(target.position)):
            if uniform(0, 1) < self.mutation_rate:
                target.position[i] = uniform(0, 1)

        return target
    
    def run(self, generations, evaluate_func):
        print("Initialize population")
        self.evaluate = evaluate_func
        self.population = self.initEvalPopulation()
        print("Start GA search")

        for gen in range(generations):
            new_population = []

            # Selection with tournament and crossover
            for i in range(self.population_size // 2):
                a, b = self.tournament(), self.tournament()
                if uniform(0, 1) < self.crossover_rate:
                    child1, child2 = self.crossover(a, b)
                    new_population.append(child1)
                    new_population.append(child2)
                else:
                    new_population.append(a)
                    new_population.append(b)

            if self.population_size & 1:
                new_population.append(self.population[randint(0, self.population_size - 1)])

            # Mutate every subsolution for every solution
            for i in range(len(new_population)):
                new_population[i] = self.mutation(new_population[i])

            self.population = new_population
            self.evaluatePopulation(self.population)
            print("Generation", gen + 1, "end, best fitness", self.best_fitness, self.roundSolution(self.best_solution))

        return self.best_solution, self.ratio_best
