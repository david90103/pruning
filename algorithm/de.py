import numpy as np
from algorithm.algorithm_base import *
from random import randint, uniform

class DE(AlgorithmBase):

    def __init__(self, dimension, population_size, crossover_rate, f):
        super().__init__()
        self.dimension = dimension
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.f = f
        self.population = []
        self.best_objective = float("inf")
        self.best_solution = None
    
    def crossover(self, population, v_arr):
        u_arr = []
        for i in range(self.population_size):
            temp = []
            for j in range(len(population[i].position)):
                if i == j or uniform(0, 1) < self.crossover_rate:
                    temp.append(v_arr[i].position[j])
                else:
                    temp.append(population[i].position[j])

            solution = Solution(self.dimension)
            solution.position = temp
            u_arr.append(solution)
       
        return u_arr

    def mutation(self, population):
        v_arr = []
        for i in range(self.population_size):
            temp = []
            r1 = randint(0, self.population_size - 1)
            r2 = randint(0, self.population_size - 1)
            r3 = randint(0, self.population_size - 1)
            for j in range(len(population[i].position)):
                mutation_value = population[r1].position[j] + self.f * (population[r2].position[j] - population[r3].position[j])
                if mutation_value > 0.99999:
                    mutation_value = 0.99999
                if mutation_value < 0:
                    mutation_value = 0
                temp.append(mutation_value)

            solution = Solution(self.dimension)
            solution.position = temp
            v_arr.append(solution)
       
        return v_arr
    
    def run(self, iterations, evaluate_func):
        print("Initialize population")
        self.evaluate = evaluate_func
        self.population = self.initEvalPopulation()
        print("Start DE search")

        for iteration in range(iterations):
            v_arr = self.mutation(self.population)
            u_arr = self.crossover(self.population, v_arr)

            for i in range(self.population_size):
                u_arr[i].fitness, u_arr[i].pruned, u_arr[i].acc = self.evaluate(u_arr[i].position)
                self.updateRatioBest(u_arr[i])

                if u_arr[i].fitness < self.population[i].fitness:
                    self.population[i] = u_arr[i]
                if u_arr[i].fitness < self.best_objective:
                    self.best_objective = u_arr[i].fitness
                    self.best_solution = u_arr[i]

            self.printRatioBest()
            print("Iteration", iteration + 1, "end, best fitness", self.best_objective, self.roundSolution(self.best_solution.position))
        
        return self.best_solution.position, self.ratio_best
