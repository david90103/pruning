from random import uniform
from algorithm.algorithm_base import *

class GWO(AlgorithmBase):

    def __init__(self, dimension, population_size):
        super().__init__()
        self.dimension = dimension
        self.population_size = population_size
        self.population = []
        self.alpha_fitness = float("inf")
        self.beta_fitness = float("inf")
        self.delta_fitness = float("inf")
        self.alpha_position = []
        self.beta_position = []
        self.delta_position = []
        self.a = 2
        self.best_fitness = float("inf")
        self.best_solution = None
    
    def evalAndUpdateWolves(self):
        # self.alpha_fitness = float("inf")
        # self.beta_fitness = float("inf")
        # self.delta_fitness = float("inf")

        for i in range(self.population_size):
            self.population[i].fitness, self.population[i].pruned, self.population[i].acc = self.evaluate(self.population[i].position)
            self.updateRatioBest(self.population[i])

        # for i in range(self.population_size):
        #     # Global best
        #     if self.population[i].fitness < self.best_fitness:
        #         self.best_fitness = self.population[i].fitness
        #         self.best_solution = self.population[i].position
        #     # Alpha wolf
        #     if self.population[i].fitness < self.alpha_fitness:
        #         self.alpha_fitness = self.population[i].fitness
        #         self.alpha_position = self.population[i].position
        #     # Beta wolf
        #     if self.population[i].fitness < self.beta_fitness and self.population[i].fitness > self.alpha_fitness:
        #         self.beta_fitness = self.population[i].fitness
        #         self.beta_position = self.population[i].position
        #     # Delta wolf
        #     if self.population[i].fitness < self.delta_fitness and self.population[i].fitness > self.alpha_fitness and self.population[i].fitness > self.beta_fitness:
        #         self.delta_fitness = self.population[i].fitness
        #         self.delta_position = self.population[i].position

        for i in range(self.population_size):
            # Global best
            if self.population[i].fitness < self.best_fitness:
                self.best_fitness = self.population[i].fitness
                self.best_solution = self.population[i].position.copy()
            # Alpha wolf
            if self.population[i].fitness < self.alpha_fitness:
                self.alpha_fitness = self.population[i].fitness
                self.alpha_position = self.population[i].position.copy()

        # Beta wolf
        for i in range(self.population_size):
            if self.population[i].fitness < self.beta_fitness and self.population[i].fitness > self.alpha_fitness:
                self.beta_fitness = self.population[i].fitness
                self.beta_position = self.population[i].position.copy()

        # Delta wolf
        for i in range(self.population_size):
            if self.population[i].fitness < self.delta_fitness and self.population[i].fitness > self.alpha_fitness and self.population[i].fitness > self.beta_fitness:
                self.delta_fitness = self.population[i].fitness
                self.delta_position = self.population[i].position.copy()

    def updatePosition(self):
        for i in range(self.population_size):
            for j in range(self.dimension):
                # Alpha
                r1 = uniform(0, 1)
                r2 = uniform(0, 1)
                A1 = 2 * self.a * r1 - self.a
                C1 = 2 * r2
                D_alpha = abs(C1 * self.alpha_position[j] - self.population[i].position[j])
                X1 = self.alpha_position[j] - A1 * D_alpha
                # Beta
                r1 = uniform(0, 1)
                r2 = uniform(0, 1)
                A2 = 2 * self.a * r1 - self.a
                C2 = 2 * r2
                D_beta = abs(C2 * self.beta_position[j] - self.population[i].position[j])
                X2 = self.beta_position[j] - A2 * D_beta
                # Delta
                r1 = uniform(0, 1)
                r2 = uniform(0, 1)
                A3 = 2 * self.a * r1 - self.a
                C3 = 2 * r2
                D_delta = abs(C3 * self.delta_position[j] - self.population[i].position[j])
                X3 = self.delta_position[j] - A3 * D_delta

                self.population[i].position[j] = (X1 + X2 + X3) / 3

                # TODO: Check if boundary is needed
                if self.population[i].position[j] > 0.9999:
                    self.population[i].position[j] = 0.9999
                if self.population[i].position[j] < 0:
                    self.population[i].position[j] = 0
    
    def printWolves(self):
        print("\nAlpha fitness", self.alpha_fitness)
        print(self.roundSolution(self.alpha_position))

        print("Beta fitness", self.beta_fitness)
        print(self.roundSolution(self.beta_position))

        print("Delta fitness", self.delta_fitness)
        print(self.roundSolution(self.delta_position))
        print()
    
    def run(self, iterations, evaluate_func):
        print("Initialize population")
        self.evaluate = evaluate_func
        self.population = self.initEvalPopulation()
        print("Start GWO search")

        for it in range(iterations):
            # Evaluate and update three wolves
            self.evalAndUpdateWolves()
            # Update a
            # TODO: Different update strategy of a
            self.a = 2 - it * (2 / iterations)
            self.updatePosition()

            print("Iteration", it + 1, "end, best fitness", self.best_fitness, self.roundSolution(self.best_solution))
        
        return self.best_solution, self.ratio_best
