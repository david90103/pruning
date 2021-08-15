from random import uniform
from algorithm.algorithm_base import *

class Random(AlgorithmBase):

    def __init__(self, dimension, log_step=30):
        super().__init__()
        self.dimension = dimension
        self.log_step = log_step
        self.best_fitness = float("inf")
        self.best_solution = None

    def run(self, arch_num, evaluate_func):
        self.evaluate = evaluate_func
        print("Start Random search")

        for i in range(arch_num):
            new_solution = Solution(self.dimension)
            new_solution.position = [uniform(0, 1) for _ in range(self.dimension)]
            new_solution.fitness, new_solution.pruned, new_solution.acc = self.evaluate(new_solution.position)
            self.updateRatioBest(new_solution)
            if new_solution.fitness < self.best_fitness:
                self.best_fitness = new_solution.fitness
                self.best_solution = new_solution.position.copy()
            if i % self.log_step == 0:
                print("Random search", i + 1, "solutions, best fitness", self.best_fitness, self.roundSolution(self.best_solution))
        
        return self.best_solution, self.ratio_best
