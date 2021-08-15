import numpy as np

class Solution:
    def __init__(self, dimension):
        self.fitness = float("inf")
        self.pruned = 0
        self.acc = 0
        self.position = [0 for _ in range(dimension)]

    def __eq__(self, other):
        if isinstance(other, Solution):
            return (self.fitness == other.fitness) and (self.pruned == other.pruned) and (self.acc == other.acc) and (self.position == other.position)
        return False

class AlgorithmBase:
    def __init__(self):
        self.dimension = 0
        self.ratio_best = [Solution(self.dimension) for _ in range(20)]
    
    def initEvalPopulation(self):
        population = []
        for _ in range(self.population_size):
            random_solution = Solution(self.dimension)
            random_solution.position = [np.random.uniform() for _ in range(self.dimension)]
            population.append(random_solution)
        return population

    def evaluate(self, target):
        raise Exception("Evaluate function is not set.")
    
    def roundSolution(self, solution):
        return list(map(lambda x: round(x, 2), solution))
    
    def updateRatioBest(self, solution):
        assert isinstance(solution, Solution)
        idx = int(solution.pruned * 100 // 5)
        if solution.fitness < self.ratio_best[idx].fitness:
            self.ratio_best[idx] = solution

    def printRatioBest(self):
        print("Ratio best")
        for s in self.ratio_best:
            print("Pruned:", round(s.pruned, 4), "Acc:", round(s.acc, 4), "Fitness:", round(s.fitness, 4))
