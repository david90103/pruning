import algorithm.algorithm_base
from random import uniform

class PSO(algorithm.algorithm_base.AlgorithmBase):

    def __init__(self, dimension, population_size, w, c1, c2):
        super().__init__()
        self.dimension = dimension
        self.population_size = population_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.population = []
        self.velocity = []
        self.individual_best = []
        self.individual_best_pos = []
        self.best_objective = float("inf")
        self.best_solution = None
    
    def evaluatePopulation(self):
        for i in range(self.population_size):
            self.population[i].fitness, self.population[i].pruned, self.population[i].acc = self.evaluate(self.population[i].position)
            self.updateRatioBest(self.population[i])
        # Global best
        for i in range(self.population_size):
            if self.population[i].fitness < self.best_objective:
                self.best_objective = self.population[i].fitness
                self.best_solution = self.population[i].position.copy()
        # Individual best
        for i in range(self.population_size):
            if self.population[i].fitness < self.individual_best[i]:
                self.individual_best[i] = self.population[i].fitness
                self.individual_best_pos[i] = self.population[i].position.copy()
    
    def initIndividual(self):
        for _ in range(self.population_size):
            self.individual_best.append(float("inf"))
            self.individual_best_pos.append([])
    
    def initVelocity(self):
        for _ in range(self.population_size):
            self.velocity.append([uniform(0, 1) for _ in range(self.dimension)])
    
    def updatePosition(self):
        for i in range(self.population_size):
            for j in range(self.dimension):
                self.velocity[i][j] = self.w * self.velocity[i][j] + \
                                      self.c1 * uniform(0, 1) * (self.individual_best_pos[i][j] - self.population[i].position[j]) + \
                                      self.c2 * uniform(0, 1) * (self.best_solution[j] - self.population[i].position[j])
        for i in range(self.population_size):
            for j in range(self.dimension):
                self.population[i].position[j] += self.velocity[i][j]
                
                # TODO: Check if boundary is needed
                # if self.population[i].position[j] > 0.9999:
                #     self.population[i].position[j] = 0.9999
                # if self.population[i].position[j] < 0:
                #     self.population[i].position[j] = 0

    def run(self, iterations, evaluate_func):
        print("Initialize population")
        self.evaluate = evaluate_func
        self.population = self.initEvalPopulation()
        self.initIndividual()
        self.initVelocity()
        print("Start PSO search")

        for iteration in range(iterations):
            self.evaluatePopulation()
            self.updatePosition()
            print("Iteration", iteration + 1, "end, best fitness", self.best_objective, self.roundSolution(self.best_solution))
        
        return self.best_solution, self.ratio_best
