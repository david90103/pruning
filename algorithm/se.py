from algorithm.functions import ackley
import numpy as np
import copy
import algorithm.algorithm_base
from random import randint, uniform

class Solution:

    def __init__(self, dimension):
        self.fitness = float("inf")
        self.position = [0 for _ in range(dimension)]

class SE(algorithm.algorithm_base.AlgorithmBase):

    def __init__(self, dimension, population_size, region_num, searcher_num, sample_num):
        self.dimension = dimension
        # self.population_size = 8  # FIXME: Hard coded population size!!!!!
        # self.region_num = 4
        # self.searcher_num = 2
        # self.sample_num = self.population_size // self.region_num
        self.region_num = int(region_num)
        self.searcher_num = int(searcher_num)
        self.sample_num = int(sample_num)
        self.best_fitness = float("inf")
        self.best_solution = None
        self.evaluate_count = 0
    
    def initialize(self):
        sample_solutions = [Solution(self.dimension) for _ in range(self.sample_num)]
        sample_2_solutions = [Solution(self.dimension) for _ in range(self.sample_num * 2)]
        region_solutions = [copy.deepcopy(sample_2_solutions) for _ in range(self.region_num)]
        self.searcher = [Solution(self.dimension) for _ in range(self.searcher_num)]
        self.sample = [copy.deepcopy(sample_solutions) for _ in range(self.region_num)]
        self.sampleV = [copy.deepcopy(region_solutions) for _ in range(self.searcher_num)]

        self.region = [1 for _ in range(self.region_num + 1)]
        for r in range(self.region_num):
            self.region[r] = (r * 2 + 1) / (self.region_num * 2)

        self.ta = [1 for _ in range(self.region_num)]
        self.tb = [1 for _ in range(self.region_num)]
        self.rBest = [float("inf") for _ in range(self.region_num)]

    def resourceArrangement(self):
        interval = 1 / self.region_num
        for r in range(self.region_num):
            lbound = max(0, self.region[r] - interval)
            rbound = min(1, self.region[r] + interval)
            for s in range(self.sample_num):
                for d in range(self.dimension):
                    self.sample[r][s].position[d] = uniform(lbound, rbound)
                    # self.sample[r][s].position[d] = 1 if p < self.region[r] else 0

                self.sample[r][s].fitness = self.evaluate(self.sample[r][s].position)
                self.evaluate_count += 1

        for s in range(self.searcher_num):
            for d in range(self.dimension):
                self.searcher[s].position[d] = uniform(0, 1)
                # searcher[s].position[d] = 1 if p < 0.5 else 0
            
            self.searcher[s].fitness = self.evaluate(self.searcher[s].position)
            self.evaluate_count += 1

    def visionSearch(self, curr_iter, total_iter):
        
        # self.printPopulation()

        # Transition
        for s in range(self.searcher_num):
            for r in range(self.region_num):
                for i in range(self.sample_num):
                    child1 = Solution(self.dimension)
                    child2 = Solution(self.dimension)

                    # NOTE: GA crossover
                    # Uniform crossover performs worse than random search
                    # for d in range(self.dimension):
                    #     p1 = uniform(0, 1)
                    #     p2 = uniform(0, 1)
                    #     thre = curr_iter / (total_iter * 2) + 0.25
                    #     if p1 > thre:
                    #         child1.position[d] = self.sample[r][i].position[d]
                    #     else:
                    #         child1.position[d] = self.searcher[s].position[d]

                    #     if p2 > thre:
                    #         child2.position[d] = self.sample[r][i].position[d]
                    #     else:
                    #         child2.position[d] = self.searcher[s].position[d]

                    # Single point crossover
                    # cross_point = randint(0, len(child1.position))
                    # child1.position = self.sample[r][i].position[:cross_point] + self.searcher[s].position[cross_point:]
                    # child2.position = self.searcher[s].position[:cross_point] + self.sample[r][i].position[cross_point:]

                    # NOTE: DE like transition
                    # Child 1
                    for d in range(self.dimension):
                        # if uniform(0, 1) < 0.5:
                        #     child1.position[d] = self.sample[r][i].position[d]
                        # else:
                            child1.position[d] = self.sample[r][i].position[d] + \
                                    uniform(0, 1) * (self.searcher[s].position[d] - self.sample[r][i].position[d])
                    # Child 2
                    for d in range(self.dimension):
                        # if uniform(0, 1) < 0.5:
                        #     child2.position[d] = self.sample[r][i].position[d]
                        # else:
                            child2.position[d] = self.sample[r][i].position[d] + \
                                    uniform(0, 1) * (self.searcher[s].position[d] - self.sample[r][i].position[d])
                    
                    # NOTE: Mutation swap
                    # m1 = randint(0, self.dimension - 1)
                    # m2 = randint(0, self.dimension - 1)
                    # temp = child1.position[m1]
                    # child1.position[m1] = child1.position[m2]
                    # child1.position[m2] = temp
                    # m1 = randint(0, self.dimension - 1)
                    # m2 = randint(0, self.dimension - 1)
                    # temp = child2.position[m1]
                    # child2.position[m1] = child2.position[m2]
                    # child2.position[m2] = temp

                    # NOTE: Mutation random
                    # Mutate 1 values of each child
                    m1 = randint(0, self.dimension - 1)
                    m2 = randint(0, self.dimension - 1)
                    child1.position[m1] = uniform(0, 1)
                    child2.position[m2] = uniform(0, 1)

                    # Evaluation
                    child1.fitness = self.evaluate(child1.position)
                    child2.fitness = self.evaluate(child2.position)
                    self.evaluate_count += 2

                    if child1.fitness < self.best_fitness:
                        self.best_fitness = child1.fitness
                        self.best_solution = child1.position.copy()
                    if child2.fitness < self.best_fitness:
                        self.best_fitness = child2.fitness
                        self.best_solution = child2.position.copy()
                    
                    self.sampleV[s][r][i * 2] = child1
                    self.sampleV[s][r][i * 2 + 1] = child2

        sum_of_sample = 0
        for r in range(self.region_num):
            for i in range(self.sample_num):
                sum_of_sample += self.sample[r][i].fitness
                if self.sample[r][i].fitness < self.rBest[r]:
                    self.rBest[r] = self.sample[r][i].fitness
        
        T = [[0 for _ in range(self.region_num)] for _ in range(self.searcher_num)]
        V = [[0 for _ in range(self.region_num)] for _ in range(self.searcher_num)]
        M = [[0 for _ in range(self.region_num)] for _ in range(self.searcher_num)]
        for s in range(self.searcher_num):
            for r in range(self.region_num):
                T[s][r] = self.tb[r] / self.ta[r]

                for i in range(self.sample_num * 2):
                    V[s][r] += self.sampleV[s][r][i].fitness

                V[s][r] /= self.sample_num * 2
                # V[s][r] = 1 - V[s][r]

                M[s][r] = self.rBest[r] / sum_of_sample
        
        # Normalize
        E = [[0 for _ in range(self.region_num)] for _ in range(self.searcher_num)]
        for s in range(self.searcher_num):
            for r in range(self.region_num):
                E[s][r] = T[s][r] * V[s][r] * M[s][r]
        
        # Sample update sampleV
        for r in range(self.region_num):
            for s in range(self.searcher_num):
                for i in range(self.sample_num):
                    max_sample = 0
                    for j in range(self.sample_num):
                        if self.sample[r][max_sample].fitness < self.sample[r][j].fitness:
                            max_sample = j
                    if self.sampleV[s][r][i].fitness < self.sample[r][max_sample].fitness:
                        self.sample[r][max_sample] = self.sampleV[s][r][i]

        # Determination
        for r in range(self.region_num):
            self.tb[r] += 1
        
        for s in range(self.searcher_num):
            chooseR = 0
            chooseE = E[s][0]
            for r in range(self.region_num):
                if chooseE < E[s][r]:
                    chooseR = r
                    chooseE = E[s][r]

            self.searcher[s] = self.sample[chooseR][0]
            for i in range(self.sample_num):
                if self.searcher[s].fitness > self.sample[chooseR][i].fitness:
                    self.searcher[s] = self.sample[chooseR][i]
            
            self.tb[chooseR] = 1
            self.ta[chooseR] += 1

    def marketingResearch(self):
        for r in range(self.region_num):
            if self.tb[r] > 1:
                self.ta[r] = 1
        for s in range(self.searcher_num):
            if self.searcher[s].fitness < self.best_fitness:
                self.best_fitness = self.searcher[s].fitness
                self.best_solution = self.searcher[s].position
    
    def printPopulation(self):
        """Show all searchers and region samples"""

        for s in range(self.searcher_num):
            print("Searcher", str(s))
            print(self.roundSolution(self.searcher[s].position))

        for r in range(self.region_num):
            print("Region", str(r))
            for s in range(self.sample_num):
                print(self.roundSolution(self.sample[r][s].position))
        print()
    
    def reduceRegion(self):
        """Reduce regions to 1/2 and increase 2x of samples in each region."""

        region_num_origin = self.region_num
        sample_num_origin = self.sample_num
        self.region_num = self.region_num // 2
        self.sample_num = self.sample_num * 2

        # Reassign sample
        queue = []
        for i in range(region_num_origin):
            for j in range(sample_num_origin):
                queue.append(self.sample[i][j])

        sample_solutions = [Solution(self.dimension) for _ in range(self.sample_num)]
        self.sample = [copy.deepcopy(sample_solutions) for _ in range(self.region_num)]

        for i in range(self.region_num):
            for j in range(self.sample_num):
                self.sample[i][j] = queue.pop(0)

        assert len(queue) == 0

        # Reassign sampleV
        queue = []
        for i in range(self.searcher_num):
            for j in range(region_num_origin):
                for k in range(sample_num_origin * 2):
                    queue.append(self.sampleV[i][j][k])

        sample_2_solutions = [Solution(self.dimension) for _ in range(self.sample_num * 2)]
        region_solutions = [copy.deepcopy(sample_2_solutions) for _ in range(self.region_num)]
        self.sampleV = [copy.deepcopy(region_solutions) for _ in range(self.searcher_num)]

        for i in range(self.searcher_num):
            for j in range(self.region_num):
                for k in range(self.sample_num * 2):
                    self.sampleV[i][j][k] = queue.pop(0)

        assert len(queue) == 0

        for r in range(self.region_num):
            self.region[r] = (r * 2 + 1) / (self.region_num * 2)

        self.ta = [1 for _ in range(self.region_num)]
        self.tb = [1 for _ in range(self.region_num)]
        self.rBest = [float("inf") for _ in range(self.region_num)]

    def run(self, iterations, evaluate_func):
        print("Initialize population")
        print("SE parameters: {} regions, {} searchers, {} samples".format(self.region_num, self.searcher_num, self.sample_num))
        self.evaluate = evaluate_func
        # self.population, self.fitness_values = self.initEvalPopulation()
        print("Start SE search")

        self.initialize()
        self.resourceArrangement()

        for it in range(iterations):
            self.visionSearch(it, iterations)
            self.marketingResearch()
            if it == iterations // 2 or it == iterations // 4 * 3:
                self.printPopulation()
                self.reduceRegion()
                self.printPopulation()
            print("Evaluations after this iteration:", str(self.evaluate_count))
            print("Iteration", it + 1, "end, best fitness", self.best_fitness, self.roundSolution(self.best_solution))

            # if self.evaluate_count >= 3200:
            #     break

        return self.best_solution
