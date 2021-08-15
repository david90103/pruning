import copy
from algorithm.algorithm_base import *
from random import randint, uniform

class SEB(AlgorithmBase):

    def __init__(self, dimension, population_size, region_num, searcher_num, sample_num, c_thre, m_thre):
        super().__init__()
        self.dimension = dimension
        self.region_num = int(region_num)
        self.searcher_num = int(searcher_num)
        self.sample_num = int(sample_num)
        # self.cthre = c_thre
        self.mthre = m_thre
        self.best_fitness = float("inf")
        self.best_solution = None
        self.evaluate_bin = None
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
        for r in range(self.region_num):
            p = (r * 2 + 1) / (self.region_num * 2)
            for s in range(self.sample_num):
                for d in range(self.dimension):
                    self.sample[r][s].position[d] = 1 if uniform(0, 1) < p else 0

                self.sample[r][s].fitness, self.sample[r][s].pruned, self.sample[r][s].acc = self.evaluate(self.sample[r][s].position)
                self.updateRatioBest(self.sample[r][s])
                self.evaluate_count += 1

        for s in range(self.searcher_num):
            for d in range(self.dimension):
                self.searcher[s].position[d] = randint(0, 1)
            
            self.searcher[s].fitness, self.searcher[s].pruned, self.searcher[s].acc = self.evaluate(self.searcher[s].position)
            self.updateRatioBest(self.searcher[s])
            self.evaluate_count += 1

    def visionSearch(self, curr_iter, total_iter):
        # Transition
        for s in range(self.searcher_num):
            for r in range(self.region_num):
                # Region bound for mutation
                for i in range(self.sample_num):
                    child1 = Solution(self.dimension)
                    child2 = Solution(self.dimension)
                    # NOTE: Uniform crossover
                    # for d in range(self.dimension):
                    #     if uniform(0, 1) > curr_iter / (total_iter * 2) + 0.25:
                    #         child1.position[d] = self.sample[r][i].position[d]
                    #     else:
                    #         child1.position[d] = self.searcher[s].position[d]
                    # # Child 2
                    # for d in range(self.dimension):
                    #     if uniform(0, 1) > curr_iter / (total_iter * 2) + 0.25:
                    #         child1.position[d] = self.sample[r][i].position[d]
                    #     else:
                    #         child1.position[d] = self.searcher[s].position[d]

                    # NOTE: One point crossover
                    pivot = randint(0, self.dimension - 1)
                    child1.position = self.searcher[s].position[:pivot] + self.sample[r][i].position[pivot:]
                    child2.position = self.sample[r][i].position[:pivot] + self.searcher[s].position[pivot:]
                    
                    # Mutation
                    for d in range(self.dimension):
                        if uniform(0, 1) < self.mthre:
                            child1.position[d] = 1 if child1.position[d] == 0 else 0
                        if uniform(0, 1) < self.mthre:
                            child2.position[d] = 1 if child2.position[d] == 0 else 0

                    # Evaluation
                    child1.fitness, child1.pruned, child1.acc = self.evaluate(child1.position)
                    child2.fitness, child2.pruned, child2.acc = self.evaluate(child2.position)
                    self.updateRatioBest(child1)
                    self.updateRatioBest(child2)
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
        # NOTE: Show all searchers and region samples
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
    
    def evaluate(self, target):
        """Override base class"""
        return self.evaluate_bin(target, is_binary=True) # TODO Better way to do this

    def run(self, iterations, evaluate_func):
        print("Initialize population")
        print("SE modified parameters: {} regions, {} searchers, {} samples".format(self.region_num, self.searcher_num, self.sample_num))
        self.evaluate_bin = evaluate_func
        print("Start SE modified search")

        self.initialize()
        self.resourceArrangement()

        for it in range(iterations):
            self.visionSearch(it, iterations)
            self.marketingResearch()
            if (it == iterations // 2 or it == iterations // 4 * 3) and self.region_num > 1:
                self.printPopulation()
                self.reduceRegion()
                self.printPopulation()
            self.printRatioBest()
            print("Evaluations after this iteration:", str(self.evaluate_count))
            print("Iteration", it + 1, "end, best fitness", self.best_fitness, self.roundSolution(self.best_solution))

        return self.best_solution, self.ratio_best
