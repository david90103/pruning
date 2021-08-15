import copy
import numpy as np
from algorithm.algorithm_base import *
from random import randint, uniform

class SEM(AlgorithmBase):

    def __init__(self, dimension, population_size, region_num, searcher_num, sample_num, cthre, mthre):
        super().__init__()
        self.dimension = dimension
        self.region_num = int(region_num)
        self.searcher_num = int(searcher_num)
        self.sample_num = int(sample_num)
        self.cthre = float(cthre)
        self.mthre = float(mthre)
        self.f = 0.5
        self.best_fitness = float("inf")
        self.region_history_best = []
        self.searcher_belong = []
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

        self.region_history_best = [Solution(self.dimension) for _ in range(self.region_num)]
        self.searcher_belong = [-1 for _ in range(self.searcher_num)]
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

                self.sample[r][s].fitness, self.sample[r][s].pruned, self.sample[r][s].acc = self.evaluate(self.sample[r][s].position)
                self.updateRatioBest(self.sample[r][s])
                self.evaluate_count += 1

        for s in range(self.searcher_num):
            for d in range(self.dimension):
                self.searcher[s].position[d] = uniform(0, 1)
            
            self.searcher[s].fitness, self.searcher[s].pruned, self.searcher[s].acc = self.evaluate(self.searcher[s].position)
            self.updateRatioBest(self.searcher[s])
            self.searcher_belong[s] = -1
            self.evaluate_count += 1

    def tournament(self, r):
        a = self.sample[r][randint(0, self.sample_num - 1)]
        b = self.sample[r][randint(0, self.sample_num - 1)]
        c = 0
        while a == b:
            c += 1
            if c > 1000:
                print("over 1000")
                print("print whole region samples")
                for s in self.sample[r]:
                    print(s.position)
                print("comparing a and b")
                print(a.fitness, a.acc, a.pruned, a.position)
                print(b.fitness, b.acc, b.pruned, b.position)
                exit(1)
            a = self.sample[r][randint(0, self.sample_num - 1)]

        if a.fitness < b.fitness:
            return a

        return b

    def transition(self, s, r, i):
        child = Solution(self.dimension)
        rand_sol_pos = self.sample[r][randint(0, self.sample_num - 1)].position.copy()
        rand_sol_pos_2 = self.sample[r][randint(0, self.sample_num - 1)].position.copy()
        # tbest = self.tournament(r)

        # NOTE: DE like transition
        # if self.searcher_belong == r:
        #     # current-to-tbest
        #     for d in range(self.dimension):
        #         child.position[d] = self.sample[r][i].position[d] + \
        #                             self.f * (self.searcher[s].position[d] - self.sample[r][i].position[d]) + \
        #                             self.f * (tbest.position[d] - rand_sol_pos[d])
        # else:
        #     # random-to-tbest
        #     for d in range(self.dimension):
        #         child.position[d] = rand_sol_pos[d] + \
        #                             self.f * (self.searcher[s].position[d] - rand_sol_pos[d]) + \
        #                             self.f * (tbest.position[d] - rand_sol_pos_2[d])

        # NOTE: PSO like transition
        for d in range(self.dimension):
            if uniform(0, 1) < self.mthre:
                # child.position[d] = uniform(0, 1)
                child.position[d] = np.random.normal(self.sample[r][i].position[d], 0.1)
            elif uniform(0, 1) < self.cthre:
                child.position[d] = rand_sol_pos[d] + uniform(0, 1) * (self.searcher[s].position[d] - rand_sol_pos_2[d])
            else:
                child.position[d] = self.sample[r][i].position[d]
            # Boundary
            if child.position[d] < 0:
                child.position[d] = 0.00

        return child
                
    def visionSearch(self, curr_iter, total_iter):
        # Transition
        for s in range(self.searcher_num):
            for r in range(self.region_num):
                # Region bound for mutation
                interval = 1 / self.region_num
                lbound = max(0, self.region[r] - interval)
                rbound = min(1, self.region[r] + interval)
                for i in range(self.sample_num):
                    # Transition
                    child1 = self.transition(s, r, i)
                    child2 = self.transition(s, r, i)
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

            # self.searcher[s] = self.sample[chooseR][0]
            for i in range(self.sample_num):
                # Use region current best as searcher
                # if self.searcher[s].fitness > self.sample[chooseR][i].fitness:
                #     self.searcher[s] = self.sample[chooseR][i]
                # Use region history best as searcher
                if self.region_history_best[chooseR].fitness > self.sample[chooseR][i].fitness:
                    self.region_history_best[chooseR] = self.sample[chooseR][i]
            
            self.searcher[s] = self.region_history_best[chooseR]
            self.searcher_belong[s] = chooseR
            
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

        # Reassign region history best
        new_rhb = []
        for i in range(self.region_num):
            if self.region_history_best[2 * i].fitness < self.region_history_best[2 * i + 1].fitness:
                new_rhb.append(self.region_history_best[2 * i])
            else:
                new_rhb.append(self.region_history_best[2 * i + 1])
        
        assert len(new_rhb) == len(self.region_history_best) // 2

        self.region_history_best = new_rhb

        for r in range(self.region_num):
            self.region[r] = (r * 2 + 1) / (self.region_num * 2)

        self.ta = [1 for _ in range(self.region_num)]
        self.tb = [1 for _ in range(self.region_num)]
        self.rBest = [float("inf") for _ in range(self.region_num)]
    
    def run(self, iterations, evaluate_func):
        print("Initialize population")
        print("SE modified parameters: {} regions, {} searchers, {} samples".format(self.region_num, self.searcher_num, self.sample_num))
        self.evaluate = evaluate_func
        print("Start SE modified search")

        self.initialize()
        self.resourceArrangement()

        for it in range(iterations):
            self.visionSearch(it, iterations)
            self.marketingResearch()
            if (it == iterations // 2 or it == iterations // 4 * 3) and self.region_num > 1:
            # if (it == iterations // 2) and self.region_num > 1:
                self.printPopulation()
                self.reduceRegion()
                self.printPopulation()
            self.printRatioBest()
            print("Evaluations after this iteration:", str(self.evaluate_count))
            print("Iteration", it + 1, "end, best fitness", self.best_fitness, self.roundSolution(self.best_solution))

            # if self.evaluate_count >= 3200:
            #     break

        return self.best_solution, self.ratio_best
