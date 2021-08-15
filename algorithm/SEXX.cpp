#include "SEXX.h"

SEXX::SEXX(int dIndex, int pCnt, int iter, int eval): Metaheuristic(dIndex, pCnt, iter, eval) {
    
    regionCnt = 8;
    searcherCnt = 2;
    sampleCnt = population / regionCnt;

    globalBest = Solution(dimension);
}

Solution SEXX::algorithm() {

    initialization();
    resourceArrangement();
    for (int iter = 0; iter < iteration; ++iter) {
        visionSearch(iter);
        marketingResearch();
    }
    cout << globalBest.fitness.fs << endl;

    return globalBest;
}

void SEXX::initialization() {
    globalBest.position.clear();
    globalBest.fitness.fs = __DBL_MAX__;

    searcher = vector<Solution>(searcherCnt, Solution(dimension));
    sample   = vector<vector<Solution>>(regionCnt, vector<Solution>(sampleCnt, Solution(dimension)));
    sampleV  = vector<vector<vector<Solution>>>(searcherCnt, vector<vector<Solution>>(regionCnt, vector<Solution>(sampleCnt * 2, Solution(dimension))));

    cout << searcher.size() << endl;
    
    cout << sample.size() << endl;
    cout << sample[0].size() << endl;

    cout << sampleV.size() << endl;
    cout << sampleV[0].size() << endl;
    cout << sampleV[0][0].size() << endl;
    cout << endl;

    // for (int jj = 0; jj < searcher[0].position.size(); jj++) {
    //     cout << searcher[0].position.at(jj);
    // }
    // cout << endl;
    // exit(0);
    // string aa;
    // cin >> aa;

    region = Double1(regionCnt + 1, 1);
    
    ta     = Double1(regionCnt, 1);
    tb     = Double1(regionCnt, 1);
    rBest  = Double1(regionCnt, __DBL_MAX__);

    evaluateCount = 0;
}

void SEXX::resourceArrangement() {

    // Devide the region 
    // Ex: [0.125, 0.375, 0.625, 0.875]
    for (int r = 0; r < regionCnt; ++r) {
        region[r] = (double) (r * 2 + 1) / (regionCnt * 2);
    }
    
    for (int r = 0; r < regionCnt; ++r)
        for (int s = 0; s < sampleCnt; ++s) {
            for (int d = 0; d < dimension; ++d) {
                double p = random01();
                sample[r][s].position[d] = (p < region[r])? 1: 0;
            }
            sample[r][s].fitness = getFitness( sample[r][s] );
        }

    for(int s = 0; s < searcherCnt; ++s) {
        for(int d = 0; d < dimension; ++d) {
            double p = random01();
            searcher[s].position[d] = (p < 0.5)? 1: 0;
        }
        searcher[s].fitness = getFitness( searcher[s] );
    }
}

void SEXX::visionSearch(int iter) {

    // transition
    for (int s = 0; s < searcherCnt; ++s)
        for (int r = 0; r < regionCnt; ++r)
            for (int i = 0; i < sampleCnt; ++i) {
                
                // crossover
                Solution child1(dimension), child2(dimension); 
                for (int d = 0; d < dimension; ++d) {
                    double p1 = random01(),
                           p2 = random01();
                    if (p1 > (double) iter / (iteration * 2) + 0.25 ) 
                        child1.position[d] = sample[r][i].position[d];
                    else 
                        child1.position[d] = searcher[s].position[d];

                    if (p2 > (double) iter / (iteration * 2) + 0.25 ) 
                        child2.position[d] = sample[r][i].position[d];
                    else 
                        child2.position[d] = searcher[s].position[d];
                }
                
                // mutation
                for (int i = 0; i < 2; ++i) {
                    int m1 = random(0, dimension-1),
                        m2 = random(0, dimension-1);
                    child1.position[m1] = (child1.position[m1] == 0)? 1: 0;
                    child2.position[m2] = (child2.position[m2] == 0)? 1: 0;
                }
                
                // evaluation
                child1.fitness = getFitness( child1 );
                child2.fitness = getFitness( child2 );

                if (globalBest.fitness > child1.fitness) globalBest = child1;
                if (globalBest.fitness > child2.fitness) globalBest = child2;

                sampleV[s][r][i*2]   = child1;
                sampleV[s][r][i*2+1] = child2;
            }



    double sumOfSample = 0;
    for (int r = 0; r < regionCnt; ++r) {
        for (int i = 0; i < sampleCnt; ++i) {
            sumOfSample += sample[r][i].fitness.fs;
            if (sample[r][i].fitness.fs < rBest[r]) 
                rBest[r] = sample[r][i].fitness.fs;
        }
    }
    // cout << sumOfSample << endl;
    Double2 T = Double2(searcherCnt, Double1(regionCnt, 0));
    Double2 V = Double2(searcherCnt, Double1(regionCnt, 0));
    Double2 M = Double2(searcherCnt, Double1(regionCnt, 0));
    for (int s = 0; s < searcherCnt; ++s) {
        for (int r = 0; r < regionCnt; ++r) {
            T[s][r] = tb[r] / ta[r];

            for (int i = 0; i < sampleCnt * 2; ++i) 
                V[s][r] += sampleV[s][r][i].fitness.fs;

            V[s][r] /= (sampleCnt * 2);

            // V[s][r] = 1 - V[s][r];

            M[s][r] = (rBest[r] / sumOfSample);
        }
    }

    
    // normalize
    for (int s = 0; s < searcherCnt; ++s) {
        double tmp = 0;
        for (int r = 1; r < regionCnt; ++r) {
            tmp += V[s][r];
        }
        for (int r = 1; r < regionCnt; ++r) {
            V[s][r] /= tmp ;
            V[s][r] = 1 - V[s][r];
        }

        for (int r = 1; r < regionCnt; ++r) {
            M[s][r] = 1 - M[s][r];
        }
    } 


    Double2 E = Double2(searcherCnt, Double1(regionCnt, 0));
    for (int s = 0; s < searcherCnt; ++s) {
        for (int r = 0; r < regionCnt; ++r) {

            E[s][r] = T[s][r] * V[s][r] * M[s][r];

        }
    }

    // sample update sampleV
    for (int r = 0; r < regionCnt; ++r) 
        for (int s = 0; s < searcherCnt; ++s)
            for (int i = 0; i < sampleCnt * 2; ++i) {
                int maxSample = 0;
                for (int j = 1; j < sampleCnt; ++j)
                    if (sample[r][maxSample].fitness < sample[r][j].fitness)
                        maxSample = j;
                
                if (sampleV[s][r][i].fitness < sample[r][maxSample].fitness)
                    sample[r][maxSample]  = sampleV[s][r][i];
            }


    // Determination
    for (int r = 0; r < regionCnt; ++r)
        tb[r]++;

    for (int s = 0; s < searcherCnt; ++s) {
        int chooseR = 0;
        int chooseE = E[s][0];
        for (int r = 1; r < regionCnt; ++r) {
            if ( chooseE < E[s][r] ) {
                chooseR = r;
                chooseE = E[s][r];
            }
        }


        searcher[s] = sample[chooseR][0];
        for (int i = 0; i < sampleCnt; ++i) {
            if ( searcher[s].fitness > sample[chooseR][i].fitness) {
                searcher[s] = sample[chooseR][i];
            }
        }

        tb[chooseR] = 1;
        ta[chooseR]++;
    }

}

void SEXX::marketingResearch() { 
    for (int r = 0; r < regionCnt; ++r)
        if (tb[r] > 1 )
            ta[r] = 1;

    for (int s = 0; s < searcherCnt; ++s)
        if (globalBest.fitness > searcher[s].fitness )
            globalBest = searcher[s];
}

double SEXX::random(int lb, int ub) { 
    return  lb + (rand() % (ub - lb + 1));
}

double SEXX::random01() { 
    return   (double) rand()/RAND_MAX ;
}
