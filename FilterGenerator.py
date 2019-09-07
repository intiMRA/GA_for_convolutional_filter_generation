import random as rd
from deap import tools
from deap import creator, base,algorithms
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from helper import generate
from helper import convolve as convolve
pooling=["min","max","mean"]
import cv2



def selection(individuals, k, tournsize, fit_attr="fitness",icls=None,numberOfFilters=20):
    best = individuals[0]
    for ind in individuals:
        if best.fitness.values[0] < ind.fitness.values[0]:
            best = ind
    newGen = tools.selTournament(individuals, k - 1, tournsize, fit_attr=fit_attr)
    if best in newGen:
        newGen.append(generate(icls,numberOfFilters))
    else:
        newGen.append(best)
    return newGen

class featureGenerator:




    def fitness(self,individual,trainingX,trainingY,fitnessX,fitnessY,classifier,kfolds=-1):
        if len(fitnessX)==0:
            fitnessX=trainingX[:len(trainingX)//2]
            fitnessY = trainingY[:len(trainingY) // 2]
            trainingX=trainingX[len(trainingX)//2:]
            trainingY = trainingY[len(trainingY) // 2:]
        if kfolds>0:
            return
        trainingFeat = np.array([self.get_feature_values(individual, i) for i in trainingX])
        fitnessFeatures = np.array([self.get_feature_values(individual, i) for i in fitnessX])

        eval = classifier.fit(trainingFeat, trainingY)
        pre = eval.predict(fitnessFeatures)
        fit = 0
        for p, tv in zip(pre, fitnessY):
            if p == tv:
                fit += 1
        fit /= len(fitnessY)
        return fit,

    def splitImage(self,image,pieces):
        splits=[]
        h=image.shape[0]//pieces
        w=image.shape[1]//pieces
        for i in range(pieces):
            for j in range(pieces):
                splits.append(image[h*i:min(h*(i+1),image.shape[0]),w*j:min(w*(j+1),image.shape[1])])
        return splits
    def get_feature_values(self,idividual, image) -> np.ndarray:
        ar=[]
        for feature_filter in idividual:
            if pooling == "min":
                conv=convolve(image, feature_filter["filter"])
                splits=self.splitImage(conv,feature_filter["splits"])
                for s in splits:
                    n=np.min(s).astype(np.float)
                    ar.append(n)
                #return np.array([np.min(convolve(data[idx], feature_filter)) for idx in range(data.shape[0])])
            elif pooling == "max":
                conv=convolve(image, feature_filter["filter"])
                splits = self.splitImage(conv, feature_filter["splits"])
                for s in splits:
                    n=np.max(s).astype(np.float)
                    ar.append(n)
            else:
                conv = convolve(image, feature_filter["filter"])
                splits = self.splitImage(conv, feature_filter["splits"])
                for s in splits:
                    n=np.mean(s).astype(np.float)
                    ar.append(n)
        return np.array(ar)


    def cross(self,individual1,individual2):
        for i in range(len(individual1)):
            if (rd.uniform(0, 1) < 0.5):
                f1 = individual1[i]["filter"]
                f2 = individual2[i]["filter"]
                row = rd.randint(0, 2)
                if f1.shape == f2.shape:
                    f1[row] = individual2[i]["filter"][row]
                    f2[row] = individual1[i]["filter"][row]
                    individual1[i]["filter"] = f2
                    individual2[i]["filter"] = f1
                else:
                    for f in range(min(f1.shape[0], f2.shape[0])):
                        individual1[i]["filter"][row] = f2[row][f]
                        individual2[i]["filter"][row] = f1[row][f]
        return individual1, individual2

    def mutation(self,individual):
        for i in range(len(individual)):
            if (rd.uniform(0, 1) < 0.2):
                individual[i]["pool"] = pooling[rd.randint(0, len(pooling) - 1)]
            if (rd.uniform(0, 1) < 0.5):
                individual[i]["filter"][rd.randint(0, 2)][rd.randint(0, 2)] = rd.uniform(-5, 5)
                if np.sum(individual[i]["filter"]) != 0:
                    individual[i]["filter"] = individual[i]["filter"] / np.sum(individual[i]["filter"])
        return individual,

    def fit(self):
        return algorithms.eaSimple(self.pop, self.toolbox,
                                   cxpb=self.crossoverRate, mutpb=self.mutationRate,
                                   ngen=self.ngens, stats=self.mstats,
                                   halloffame=self.hof,verbose=True)
    def predict(self,testX,testY,classifier=KNN(n_neighbors=3,n_jobs=-1),filename="individual.txt"):

        f=open(filename+".txt",mode="w")
        for fl in self.hof[0]:
            f.write(fl["pool"]+"\n")
            for v in fl["filter"]:
                for i in range(len(v)-1):
                    f.write(str(v[i])+",")
                f.write(str(v[-1])+"\n")
            f.write("\n")
        f.close()
        print("Evaluating")
        trainingFeat=np.array([self.get_feature_values(self.hof[0],i) for i in self.trainingX])
        testFeatures=np.array([self.get_feature_values(self.hof[0],i) for i in np.array(testX)])
        eval = classifier.fit(trainingFeat, self.trainingY)
        pre = eval.predict(testFeatures)
        fit = 0
        for p, tv in zip(pre, testY):
            if p == tv:
                fit += 1
        fit /= len(testY)
        tst=open("test_features.csv",mode="w")
        trn = open("train_features.csv", mode="w")
        for x,l in zip(trainingFeat,self.trainingY):
            for f in x:
                trn.write(str(round(f,4))+",")
            trn.write(str(l)+"\n")
        for x,l in zip(testFeatures,testY):
            for f in x:
                tst.write(str(round(f,4))+",")
            tst.write(str(l)+"\n")
        tst.close()
        trn.close()
        return fit

    def splitData(self,trainX,trainY):
        classes = {}
        FX=[]
        FY=[]
        for d in trainY:
            if d not in classes:

                classes[d] = 1
            else:
                classes[d] += 1

        for c in classes.keys():
            count = classes[c] // 3
            moved = 0
            while moved < count:
                for i in range(len(trainX)):
                    if trainY[i] == c:
                        FX.append(np.array(trainX[i]))
                        FY.append(np.array(trainY[i]))
                        trainY.pop(i)
                        trainX.pop(i)
                        moved += 1
                        break
        return np.array(trainX),np.array(trainY),np.array(FX),np.array(FY)

    def showAll(self):
        best = self.hof[0]
        image=cv2.imread("download.jpg" , cv2.IMREAD_GRAYSCALE)
        for i, f in enumerate(best):
            print("COOL")
            img = convolve(image, f["filter"])
            cv2.imwrite('out' + str(i) + ".jpeg", img)

    def __init__(self,trainingX,trainingY,classifier=KNN(n_neighbors=3,n_jobs=-1),
                 populationSize=100,mutationRate=0.3,crossoverRate=0.7,
                 kfodls=-1,ngens=100,numberOfFilters=20):
        creator.create("FitnessMin", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin,
                       __eq__=lambda self, other: np.array_equal(self[0], other[0]))
        self.mutationRate=mutationRate
        self.crossoverRate=crossoverRate
        self.ngens=ngens
        self.numberOfFilters=numberOfFilters
        self.trainingX=np.array(trainingX)
        self.trainingY=np.array(trainingY)
        toolbox = base.Toolbox()

        toolbox.register("individual",generate, icls=creator.Individual,numberOfFilters=numberOfFilters)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        TX,TY,FX,FY=self.splitData(trainingX,trainingY)
        toolbox.register("evaluate", self.fitness, trainingX=TX, trainingY=TY,fitnessX=FX,fitnessY=FY,classifier=classifier)

        toolbox.register("mate", self.cross)
        toolbox.register("mutate", self.mutation)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)

        mstats = tools.MultiStatistics(fitness=stats_fit)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        mstats.register("av", np.average)

        toolbox.register("select", selection, tournsize=3,icls=creator.Individual,numberOfFilters=numberOfFilters)
        self.hof = tools.HallOfFame(1)
        self.pop = toolbox.population(populationSize)
        self.mstats=mstats
        self.toolbox=toolbox
