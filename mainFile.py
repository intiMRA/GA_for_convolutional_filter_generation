import Data
from FilterGenerator import featureGenerator as GEN

def main():
    trainingX,trainingY,testX,testY=Data.getFaces()
    a=GEN(list(trainingX),list(trainingY),ngens=100,populationSize=20,numberOfFilters=(trainingX[0].shape[0]*2)//10)
    a.fit()
    print(a.predict(testX,testY))
    a.showAll()
main()


