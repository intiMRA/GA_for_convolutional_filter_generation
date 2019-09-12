import Data
from FilterGenerator import featureGenerator as GEN

def main():
    trainingX,trainingY,testX,testY=Data.getFaces()
    a=GEN(list(trainingX),list(trainingY),ngens=100,populationSize=20,numberOfFilters=500)
    a.fit()
    print(a.predict(testX,testY))
    a.showAll("download.jpg","out")
main()


