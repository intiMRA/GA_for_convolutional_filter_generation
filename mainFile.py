import Data
from FilterGenerator import featureGenerator as GEN

def main():
    trainingX,trainingY,testX,testY=Data.getFashion()
    a=GEN(list(trainingX),list(trainingY),ngens=50,populationSize=20)
    a.fit()
    print(a.predict(testX,testY))
main()


