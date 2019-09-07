import os
import random as rd
import cv2

def getFaces() -> ([int], [int], [int], [int]):
    rd.seed(10)
    testX = []
    testY = []
    files = os.listdir("jaffe")
    all = []
    for f in files:
        image=cv2.imread("jaffe/" + f, cv2.IMREAD_GRAYSCALE)
        all.append([image,f.split(".")[1][:-1]])
    rd.shuffle(all)
    classes = {}
    cn=0
    for d in all:
        if d[-1] not in classes:

            classes[d[-1]] = [1,cn]
            cn+=1
        else:
            classes[d[-1]][0] += 1

    for c in classes.keys():
        count = classes[c][0] // 3
        moved = 0
        while moved < count:
            for i in range(len(all)):
                if all[i][-1] == c:
                    testX.append(all[i][0])
                    testY.append(classes[c][1])
                    all.pop(i)
                    moved += 1
                    break
    traingY=[]
    traingX=[]
    for i in range(len(all)):
        traingX.append(all[i][0])
        traingY.append(classes[all[i][-1]][1])
    return traingX,traingY,testX,testY