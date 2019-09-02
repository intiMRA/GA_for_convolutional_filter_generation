import numpy as np
import random
def getFashion():
    f1=[f.strip("\n").split(",") for f in open("fashionmnist/fashion-mnist_test.csv")]
    f1.pop(0)
    f2 = [f.strip("\n").split(",") for f in open("fashionmnist/fashion-mnist_train.csv")]
    f2.pop(0)
    random.seed(10)
    random.shuffle(f1)
    random.shuffle(f2)
    file1=[]
    file2=[]
    classes1={}
    for f in f1:
        c=f[0]
        if c not in classes1:
            count=0
            for ins in f1:
                if ins[0]==c:
                    count+=1
            classes1[c]=count
    classes2={}
    for f in f2:
        c=f[0]
        if c not in classes2:
            count=0
            for ins in f2:
                if ins[0]==c:
                    count+=1
            classes2[c]=count
    for c in classes1.keys():
        for i in range(classes1[c]//100):
            for idx,ins in enumerate(f1):
                if ins[0]==c:
                    file1.append(ins)
                    f1.pop(idx)
                    break
    for c in classes2.keys():
        for i in range(classes2[c] // 100):
            for idx, ins in enumerate(f2):
                if ins[0] == c:
                    file2.append(ins)
                    f2.pop(idx)
                    break
    f1=file1
    f2=file2
    print(len(f1),len(f2))
    traingX=[]
    traingY = []
    testX=[]
    testY = []

    for f in f1:
        im=[]
        row = []
        count=0
        for i in range(1,len(f)):
            if count==28:
                im.append(np.array(row).astype(float))
                row = []
                count=0
            count+=1
            row.append(f[i])
        testY.append(f[0])
        testX.append(np.array(im))

    for f in f2:
        im = []
        row = []
        count=0
        for i in range(1,len(f)):
            if count==28:
                im.append(np.array(row).astype(float))
                row = []
                count=0
            count+=1
            row.append(f[i])
        traingY.append(f[0])
        traingX.append(np.array(im))
    return traingX,traingY,testX,testY