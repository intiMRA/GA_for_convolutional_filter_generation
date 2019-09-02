import numpy as np
import random as rd
from numpy.lib.stride_tricks import as_strided
pooling=["min","max","mean"]
def convolve(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    sub_arrays = as_strided(image, tuple(np.subtract(image.shape, filter.shape) + 1) + filter.shape, image.strides * 2)
    return np.tensordot(sub_arrays, filter, ((2, 3), (0, 1)))

def gen_filter(size: (int, int)) -> np.ndarray:
    filter = []
    for f in range(size[0]):
        filter.append([])
        for f1 in range(size[1]):
            filter[-1].append(rd.uniform(-5, 5))
    filter = np.array(filter)
    if np.sum(filter) == 0:
        return filter
    return filter / np.sum(filter)

def generate(icls, numberOfFilters):
    global pooling
    size = rd.randint(3, 7)
    individual = [{"splits":3,"filter": gen_filter((size, size)), "pool": pooling[rd.randint(0, len(pooling) - 1)]} for i in
                  range(numberOfFilters)]
    return icls(individual)

def convolve2(image, F):

    image_height = image.shape[0]
    image_width = image.shape[1]


    F_height = F.shape[0]
    F_width = F.shape[1]

    H = (F_height - 1) // 2
    W = (F_width - 1) // 2

    out = np.zeros((image_height, image_width))

    for i in np.arange(H, image_height - H):
        for j in np.arange(W, image_width - W):
            sum = 0

            for k in np.arange(-H, H + 1):
                for l in np.arange(-W, W + 1):

                    a = image[i + k, j + l]
                    w = F[H + k, W + l]
                    sum += (w * a)
            out[i, j] = sum

    return out
