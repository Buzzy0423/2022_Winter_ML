import imageio as iio
import requests
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    im = iio.imread("Forest-House.jpg")
    im = np.array(im)
    im = np.sum(im, axis=2) / 3
    im.shape
    im = im.reshape(436, 658, 1)
    print(im.shape)
    im = im.reshape(-1, 1)
    print(im.shape)
    N = 4

    center = np.random.randint(0, 255, N)

    for i in range(5):
        diff = abs(im - center)
        arg = np.argmin(diff, axis=1)
        new_center = []
        for num in np.unique(arg):
            t = im * (arg == num)
            new_center.append(np.sum(t) / np.sum(arg == num))
        center = np.array(new_center)

    print(center)
