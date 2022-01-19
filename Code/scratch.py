import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    gmoon = 9.8 / 6
    t = np.linspace(1, 3, 20)
    s = 0.5 * gmoon * t ** 2 + 0.0 * np.random.randn(t.size)
    plt.plot(t, s, '.')
    plt.show()