import numpy as np
import scipy as sp
from scipy import fftpack

if __name__ == '__main__':
    data = np.random.random(size=(32, 32, 3))
    dct = fftpack.dct(data, n=4, axis=1, norm='ortho')
    reshaped = dct.transpose((0, 2, 1)).reshape((32, -1))
    print(reshaped[0], reshaped.shape)

    dct2 = fftpack.dct(data.transpose(0, 2, 1), n=4, axis=-1, norm='ortho')
    reshaped2 = dct2.reshape((32, -1))
    print(reshaped2[0], reshaped.shape)
    print('Are 2 results same ? ->', np.amin(np.equal(reshaped, reshaped2)))
