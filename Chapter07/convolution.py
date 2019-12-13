import numpy as np
from scipy import signal
from timeit import default_timer as timer


def conv2d_direct(x, w):
    w = np.flip(np.flip(w, 0), 1)
    rows = x.shape[0]
    cols = x.shape[1]
    kh = w.shape[0]
    kw = w.shape[1]
    rst = np.zeros((rows-kh+1, cols-kw+1))
    for i in range(rst.shape[0]):
        for j in range(rst.shape[1]):
            tmp = 0.
            for ki in range(kh):
                for kj in range(kw):
                    tmp += x[i+ki][j+kj] * w[ki][kj]
            rst[i][j] = tmp
    return rst


def conv2d_fft(x, w):
    # return signal.fftconvolve(x, w, mode='valid')
    size = np.array(x.shape) + np.array(w.shape) - 1
    fsize = 2 ** np.ceil(np.log2(size)).astype(int)
    fslice = tuple([slice(kn-1, int(sz)-kn+1) for sz, kn in zip(size, w.shape)])
    x_fft = np.fft.fft2(x , fsize)
    w_fft = np.fft.fft2(w , fsize)
    rst = np.fft.ifft2(x_fft * w_fft)
    rst = rst[fslice].real
    return rst


def im2col(x, stride=1):
    # https://stackoverflow.com/a/30110497/3829845
    rows = x.shape[0]
    cols = x.shape[1]
    kh = w.shape[0]
    kw = w.shape[1]
    s0, s1 = x.strides
    nrows = rows-kh+1
    ncols = cols-kw+1
    shape = kh, kw, nrows, ncols
    slides = s0, s1, s0, s1
    L = kh*kw

    x_unfold = np.lib.stride_tricks.as_strided(x, shape=shape, strides=slides)
    return x_unfold.reshape(L, -1)[:,::stride]


def conv2d_gemm(x, w, stride=1):
    w = np.flip(np.flip(w, 0), 1)
    rows = x.shape[0]
    cols = x.shape[1]
    kh = w.shape[0]
    kw = w.shape[1]
    L = kh*kw

    x_unfold = im2col(x)
    y_unfold = np.matmul(x_unfold.transpose(), w.reshape((L, 1)))
    return y_unfold.reshape(rows-kh+1, cols-kw+1)


x = np.random.randn(512, 512)
w = np.random.randn(5, 5)
# x = np.ones((12, 12))
# w = np.ones((5, 5))
# x = np.arange(16).reshape((4,4))
# w = np.arange(9).reshape((3,3))

start = timer()
rst0 = signal.convolve2d(x, w, mode='valid')
end = timer()
print('Elapsed time (reference): {}'.format(end - start))

# print(rst0.shape)
# print(rst0)

start = timer()
rst1 = conv2d_direct(x, w)
end = timer()
print('Elapsed time (direct): {}'.format(end - start))

# print(rst1.shape)
# print(rst1)
error1 = np.max(np.abs(rst1 - rst0))
print('Error: {}'.format(error1))

start = timer()
rst2 = conv2d_fft(x, w)
end = timer()
print('Elapsed time (FFT): {}'.format(end - start))

# print(rst2.shape)
# print(rst2)
error2 = np.max(np.abs(rst2 - rst0))
print('Error: {}'.format(error2))

start = timer()
rst3 = conv2d_gemm(x, w)
end = timer()
print('Elapsed time (im2col): {}'.format(end - start))

# print(rst3.shape)
# print(rst3)
error3 = np.max(np.abs(rst3 - rst0))
print('Error: {}'.format(error3))

import torch

inp = torch.randn(1, 1, 512, 512)
w = torch.randn(1, 1, 5, 5)
start = timer()
inp_unf = torch.nn.functional.unfold(inp, (5, 5))
# print(inp_unf.shape)
out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
# print(out_unf.shape)
# out = torch.nn.functional.fold(out_unf, (508, 508), (1, 1))
out = out_unf.view(1, 1, 508, 508)
end = timer()
print('Elapsed time (nn.Unfold): {}'.format(end - start))
error4 = (torch.nn.functional.conv2d(inp, w) - out).abs().max()
print('Error: {}'.format(error4))
