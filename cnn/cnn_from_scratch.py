import numpy as np

def cnn_2d(input, stride, n_kernels, kernel_size):
    x_size = input.shape[0]
    y_size = input.shape[1]

    kernels = [[] for k in range(kernel_size)]

    feature_map = []

    for kernel in kernels:
        for y in range(0, y_size, stride):
            for x in range(0, x_size, stride):
                pixel_output = kernel * input[x:x+kernel_size y:y+kernel_size]