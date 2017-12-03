#!/usr/bin/env python3

import time
import numpy as np

grid_shape = (512, 512)


def laplacian(grid):
    return np.roll(grid, +1, 0) + np.roll(grid, -1, 0) +\
        np.roll(grid, +1, 1) + np.roll(grid, -1, 1) -4 * grid


def evolve(grid, dt, D=1.0):
    return grid + dt * D * laplacian(grid)


def run_experiment(num_iterations):
    # setting up initial conditions
    xmax, ymax = grid_shape
    grid = np.zeros(grid_shape)

    # initialization assumes that xmax <= ymax
    block_low = int(xmax * .4)
    block_high = int(xmax * .5)
    for i in range(block_low, block_high):
        for j in range(block_low, block_high):
            grid[i][j] = 0.005

    start = time.time()
    for i in range(num_iterations):
        grid = evolve(grid, 0.1)
    return time.time() - start

if __name__ == "__main__":
    print(run_experiment(500))
