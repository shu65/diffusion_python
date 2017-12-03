#!/usr/bin/env python3

import time
import numpy as np

grid_shape = (512, 512)


@profile
def laplacian(grid, out):
    np.copyto(out, grid)
    out *= -4
    out += np.roll(grid, +1, 0)
    out += np.roll(grid, -1, 0)
    out += np.roll(grid, +1, 1)
    out += np.roll(grid, -1, 1)

@profile
def evolve(grid, dt, out, D=1.0):
    laplacian(grid, out)
    out *= dt * D
    out += grid


def run_experiment(num_iterations):
    # setting up initial conditions
    xmax, ymax = grid_shape
    grid = np.zeros(grid_shape)
    next_grid = np.zeros(grid_shape)

    # initialization assumes that xmax <= ymax
    block_low = int(xmax * .4)
    block_high = int(xmax * .5)
    for i in range(block_low, block_high):
        for j in range(block_low, block_high):
            grid[i][j] = 0.005

    start = time.time()
    for i in range(num_iterations):
        evolve(grid, 0.1, next_grid)
        next_grid, grid = next_grid, grid
    return time.time() - start

if __name__ == "__main__":
    print(run_experiment(500))
