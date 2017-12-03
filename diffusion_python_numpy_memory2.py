#!/usr/bin/env python3

import time
import numpy as np

grid_shape = (512, 512)


def roll_add(rollee, shift, axis, out):
    if shift == 1 and axis == 0:
        out[1:, :] += rollee[:-1, :]
        out[0, :] += rollee[-1, :]
    elif shift == -1 and axis == 0:
        out[:-1, :] += rollee[1:, :]
        out[-1, :] += rollee[0, :]
    elif shift == 1 and axis == 1:
        out[:, 1:] += rollee[:, :-1]
        out[:, 0] += rollee[:, -1]
    elif shift == -1 and axis == 1:
        out[:, :-1] += rollee[:, 1:]
        out[:, -1] += rollee[:, 0]


def test_roll_add():
    rollee = np.asarray([[1,2],[3,4]])
    for shift in (-1, 1):
        for axis in (0, 1):
            out = np.asarray([[5,6],[7,8]])
            expected_result = out + np.roll(rollee, shift, axis)
            roll_add(rollee, shift, axis, out)
            assert np.all(expected_result == out)


def laplacian(grid, out):
    np.copyto(out, grid)
    out *= -4
    roll_add(grid, +1, 0, out)
    roll_add(grid, -1, 0, out)
    roll_add(grid, +1, 1, out)
    roll_add(grid, -1, 1, out)


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
    test_roll_add()
    print(run_experiment(500))
