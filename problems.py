import numpy as np


def sphere(x):
    return (x[0] - 1) ** 4 + (x[1] - 1) ** 4


def sphere_grad(x):
    return np.array([
        4 * (x[0] - 1) ** 3,
        4 * (x[1] - 1) ** 3
    ])


def sphere_hess(x):
    return np.array([
        [12 * (x[0] - 1) ** 2, 0],
        [0, 12 * (x[1] - 1) ** 2]
    ])


def matyas(x):
    return 0.26 * (x[0] ** 4 + x[1] ** 4) - 0.48 * x[0] * x[1]


def matyas_grad(x):
    return np.array([
        1.04 * x[0] ** 3 - 0.48 * x[1],
        1.04 * x[1] ** 3 - 0.48 * x[0]
    ])


def matyas_hess(x):
    return np.array([
        [3.12 * x[0] ** 2, -0.48],
        [-0.48, 3.12 * x[1] ** 2]
    ])


def booth(x):
    return (x[0] + 2 * x[1] - 7) ** 4 + (2 * x[0] + x[1] - 5) ** 4


def booth_grad(x):
    return np.array([
        8 * (2 * x[0] + x[1] - 5) ** 3 + 4 * (x[0] + 2 * x[1] - 7) ** 3,
        4 * (2 * x[0] + x[1] - 5) ** 3 + 8 * (x[0] + 2 * x[1] - 7) ** 3
    ])


def booth_hess(x):
    return np.array([
        [12 * (4 * (2 * x[0] + x[1] - 5) ** 2 + (x[0] + 2 * x[1] - 7) ** 2),
         24 * ((2 * x[0] + x[1] - 5) ** 2 + (x[0] + 2 * x[1] - 7) ** 2)],
        [24 * ((2 * x[0] + x[1] - 5) ** 2 + (x[0] + 2 * x[1] - 7) ** 2),
         12 * ((2 * x[0] + x[1] - 5) ** 2 + 4 * (x[0] + 2 * x[1] - 7) ** 2)]
    ])


def himmel(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def himmel_grad(x):
    return np.array([
        (4 * x[0] ** 3 + 4 * x[0] * x[1] - 42 * x[0] + 2 * x[1] ** 2 - 14),
        (4 * x[1] ** 3 + 4 * x[0] * x[1] - 26 * x[1] + 2 * x[0] ** 2 - 22)
    ])


def himmel_hess(x):
    return np.array([
        [4 * (x[0] ** 2 + x[1] - 11) + 8 * x[0] ** 2 + 2, 4 * x[0] + 4 * x[1]],
        [4 * x[0] + 4 * x[1], 4 * (x[0] + x[1] ** 2 - 7) + 8 * x[1] ** 2 + 2]
    ])

def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def rosenbrock_grad(x):
    return np.array([
        -400 * x[0] * x[1] + 400 * x[0] ** 3 - 2 + 2 * x[0],
        200 * (x[1] - x[0] ** 2)
    ])


def rosenbrock_hess(x):
    return np.array([
        [-400 * x[1] + 1200 * x[0] ** 2 + 2, -400 * x[0]],
        [-400 * x[0], 200]
    ])
