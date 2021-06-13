import numpy as np
import methods as m
import problems as p
import matplotlib.pyplot as plt

import torch

MIN = 1
MAX = 9


def apply_all_methods():
    for i in range(5):
        dim = 10
        x_solution = torch.randint(low=MIN, high=MAX, size=(dim, 1)).type(torch.DoubleTensor)

        A = torch.randint(low=MIN, high=MAX, size=(dim, dim)).type(torch.DoubleTensor)
        A = A @ A.T
        b = A @ x_solution

        apply_method(m.steepest_descent, dim, A, b)
        apply_method(m.newton, dim, A, b)
        apply_method(m.quasi_newton, dim, A, b)
        apply_method(m.linear_conjugate_gradient, dim, A, b)


def apply_method(method, dim, A, b, max_iter=100, k=1):
    initial_point = torch.randint(low=MIN, high=MAX, size=(dim, 1)).type(torch.DoubleTensor)
    print(f'Trying {method.__name__}')
    pos = method(initial_point,
                 lambda x: p.qof(x, A, b),
                 lambda x: p.qof_grad(x, A, b),
                 lambda x: p.qof_hess(x, A, b),
                 max_iterations=max_iter,
                 epsilon=0.1,
                 k=k)
    m.check_convergence(pos, lambda x: p.qof_grad(x, A, b), method.__name__, k)


# apply_all_methods()


# steepest descent
def try_steepest_descent():
    minimizer = np.array([1, 1])
    pos = m.steepest_descent(torch.DoubleTensor([-1.5, 1.5]), p.sphere, p.sphere_grad, p.sphere_hess)
    m.plot_steps(p.sphere, pos, minimizer, 'steepest_descent', -2, 4, -2, 4)

    minimizer = np.array([[0.679366, -0.679366], [0.679366, -0.679366]])
    pos = m.steepest_descent(torch.DoubleTensor([-1.9, 1.5]), p.matyas, p.matyas_grad, p.matyas_hess)
    m.plot_steps(p.matyas, pos, minimizer, 'steepest_descent')
    pos = m.steepest_descent(torch.DoubleTensor([0, -1.5]), p.matyas, p.matyas_grad, p.matyas_hess)
    m.plot_steps(p.matyas, pos, minimizer, 'steepest_descent')

    minimizer = np.array([1, 3])
    pos = m.steepest_descent(torch.DoubleTensor([-1.0, 0.0]), p.booth, p.booth_grad, p.booth_hess)
    m.plot_steps(p.booth, pos, minimizer, 'steepest_descent', -2, 5, -1, 6)

    minimizer = np.array([[3, -2.805118, -3.779310, 3.584428], [2, 3.131312, -3.283186, -1.848126]])
    pos = m.steepest_descent(torch.DoubleTensor([4.0, 4.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_steps(p.himmel, pos, minimizer, 'steepest_descent', -5, 5, -5, 5)
    pos = m.steepest_descent(torch.DoubleTensor([-3.0, -1.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_steps(p.himmel, pos, minimizer, 'steepest_descent', -5, 5, -5, 5)
    pos = m.steepest_descent(torch.DoubleTensor([1.0, -1.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_steps(p.himmel, pos, minimizer, 'steepest_descent', -5, 5, -5, 5)
    pos = m.steepest_descent(torch.DoubleTensor([-1.0, 4.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_steps(p.himmel, pos, minimizer, 'steepest_descent', -5, 5, -5, 5)

    minimizer = np.array([1, 1])
    pos = m.steepest_descent(torch.DoubleTensor([1.2, 1.2]), p.rosenbrock, p.rosenbrock_grad,
                             p.rosenbrock_hess, max_iterations=10000)
    m.plot_steps(p.rosenbrock, pos, minimizer, 'steepest_descent', -2, 2, -1, 3)


# newton method
def try_newton():
    method_name = 'newton'

    minimizer = np.array([1, 1])
    pos = m.newton(torch.DoubleTensor([-1.5, 1.5]), p.sphere, p.sphere_grad, p.sphere_hess)
    m.plot_steps(p.sphere, pos, minimizer, method_name, -2, 4, -2, 4)

    minimizer = np.array([[0.679366, -0.679366], [0.679366, -0.679366]])
    pos = m.newton(torch.DoubleTensor([-1.5, 0.5]), p.matyas, p.matyas_grad, p.matyas_hess)
    m.plot_steps(p.matyas, pos, minimizer, method_name)
    pos = m.newton(torch.DoubleTensor([1.0, 1.5]), p.matyas, p.matyas_grad, p.matyas_hess)
    m.plot_steps(p.matyas, pos, minimizer, method_name)

    minimizer = np.array([1, 3])
    pos = m.newton(torch.DoubleTensor([-1.0, 0.0]), p.booth, p.booth_grad, p.booth_hess)
    m.plot_steps(p.booth, pos, minimizer, method_name, -2, 5, -1, 6)

    minimizer = np.array([[3, -2.805118, -3.779310, 3.584428], [2, 3.131312, -3.283186, -1.848126]])
    pos = m.newton(torch.DoubleTensor([4.0, 4.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_steps(p.himmel, pos, minimizer, method_name, -5, 5, -5, 5)
    pos = m.newton(torch.DoubleTensor([-3.0, -4.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_steps(p.himmel, pos, minimizer, 'newton', -5, 5, -5, 5)
    pos = m.newton(torch.DoubleTensor([4.0, -4.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_steps(p.himmel, pos, minimizer, 'newton', -5, 5, -5, 5)
    pos = m.newton(torch.DoubleTensor([-4.0, 4.5]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_steps(p.himmel, pos, minimizer, 'newton', -5, 5, -5, 5)

    minimizer = np.array([1, 1])
    pos = m.newton(torch.DoubleTensor([-1.5, 1.2]), p.rosenbrock, p.rosenbrock_grad, p.rosenbrock_hess)
    m.plot_steps(p.rosenbrock, pos, minimizer, 'newton', -2, 2, -1, 3)


# quasi-newton method
def try_quasi_newton_method():
    method_name = 'quasi newton'

    minimizer = np.array([1, 1])
    pos = m.quasi_newton(torch.DoubleTensor([-1.5, 1.5]), p.sphere, p.sphere_grad, p.sphere_hess)
    m.plot_steps(p.sphere, pos, minimizer, method_name, -2, 4, -2, 4)

    minimizer = np.array([[0.679366, -0.679366], [0.679366, -0.679366]])
    pos = m.quasi_newton(torch.DoubleTensor([-1.9, 1.5]), p.matyas, p.matyas_grad, p.matyas_hess)
    m.plot_steps(p.matyas, pos, minimizer, method_name)
    pos = m.quasi_newton(torch.DoubleTensor([1.5, 1.5]), p.matyas, p.matyas_grad, p.matyas_hess)
    m.plot_steps(p.matyas, pos, minimizer, method_name)

    minimizer = np.array([1, 3])
    pos = m.quasi_newton(torch.DoubleTensor([-1.0, 0.0]), p.booth, p.booth_grad, p.booth_hess, max_iterations=1000,
                         k=100)
    m.plot_steps(p.booth, pos, minimizer, method_name, -2, 5, -1, 6)

    minimizer = np.array([[3, -2.805118, -3.779310, 3.584428], [2, 3.131312, -3.283186, -1.848126]])
    pos = m.quasi_newton(torch.DoubleTensor([4.0, 4.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_steps(p.himmel, pos, minimizer, method_name, -5, 5, -5, 5)
    pos = m.quasi_newton(torch.DoubleTensor([-3.0, -1.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_steps(p.himmel, pos, minimizer, method_name, -5, 5, -5, 5)
    pos = m.quasi_newton(torch.DoubleTensor([1.0, -1.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_steps(p.himmel, pos, minimizer, method_name, -5, 5, -5, 5)
    pos = m.quasi_newton(torch.DoubleTensor([-1.0, 4.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_steps(p.himmel, pos, minimizer, method_name, -5, 5, -5, 5)

    minimizer = np.array([1, 1])
    pos = m.quasi_newton(torch.DoubleTensor([1.5, 1.0]), p.rosenbrock, p.rosenbrock_grad,
                         p.rosenbrock_hess, max_iterations=1000, k=100)
    m.plot_steps(p.rosenbrock, pos, minimizer, method_name, -2, 2, -1, 3)


problems = [
    [p.sphere, p.sphere_grad, p.sphere_hess, [1, 1], -2, 4, -2, 4],
    [p.matyas, p.matyas_grad, p.matyas_hess, [[0.679366, -0.679366], [0.679366, -0.679366]], -2, 2, -2, 2],
    [p.booth, p.booth_grad, p.booth_hess, [1, 3], -2, 5, -1, 6],
    [p.himmel, p.himmel_grad, p.himmel_hess, [[3, -2.805118, -3.779310, 3.584428], [2, 3.131312, -3.283186, -1.848126]],
     -5, 5, -5, 5],
    [p.rosenbrock, p.rosenbrock_grad, p.rosenbrock_hess, [1, 1], -2, 2, -1, 3]
]


def test_method(method, method_name, meta):
    for i in range(len(meta)):
        j = meta[i][0]
        max_iterations = 100
        k = 10
        if len(meta[i]) > 2:
            max_iterations = meta[i][2]
            k = meta[i][3]
        pos = method(torch.DoubleTensor(meta[i][1]), problems[j][0], problems[j][1], problems[j][2],
                     max_iterations=max_iterations, k=k)
        m.plot_steps(problems[j][0], pos, np.array(problems[j][3]), method_name, problems[j][4], problems[j][5],
                     problems[j][6], problems[j][7])




test_method(m.non_linear_conjugate_gradient, 'conjugate gradient', [
    [0, [-1.5, 1.5]],
    [1, [-1.9, -1.5]],
    [1, [1.5, 1.5]],
    [2, [-1.0, 0.0], 1000, 100],
    [3, [4.0, 4.0]],
    [3, [-3.0, -1.0]],
    [3, [1.0, -1.0]],
    [3, [-1.0, 4.0]],
    [4, [1.5, 1.0]]
])
