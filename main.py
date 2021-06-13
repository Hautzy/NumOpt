import numpy as np
import methods as m
import problems as p
import matplotlib.pyplot as plt

import torch

MIN = 1
MAX = 9


def apply_method(method, method_name, max_iter=100, k=10):
    for i in range(5):
        dim = 10
        x_solution = torch.randint(low=MIN, high=MAX, size=(dim, 1)).type(torch.DoubleTensor)

        A = torch.randint(low=MIN, high=MAX, size=(dim, dim)).type(torch.DoubleTensor)
        A = A @ A.T
        b = A @ x_solution

        initial_point = torch.randint(low=MIN, high=MAX, size=(dim, 1)).type(torch.DoubleTensor)

        print(f'Trying {method_name}')
        pos = method(initial_point,
                     lambda x: p.qof(x, A, b),
                     lambda x: p.qof_grad(x, A, b),
                     lambda x: p.qof_hess(x, A, b),
                     max_iterations=max_iter,
                     epsilon=0.1,
                     k=k)
        m.check_convergence(pos, lambda x: p.qof_grad(x, A, b), method_name, k)


'''
# qof examples
apply_method(m.steepest_descent, 'steepest descent', 100, 1)
apply_method(m.newton, 'newton', 100, 1)
apply_method(m.quasi_newton, 'quasi newton', 100, 1)
apply_method(m.conjugated_gradiant, 'quasi newton', 100, 1)
'''


# steepest descent
def try_steepest_descent():
    minimizer = np.array([1, 1])
    pos = m.steepest_descent(torch.DoubleTensor([-1.5, 1.5]), p.sphere, p.sphere_grad, p.sphere_hess)
    m.plot_step_size(p.sphere, pos, minimizer, 'steepest_descent', -2, 4, -2, 4)

    minimizer = np.array([[0.679366, -0.679366], [0.679366, -0.679366]])
    pos = m.steepest_descent(torch.DoubleTensor([-1.9, 1.5]), p.matyas, p.matyas_grad, p.matyas_hess)
    m.plot_step_size(p.matyas, pos, minimizer, 'steepest_descent')
    pos = m.steepest_descent(torch.DoubleTensor([0, -1.5]), p.matyas, p.matyas_grad, p.matyas_hess)
    m.plot_step_size(p.matyas, pos, minimizer, 'steepest_descent')

    minimizer = np.array([1, 3])
    pos = m.steepest_descent(torch.DoubleTensor([-1.0, 0.0]), p.booth, p.booth_grad, p.booth_hess)
    m.plot_step_size(p.booth, pos, minimizer, 'steepest_descent', -2, 5, -1, 6)

    minimizer = np.array([[3, -2.805118, -3.779310, 3.584428], [2, 3.131312, -3.283186, -1.848126]])
    pos = m.steepest_descent(torch.DoubleTensor([4.0, 4.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_step_size(p.himmel, pos, minimizer, 'steepest_descent', -5, 5, -5, 5)
    pos = m.steepest_descent(torch.DoubleTensor([-3.0, -1.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_step_size(p.himmel, pos, minimizer, 'steepest_descent', -5, 5, -5, 5)
    pos = m.steepest_descent(torch.DoubleTensor([1.0, -1.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_step_size(p.himmel, pos, minimizer, 'steepest_descent', -5, 5, -5, 5)
    pos = m.steepest_descent(torch.DoubleTensor([-1.0, 4.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_step_size(p.himmel, pos, minimizer, 'steepest_descent', -5, 5, -5, 5)

    minimizer = np.array([1, 1])
    pos = m.steepest_descent(torch.DoubleTensor([1.2, 1.2]), p.rosenbrock, p.rosenbrock_grad,
                                                    p.rosenbrock_hess, max_iterations=10000)
    m.plot_step_size(p.rosenbrock, pos, minimizer, 'steepest_descent', -2, 2, -1, 3)


# newton method
def try_newton():
    minimizer = np.array([1, 1])
    pos = m.newton(torch.DoubleTensor([-1.5, 1.5]), p.sphere, p.sphere_grad, p.sphere_hess)
    m.plot_step_size(p.sphere, pos, minimizer, 'newton', -2, 4, -2, 4)

    minimizer = np.array([[0.679366, -0.679366], [0.679366, -0.679366]])
    pos = m.newton(torch.DoubleTensor([-1.5, 0.5]), p.matyas, p.matyas_grad, p.matyas_hess)
    m.plot_step_size(p.matyas, pos, minimizer, 'newton')
    pos = m.newton(torch.DoubleTensor([1.0, 1.5]), p.matyas, p.matyas_grad, p.matyas_hess)
    m.plot_step_size(p.matyas, pos, minimizer, 'newton')

    minimizer = np.array([1, 3])
    pos = m.newton(torch.DoubleTensor([-1.0, 0.0]), p.booth, p.booth_grad, p.booth_hess)
    m.plot_step_size(p.booth, pos, minimizer, 'newton', -2, 5, -1, 6)

    minimizer = np.array([[3, -2.805118, -3.779310, 3.584428], [2, 3.131312, -3.283186, -1.848126]])
    pos = m.newton(torch.DoubleTensor([4.0, 4.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_step_size(p.himmel, pos, minimizer, 'newton', -5, 5, -5, 5)
    pos = m.newton(torch.DoubleTensor([-3.0, -4.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_step_size(p.himmel, pos, minimizer, 'newton', -5, 5, -5, 5)
    pos = m.newton(torch.DoubleTensor([4.0, -4.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_step_size(p.himmel, pos, minimizer, 'newton', -5, 5, -5, 5)
    pos = m.newton(torch.DoubleTensor([-4.0, 4.5]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_step_size(p.himmel, pos, minimizer, 'newton', -5, 5, -5, 5)

    minimizer = np.array([1, 1])
    pos = m.newton(torch.DoubleTensor([-1.5, 1.2]), p.rosenbrock, p.rosenbrock_grad, p.rosenbrock_hess)
    m.plot_step_size(p.rosenbrock, pos, minimizer, 'newton', -2, 2, -1, 3)


# quasi-newton method
def try_quasi_newton_method():
    minimizer = np.array([1, 1])
    pos = m.quasi_newton(torch.DoubleTensor([-1.5, 1.5]), p.sphere, p.sphere_grad, p.sphere_hess)
    m.plot_step_size(p.sphere, pos, minimizer, 'quasi newton', -2, 4, -2, 4)

    minimizer = np.array([[0.679366, -0.679366], [0.679366, -0.679366]])
    pos = m.quasi_newton(torch.DoubleTensor([-1.9, 1.5]), p.matyas, p.matyas_grad, p.matyas_hess)
    m.plot_step_size(p.matyas, pos, minimizer, 'quasi newton')
    pos = m.quasi_newton(torch.DoubleTensor([1.5, 1.5]), p.matyas, p.matyas_grad, p.matyas_hess)
    m.plot_step_size(p.matyas, pos, minimizer, 'quasi newton')

    minimizer = np.array([1, 3])
    pos = m.quasi_newton(torch.DoubleTensor([-1.0, 0.0]), p.booth, p.booth_grad, p.booth_hess, max_iterations=1000)
    m.plot_step_size(p.booth, pos, minimizer, 'quasi newton', -2, 5, -1, 6)

    minimizer = np.array([[3, -2.805118, -3.779310, 3.584428], [2, 3.131312, -3.283186, -1.848126]])
    pos = m.quasi_newton(torch.DoubleTensor([4.0, 4.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_step_size(p.himmel, pos, minimizer, 'quasi newton', -5, 5, -5, 5)
    pos = m.quasi_newton(torch.DoubleTensor([-3.0, -1.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_step_size(p.himmel, pos, minimizer, 'quasi newton', -5, 5, -5, 5)
    pos = m.quasi_newton(torch.DoubleTensor([1.0, -1.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_step_size(p.himmel, pos, minimizer, 'quasi newton', -5, 5, -5, 5)
    pos = m.quasi_newton(torch.DoubleTensor([-1.0, 4.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_step_size(p.himmel, pos, minimizer, 'quasi newton', -5, 5, -5, 5)

    minimizer = np.array([1, 1])
    pos = m.quasi_newton(torch.DoubleTensor([1.2, 1.2]), p.rosenbrock, p.rosenbrock_grad,
                                                p.rosenbrock_hess, max_iterations=1000)
    m.plot_step_size(p.rosenbrock, pos, minimizer, 'quasi newton', -2, 2, -1, 3)


#try_steepest_descent()
#try_newton()
#try_quasi_newton_method()

minimizer = np.array([1, 1])
pos = m.conjugated_gradiant(torch.DoubleTensor([-1.0, -1.0]), p.sphere, p.sphere_grad, p.sphere_hess)
m.plot_step_size(p.sphere, pos, minimizer, 'conjugated gradiant', -2, 4, -2, 4)