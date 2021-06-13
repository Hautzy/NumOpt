import qof as q
import numpy as np
import methods as m
import problems as p
import matplotlib.pyplot as plt

import torch

def apply_method(method, method_name, max_iter=100, k=10):
    for i in range(1):
        dim = 10
        x_solution = torch.randint(low=1, high=9, size=(dim, 1)).type(torch.DoubleTensor)

        A = torch.randint(low=1, high=9, size=(dim, dim)).type(torch.DoubleTensor)
        A = A @ A.T
        b = A @ x_solution

        initial_point = torch.randint(low=1, high=9, size=(dim, 1)).type(torch.DoubleTensor)

        print(f'Trying {method_name}')
        x_min, it, step_sizes, pos = method(initial_point,
                                            lambda x: q.qof(x, A, b),
                                            lambda x: q.qof_grad(x, A, b),
                                            lambda x: q.qof_hess(x, A, b),
                                            max_iterations=max_iter,
                                            epsilon=0.1,
                                            k=k)
        m.check_convergence(pos, lambda x: q.qof_grad(x, A, b), method, k)

# qof examples
#apply_method(m.steepest_descent, 'steepest descent', 1000, 100)
#apply_method(m.newton, 'newton', 100, 1)
#apply_method(m.quasi_newton, 'quasi newton', 1000, 1)
apply_method(m.conjugated_gradiant, 'quasi newton', 1000, 1)

# steepest descent
def try_steepest_descent():
    minimizer = np.array([1, 1])
    x_min, it, step_sizes, pos = m.steepest_descent(np.array([-1.5, 1.5]), p.sphere, p.sphere_grad, p.sphere_hess)
    m.plot_step_size(p.sphere, step_sizes, pos, minimizer, 'steepest_descent', -2, 4, -2, 4)

    minimizer = np.array([[0.679366, -0.679366], [0.679366, -0.679366]])
    x_min, it, step_sizes, pos = m.steepest_descent(np.array([-1.9, 1.5]), p.matyas, p.matyas_grad, p.matyas_hess)
    m.plot_step_size(p.matyas, step_sizes, pos, minimizer, 'steepest_descent')
    x_min, it, step_sizes, pos = m.steepest_descent(np.array([0, -1.5]), p.matyas, p.matyas_grad, p.matyas_hess)
    m.plot_step_size(p.matyas, step_sizes, pos, minimizer, 'steepest_descent')

    minimizer = np.array([1, 3])
    x_min, it, step_sizes, pos = m.steepest_descent(np.array([-1.0, 0.0]), p.booth, p.booth_grad, p.booth_hess)
    m.plot_step_size(p.booth, step_sizes, pos, minimizer, 'steepest_descent', -2, 5, -1, 6)

    minimizer = np.array([[3, -2.805118, -3.779310, 3.584428], [2, 3.131312, -3.283186, -1.848126]])
    x_min, it, step_sizes, pos = m.steepest_descent(np.array([4.0, 4.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_step_size(p.himmel, step_sizes, pos, minimizer, 'steepest_descent', -5, 5, -5, 5)
    x_min, it, step_sizes, pos = m.steepest_descent(np.array([-3.0, -1.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_step_size(p.himmel, step_sizes, pos, minimizer, 'steepest_descent', -5, 5, -5, 5)
    x_min, it, step_sizes, pos = m.steepest_descent(np.array([1.0, -1.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_step_size(p.himmel, step_sizes, pos, minimizer, 'steepest_descent', -5, 5, -5, 5)
    x_min, it, step_sizes, pos = m.steepest_descent(np.array([-1.0, 4.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_step_size(p.himmel, step_sizes, pos, minimizer, 'steepest_descent', -5, 5, -5, 5)

    minimizer = np.array([1, 1])
    x_min, it, step_sizes, pos = m.steepest_descent(np.array([1.2, 1.2]), p.rosenbrock, p.rosenbrock_grad, p.rosenbrock_hess, max_iterations=10000)
    m.plot_step_size(p.rosenbrock, step_sizes, pos, minimizer, 'steepest_descent', -2, 2, -1, 3)

# newton method
def try_newton():
    minimizer = np.array([1, 1])
    x_min, it, step_sizes, pos = m.newton(np.array([-1.5, 1.5]), p.sphere, p.sphere_grad, p.sphere_hess)
    m.plot_step_size(p.sphere, step_sizes, pos, minimizer, 'newton', -2, 4, -2, 4)

    minimizer = np.array([[0.679366, -0.679366], [0.679366, -0.679366]])
    x_min, it, step_sizes, pos = m.newton(np.array([-1.5, 0.5]), p.matyas, p.matyas_grad, p.matyas_hess)
    m.plot_step_size(p.matyas, step_sizes, pos, minimizer, 'newton')
    x_min, it, step_sizes, pos = m.newton(np.array([1.0, 1.5]), p.matyas, p.matyas_grad, p.matyas_hess)
    m.plot_step_size(p.matyas, step_sizes, pos, minimizer, 'newton')

    minimizer = np.array([1, 3])
    x_min, it, step_sizes, pos = m.newton(np.array([-1.0, 0.0]), p.booth, p.booth_grad, p.booth_hess)
    m.plot_step_size(p.booth, step_sizes, pos, minimizer, 'newton', -2, 5, -1, 6)

    minimizer = np.array([[3, -2.805118, -3.779310, 3.584428], [2, 3.131312, -3.283186, -1.848126]])
    x_min, it, step_sizes, pos = m.newton(np.array([4.0, 4.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_step_size(p.himmel, step_sizes, pos, minimizer, 'newton', -5, 5, -5, 5)
    x_min, it, step_sizes, pos = m.newton(np.array([-4.0, -2.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_step_size(p.himmel, step_sizes, pos, minimizer, 'newton', -5, 5, -5, 5)
    x_min, it, step_sizes, pos = m.newton(np.array([4.0, -4.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_step_size(p.himmel, step_sizes, pos, minimizer, 'newton', -5, 5, -5, 5)
    x_min, it, step_sizes, pos = m.newton(np.array([-3.0, 4]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_step_size(p.himmel, step_sizes, pos, minimizer, 'newton', -5, 5, -5, 5)

    minimizer = np.array([1, 1])
    x_min, it, step_sizes, pos = m.newton(np.array([1.2, 1.2]), p.rosenbrock, p.rosenbrock_grad, p.rosenbrock_hess)
    m.plot_step_size(p.rosenbrock, step_sizes, pos, minimizer, 'newton', -2, 2, -1, 3)

# quasi-newton method
def try_quasi_newton_method():
    minimizer = np.array([1, 1])
    x_min, it, step_sizes, pos = m.quasi_newton(np.array([-1.5, 1.5]), p.sphere, p.sphere_grad, p.sphere_hess)
    m.plot_step_size(p.sphere, step_sizes, pos, minimizer, 'quasi newton', -2, 4, -2, 4)

    minimizer = np.array([[0.679366, -0.679366], [0.679366, -0.679366]])
    x_min, it, step_sizes, pos = m.quasi_newton(np.array([-1.9, 1.5]), p.matyas, p.matyas_grad, p.matyas_hess)
    m.plot_step_size(p.matyas, step_sizes, pos, minimizer, 'quasi newton')
    x_min, it, step_sizes, pos = m.quasi_newton(np.array([1.5, 1.5]), p.matyas, p.matyas_grad, p.matyas_hess)
    m.plot_step_size(p.matyas, step_sizes, pos, minimizer, 'quasi newton')

    minimizer = np.array([1, 3])
    x_min, it, step_sizes, pos = m.quasi_newton(np.array([-1.0, 0.0]), p.booth, p.booth_grad, p.booth_hess)
    m.plot_step_size(p.booth, step_sizes, pos, minimizer, 'quasi newton', -2, 5, -1, 6)

    minimizer = np.array([[3, -2.805118, -3.779310, 3.584428], [2, 3.131312, -3.283186, -1.848126]])
    x_min, it, step_sizes, pos = m.quasi_newton(np.array([4.0, 4.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_step_size(p.himmel, step_sizes, pos, minimizer, 'quasi newton', -5, 5, -5, 5)
    x_min, it, step_sizes, pos = m.quasi_newton(np.array([-3.0, -1.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_step_size(p.himmel, step_sizes, pos, minimizer, 'quasi newton', -5, 5, -5, 5)
    x_min, it, step_sizes, pos = m.quasi_newton(np.array([1.0, -1.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_step_size(p.himmel, step_sizes, pos, minimizer, 'quasi newton', -5, 5, -5, 5)
    x_min, it, step_sizes, pos = m.quasi_newton(np.array([-1.0, 4.0]), p.himmel, p.himmel_grad, p.himmel_hess)
    m.plot_step_size(p.himmel, step_sizes, pos, minimizer, 'quasi newton', -5, 5, -5, 5)

    minimizer = np.array([1, 1])
    x_min, it, step_sizes, pos = m.quasi_newton(np.array([1.2, 1.2]), p.rosenbrock, p.rosenbrock_grad, p.rosenbrock_hess, max_iterations=10000)
    m.plot_step_size(p.rosenbrock, step_sizes, pos, minimizer, 'quasi newton', -2, 2, -1, 3)


#try_quasi_newton_method()