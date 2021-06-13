import numpy as np
import torch

import methods as m
import problems as p

MIN = 1
MAX = 9


def test_all_qof_methods():
    for i in range(5):
        print(f'##############Example-{i}##############')
        dim = 10
        x_solution = torch.randint(low=MIN, high=MAX, size=(dim, 1)).type(torch.DoubleTensor)

        A = torch.randint(low=MIN, high=MAX, size=(dim, dim)).type(torch.DoubleTensor)
        A = A @ A.T
        b = A @ x_solution

        apply_method(m.steepest_descent, dim, A, b, max_iter=1000, k_p=100, k_d=10)
        print('---------------------------------------')
        apply_method(m.newton, dim, A, b)
        print('---------------------------------------')
        apply_method(m.quasi_newton, dim, A, b)
        print('---------------------------------------')
        apply_method(m.linear_conjugate_gradient, dim, A, b, max_iter=1000, k_p=100, k_d=10)


def apply_method(method, dim, A, b, max_iter=100, k_p=1, k_d=1):
    initial_point = torch.randint(low=MIN, high=MAX, size=(dim, 1)).type(torch.DoubleTensor)
    print(f'Trying {method.__name__}')
    pos = method(initial_point,
                 lambda x: p.qof(x, A, b),
                 lambda x: p.qof_grad(x, A, b),
                 lambda x: p.qof_hess(x, A, b),
                 max_iterations=max_iter,
                 epsilon=0.1,
                 k=k_p)
    m.check_convergence(pos, lambda x: p.qof_grad(x, A, b), method.__name__, k_d)



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


meta_steepest_descent = [
    [0, [-1.5, 1.5]],
    [1, [-1.9, 1.5]],
    [1, [0.0, -1.5]],
    [2, [-1.0, 0.0]],
    [3, [4.0, 4.0]],
    [3, [-3.0, -1.0]],
    [3, [1.0, -1.0]],
    [3, [-1.0, 4.0]],
    [4, [1.5, -0.5], 3000, 100]
]

meta_newton = [
    [0, [-1.5, 1.5]],
    [1, [-1.5, 0.5]],
    [1, [1.0, 1.5]],
    [2, [-1.0, 0.0]],
    [3, [4.0, 4.0]],
    [3, [-3.0, -4.0]],
    [3, [4.0, -4.0]],
    [3, [-4.0, 4.5]],
    [4, [-1.5, 1.2], 1000, 100]
]

meta_quasi_newton = [
    [0, [-1.5, 1.5], 3000, 100],
    [1, [-1.9, -1.5]],
    [1, [1.5, 1.5]],
    [2, [-1.0, 0.0], 3000, 100],
    [3, [4.0, 4.0]],
    [3, [-3.0, -1.0]],
    [3, [1.0, -1.0]],
    [3, [-1.0, 4.0]],
    [4, [1.5, 1.0], 1000, 100]
]

meta_conjugate_gradient = [
    [0, [-1.5, 1.5]],
    [1, [-1.9, -1.5]],
    [1, [1.5, 1.5]],
    [2, [-1.0, 0.0], 1000, 100],
    [3, [4.0, 4.0]],
    [3, [-3.0, -1.0]],
    [3, [1.0, -1.0]],
    [3, [-1.0, 4.0]],
    [4, [1.5, 1.0]]
]

#apply_all_methods()
#test_method(m.steepest_descent, 'steepest descent', meta_steepest_descent)
# test_method(m.newton, 'newton', meta_newton)
# test_method(m.quasi_newton, 'quasi newton', meta_quasi_newton)
# test_method(m.non_linear_conjugate_gradient, 'conjugate gradient', meta_conjugate_gradient)
