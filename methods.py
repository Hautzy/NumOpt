import matplotlib.pyplot as plt
import numpy as np


def newton(x_init, f, f_g, f_h, epsilon=1e-6, max_iterations=100, k=10):
    x = x_init
    step_sizes = []
    pos = [np.copy(x_init)]
    gradient = 0
    print(f"starting newton method from x0={x_init}")
    for i in range(max_iterations):
        gradient = f_g(x)
        transposed_gradient = np.transpose(gradient)
        hessian = f_h(x)
        inverse_hessian = np.linalg.inv(hessian)

        direction = -inverse_hessian @ gradient
        decrement = transposed_gradient @ inverse_hessian @ gradient

        if decrement / 2 <= epsilon:
            print(f'{i}) loss: {np.linalg.norm(gradient)}')
            print(f'finished after {i} iterations at x={x}')
            return x, i + 1, step_sizes, pos

        step_size = iterate_step_length(x, f, direction, gradient)
        x += step_size * direction
        step_sizes.append(step_size)
        pos.append(np.copy(x))
        if i % k == 0:
            print(f'{i}) loss: {np.linalg.norm(gradient)}')

    print(f'{i}) loss: {np.linalg.norm(gradient)}')
    print(f'finished after {i} iterations at x={x}')
    return x, max_iterations, step_sizes, pos


def steepest_descent(x_init, f, f_g, f_h, epsilon=1e-3, max_iterations=1000, k=10):
    x = x_init
    step_sizes = []
    pos = [np.copy(x_init)]
    gradient = 0
    print(f"starting steepest descent from x0={x_init}")
    for i in range(max_iterations):
        gradient = f_g(x)
        direction = - gradient
        if np.linalg.norm(gradient) <= epsilon:
            print(f'{i}) loss: {np.linalg.norm(gradient)}')
            print(f'finished after {i} iterations at x={x}')
            return x, i, step_sizes, pos

        step_size = iterate_step_length(x, f, direction, gradient)
        x += step_size * direction
        step_sizes.append(step_size)
        pos.append(np.copy(x))
        if i % k == 0:
            print(f'{i}) loss: {np.linalg.norm(gradient)}')
    print(f'{i}) loss: {np.linalg.norm(gradient)}')
    print(f'finished after {i} iterations at x={x}')
    return x, max_iterations, step_sizes, pos


def quasi_newton(x_init, f, f_g, f_h, epsilon=1e-3, max_iterations=1000, k=10):
    x = x_init
    step_sizes = []
    pos = [np.copy(x_init)]
    grad = f_g(x)
    I = np.identity(len(x))
    H = np.copy(I)
    loss = np.linalg.norm(grad)
    i = 0

    while loss > epsilon:
        if i >= max_iterations:
            break
        p = -1 * H @ grad
        alpha = iterate_step_length(x, f, p, grad)
        x_next = x + alpha * p
        grad_next = f_g(x_next)

        s = x_next - x  # 6.5
        y = grad_next - grad  # 6.5
        r = 1 / (y.T @ s)  # 6.14
        H_next = (I - r * s @ y.T) @ H @ (I - r * y @ s.T) + r * s @ s.T  # 6.17

        loss = np.linalg.norm(grad_next)

        if i % k == 0:
            print(f'{i}) loss: {loss}')

        x = x_next
        grad = grad_next
        H = H_next

        step_sizes.append(alpha)
        pos.append(np.copy(x))

        i += 1

    print(f'{i}) loss: {loss}')
    print(f'finished after {i} iterations at x={x}')
    return x, max_iterations, step_sizes, pos


def iterate_step_length(x, f, direction, gradient, rho=0.49, beta=0.99):
    step_size = 1
    while f(x + step_size * direction) > f(x) + rho * step_size * np.inner(gradient, direction):
        step_size *= beta
    return step_size


def plot_step_size(f, step_sizes, pos, minimizers, title, min_x=-2, max_x=2, min_y=-2, max_y=2, num=250):
    '''plt.figure(figsize=(16, 9))
    plt.plot(range(len(step_sizes)), step_sizes)
    plt.title(f'{title} - step size')
    plt.xlabel('iterations')
    plt.ylabel('step-size')
    plt.show()'''

    x = np.linspace(min_x, max_x, num)
    y = np.linspace(min_y, max_y, num)
    X, Y = np.meshgrid(x, y)
    Z = f(np.array([X, Y]))
    plt.figure(figsize=(16, 9))
    plt.contour(X, Y, Z, 50, cmap='jet')

    x1 = [x[0] for x in pos]
    x2 = [x[1] for x in pos]
    plt.plot(x1, x2, color='b', alpha=0.25)
    plt.scatter(x1, x2, color='y', alpha=0.5, label='steps')
    plt.scatter(minimizers[0], minimizers[1], color='r', marker='o', label='minimizer')
    plt.scatter(pos[0][0], pos[0][1], color='b', marker='*', label='start-point')
    plt.scatter(pos[-1][0], pos[-1][1], color='g', marker='*', label='my-solution')
    plt.title(f'{title} / my-solution:{pos[-1]}')
    plt.legend()
    plt.show()

    print(minimizers)
    print('----------------------------')


def check_convergence(pos, grad, method, k=10):
    print('Check convergence')
    pos = np.array([pos[i] for i in range(len(pos)) if i % k == 0])
    data = np.zeros(shape=(len(pos), len(pos[0])))
    for i in range(len(pos)):
        data[i] = grad(pos[i])
    data = data.T

    x = range(len(pos))
    plt.figure(figsize=(16, 9))
    for i in range(len(data)):
        plt.plot(x, data[i], label=f'{i}th-gradient')
    plt.legend()
    plt.title(f'gradient per dimension (only every {k}th step) - {method}')
    plt.xlabel('iterations')
    plt.ylabel('gradient')
    plt.show()
