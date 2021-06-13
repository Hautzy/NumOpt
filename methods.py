import matplotlib.pyplot as plt
import numpy as np
import torch


def newton(x_init, f, f_g, f_h, epsilon=1e-6, max_iterations=100, k=10):
    x = x_init
    pos = [x_init.clone().detach()]
    print(f"starting newton method from x0={x_init}")
    for i in range(max_iterations):
        gradient = f_g(x)
        transposed_gradient = gradient.T
        hessian = f_h(x)
        inverse_hessian = hessian.inverse()

        direction = -inverse_hessian @ gradient
        decrement = transposed_gradient @ inverse_hessian @ gradient

        loss = torch.norm(gradient)
        if decrement / 2 <= epsilon:
            break

        step_size = iterate_step_length(x, f, direction, gradient)
        x += step_size * direction
        pos.append(x.clone().detach())
        if i % k == 0:
            print(f'{i}) loss: {loss}')

    print(f'{i}) loss: {loss}')
    print(f'finished after {i} iterations at x={x}')
    return pos


def steepest_descent(x_init, f, f_g, f_h, epsilon=1e-3, max_iterations=1000, k=10):
    x = x_init
    pos = [x_init.clone().detach()]
    print(f"starting steepest descent from x0={x_init}")
    for i in range(max_iterations):
        gradient = f_g(x)
        direction = - gradient
        loss = torch.norm(gradient)
        if loss <= epsilon:
            break

        step_size = iterate_step_length(x, f, direction, gradient)
        x += step_size * direction
        pos.append(x.clone().detach())
        if i % k == 0:
            print(f'{i}) loss: {loss}')
    print(f'{i}) loss: {loss}')
    print(f'finished after {i} iterations at x={x}')
    return pos


def quasi_newton(x_init, f, f_g, f_h, epsilon=1e-3, max_iterations=100, k=10):
    x = x_init
    pos = [x_init.clone().detach()]
    grad = f_g(x)
    I = torch.eye(len(x_init)).type(torch.DoubleTensor)
    H = torch.clone(I)
    loss = torch.norm(grad)
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

        loss = torch.norm(grad_next)
        if i % k == 0:
            print(f'{i}) loss: {loss}')

        x = x_next
        grad = grad_next
        H = H_next
        pos.append(x.clone().detach())

        i += 1

    print(f'{i}) loss: {loss}')
    print(f'finished after {i} iterations at x={x}')
    return pos


def linear_conjugated_gradient(x_init, f, f_g, f_h, epsilon=1e-3, max_iterations=1000, k=10):
    i = 0
    x = x_init
    r = f_g(x)
    p = -r
    loss = torch.norm(f_g(x))
    pos = [x_init.clone().detach()]
    A = f_h(x)

    while loss > epsilon:
        if i >= max_iterations:
            break

        alpha = (r.T @ r) / (p.T @ (A @ p))
        x_next = x + alpha * p

        r_next = r + alpha * A @ p
        beta_next = (r_next.T @ r_next) / (r.T @ r)
        p_next = -r_next + beta_next * p

        loss = torch.norm(f_g(x_next))
        if i % k == 0:
            print(f'{i}) loss: {loss}')

        x = x_next
        r = r_next
        p = p_next
        pos.append(x.clone().detach())

        i += 1

    print(f'{i}) loss: {loss}')
    print(f'finished after {i} iterations at x={x}')
    return pos


def iterate_step_length(x, f, direction, gradient, rho=0.49, beta=0.99):
    step_size = 1
    while f(x + step_size * direction) > f(x) + rho * step_size * gradient.T @ direction:
        step_size *= beta
    return step_size


def plot_step_size(f, pos, minimizers, title, min_x=-2, max_x=2, min_y=-2, max_y=2, num=250):
    x = np.linspace(min_x, max_x, num)
    y = np.linspace(min_y, max_y, num)
    X, Y = np.meshgrid(x, y)
    t = torch.from_numpy(np.array([X, Y]))
    Z = f(t)
    plt.figure(figsize=(16, 9))
    plt.contour(X, Y, Z.numpy(), 50, cmap='jet')

    x1 = [x[0] for x in pos]
    x2 = [x[1] for x in pos]
    plt.plot(x1, x2, color='b', alpha=0.25)
    plt.scatter(x1, x2, color='y', alpha=0.5, label='steps')
    plt.scatter(minimizers[0], minimizers[1], color='r', marker='o', label='minimizer')
    plt.scatter(pos[0][0], pos[0][1], color='b', marker='*', label='start-point')
    plt.scatter(pos[-1][0], pos[-1][1], color='g', marker='*', label='my-solution')
    plt.title(f'{title} / my-solution:{pos[-1].numpy()}')
    plt.legend()
    plt.show()

    print(minimizers)
    print('----------------------------')


def check_convergence(pos, grad, method, k=10):
    print('Check convergence')
    t = len(pos)
    pos = [pos[i] for i in range(len(pos)) if i % k == 0]
    data = np.zeros(shape=(len(pos), len(pos[0])))
    for i in range(len(pos)):
        data[i] = grad(pos[i]).numpy().ravel()
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
