import numpy as np
# ################################################################################
# Gradient descent or Cauchy method
# First order method
def gradient_descent(max_iters, threshold, XY_init, func, grad_func, learning_rate=0.05):
    X, Y = XY_init
    w = np.array([X, Y])
    w_history = X, Y
    f_history = func(X, Y, extra_param)
    delta_w = np.zeros(XY_init.shape)
    i = 0
    # start diff |f2 - f1| for stop criteria
    diff_f = 1.0e10
    eps_history_f = np.array([0.0])
    eps_history_xy = np.array([0.0 , 0.0])

    while i < max_iters and diff_f >= threshold:
        delta_w = -learning_rate * grad_func(w[0], w[1], extra_param)
        w = w + delta_w
        # store the history of w and f
        w_history = np.vstack((w_history, w))
        f_history = np.vstack((f_history, func(w[0], w[1], extra_param)))
        i += 1
        diff_f = np.absolute(f_history[-1] - f_history[-2])
        diff_xy = np.absolute(w_history[-1] - w_history[-2])

        eps_history_f = np.vstack((eps_history_f, diff_f))
        eps_history_xy = np.vstack((eps_history_xy, diff_xy))

    return w_history, f_history, eps_history_f, eps_history_xy
