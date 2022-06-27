import numpy as np


def grad_finite_diff(function, w, eps=1e-8):
    """
    Возвращает численное значение градиента, подсчитанное по следующией формуле:
        result_i := (f(w + eps * e_i) - f(w)) / eps,
        где e_i - следующий вектор:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    res = []
    for i in range(w.shape[0]):
        e = np.zeros(w.shape[0])
        e[i] = 1
        res_i = (function(w + eps*e) - function(w)) / eps
        res.append(res_i)
    return np.array(res)
