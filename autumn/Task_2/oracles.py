class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """

    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        from scipy.special import expit
        import numpy as np
        deg = self.X @ w
        deg = deg * self.y
        Q = expit(deg)
        Q = -np.log(Q)
        Q = Q.sum() / self.X.shape[0] +\
            (self.l2_coef / 2)*(np.linalg.norm(w)**2)
        return Q
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        from scipy.special import expit
        deg = self.X @ w
        deg = deg * self.y
        Q = expit(deg)
        Q = ((1 - Q) * self.y) @ self.X
        Q = -Q / self.X.shape[0]
        return Q + self.l2_coef * w
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.
    Оракул должен поддерживать l2 регуляризацию.
    """

    def __init__(self, l2_coef):
        """
        Задание параметров оракула.
        l2_coef - коэффициент l2 регуляризации
        """
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w - одномерный numpy array
        """
        self.X = X
        self.y = y
        return super().func(w)

    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w - одномерный numpy array
        """
        self.X = X
        self.y = y
        return super().grad(w)
