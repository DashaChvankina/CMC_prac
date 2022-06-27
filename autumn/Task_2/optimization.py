from oracles import BinaryLogistic


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
        self, loss_function, step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        step_alpha - float, параметр выбора шага из текста задания
        step_beta- float, параметр выбора шага из текста задания
        tolerance - точность, по достижении которой,
                    необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю
            разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход
        max_iter - максимальное число итераций
        **kwargs - аргументы, необходимые для инициализации
        """
        self.loss_function = loss_function
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.oracle = BinaryLogistic(**kwargs)

    # def fit(self, X, y, X_test, y_test, w_0=None, trace=False):
    def fit(self, X, y, w_0=None, trace=False):
        """
        Обучение метода по выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w_0 - начальное приближение в методе
        trace - переменная типа bool
        Если trace = True, то метод должен вернуть словарь history,
        содержащий информацию о поведении метода.
        Длина словаря history=количество итераций + 1
        (начальное приближение)
        history['time']: list of floats,
            содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats,
            содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        import timeit
        import numpy as np
        if w_0 is not None:
            self.w = w_0
        else:
            self.w = np.zeros(X.shape[1])
        time = [0]
        Q_old = self.oracle.func(X, y, self.w)
        func = [Q_old]
        # accuracy = [np.mean(y_test == self.predict(X_test))]
        k = 1
        start_time = timeit.default_timer()
        while k <= self.max_iter:
            n_k = self.step_alpha / (k**self.step_beta)
            self.w = self.w - n_k * self.oracle.grad(X, y, self.w)
            Q_new = self.oracle.func(X, y, self.w)
            func.append(Q_new)
            # accuracy.append(np.mean(y_test == self.predict(X_test)))
            time.append(timeit.default_timer() - start_time)
            # start_time = timeit.default_timer()
            if abs(Q_new - Q_old) < self.tolerance:
                break
            Q_old = Q_new
            k += 1
        if trace:
            # history = {'time': time, 'func': func, 'accuracy': accuracy}
            history = {'time': time, 'func': func}
            return history

    def predict(self, X):
        """
        Получение меток ответов на выборке X
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        return: одномерный numpy array с предсказаниями
        """
        import numpy as np
        return np.sign(X @ self.w)

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        return: двумерной numpy array,
            [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """
        from scipy.special import expit
        import numpy as np
        proba = np.array([expit(X @ self.w), expit(-X @ self.w)])
        return proba.T

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        return: float
        """
        return self.oracle.func(X, y, self.w)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        return: numpy array, размерность зависит от задачи
        """
        return self.oracle.grad(X, y, self.w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.w

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return (y_pred == y_test).sum().mean()


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
        self, loss_function, batch_size, step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        batch_size - размер подвыборки, по которой считается градиент
        step_alpha - float, параметр выбора шага из текста задания
        step_beta- float, параметр выбора шага из текста задания
        tolerance - точность, по достижении которой,
            необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних
            значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход
        max_iter - максимальное число итераций (эпох)
        random_seed - в начале метода fit необходимо вызвать
            np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости
            результатов на разных машинах.
        **kwargs - аргументы, необходимые для инициализации
        """
        super().__init__(loss_function, step_alpha, step_beta, tolerance,
                         max_iter, **kwargs)
        self.random_seed = random_seed
        self.batch_size = batch_size

    # def fit(self, X, y, X_test, y_test, w_0=None, trace=False, log_freq=1):
    def fit(self, X, y, w_0=None, trace=False, log_freq=1):
        """
        Обучение метода по выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w_0 - начальное приближение в методе
        Если trace = True, то метод должен вернуть словарь history,
            содержащий информацию
        о поведении метода. Если обновлять history после каждой итерации,
            метод перестанет
        превосходить в скорости метод GD. Поэтому,
            необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов
            в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} /
                {количество объектов в выборке}
        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно проиходить каждый раз,
            когда разница между двумя значениями приближённого номера эпохи
            будет превосходить log_freq.
        history['epoch_num']: list of floats,
            в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats,
            содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats,
            содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы
            разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        import numpy as np
        import scipy
        np.random.seed(self.random_seed)
        import timeit
        self.w = w_0
        Q_old = self.oracle.func(X, y, self.w)
        k = 1
        k_ind = 1
        Ind = np.random.permutation(X.shape[0])
        if trace:
            time = [0]
            func = [Q_old]
            weights_diff = [0]
            epoch_num = [0]
            w_old = w_0
            epoha_old = 0
            epoha_new = 0
            # accuracy = [np.mean(y_test == self.predict(X_test))]
            start_time = timeit.default_timer()
        if isinstance(X, scipy.sparse.coo.coo_matrix):
            X = X.tocsr()
        while k <= self.max_iter:
            epoha_new += self.batch_size
            if (k_ind - 1)*self.batch_size >= X.shape[0]:
                Ind = np.random.permutation(X.shape[0])
                k_ind = 1
            x_new = X[Ind[(k_ind - 1)*self.batch_size: k_ind*self.batch_size]]
            y_new = y[Ind[(k_ind - 1)*self.batch_size: k_ind*self.batch_size]]
            n_k = self.step_alpha / (k ** self.step_beta)
            self.w = self.w - n_k * self.oracle.grad(x_new, y_new, self.w)
            eps = self.oracle.func(x_new, y_new, self.w)
            Q_new = 0.1 * eps + 0.9 * Q_old
            if abs(Q_new - Q_old) < self.tolerance:
                break
            Q_old = Q_new
            if trace:
                if (epoha_new - epoha_old) / X.shape[0] > log_freq:
                    epoch_num.append(epoha_new / X.shape[0])
                    epoha_old = epoha_new
                    time.append(timeit.default_timer() - start_time)
                    weights_diff.append((w_old - self.w) @ (w_old - self.w))
                    w_old = self.w
                    func.append(Q_new)
                    # accuracy.append(np.mean(y_test == self.predict(X_test)))
            k += 1
            k_ind += 1
        if trace:
            # accuracy.append(np.mean(y_test == self.predict(X_test)))
            epoch_num.append(epoha_new / X.shape[0])
            time.append(timeit.default_timer() - start_time)
            weights_diff.append((w_old - self.w) @ (w_old - self.w))
            func.append(Q_new)
            # history = {'time': time, 'epoch_num': epoch_num,
            #          'weights_diff': weights_diff, 'func': func, 'accuracy':accuracy}
            history = {'time': time, 'epoch_num': epoch_num,
                       'weights_diff': weights_diff, 'func': func}
            return history
