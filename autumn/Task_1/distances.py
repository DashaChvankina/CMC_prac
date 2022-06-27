def euclidean_distance(X, Y):
    import numpy as np
    X_norm = np.linalg.norm(X, ord=2, axis=1)**2
    Y_norm = np.linalg.norm(Y, ord=2, axis=1)**2
    res = X_norm[:, np.newaxis] + Y_norm
    res = res - 2*np.dot(X, Y[:, :, np.newaxis]).sum(axis=2)
    res[res < 0] = 0
    return res**(0.5)


def cosine_distance(X, Y):
    import numpy as np
    X_norm = np.linalg.norm(X, ord=2, axis=1)
    Y_norm = np.linalg.norm(Y, ord=2, axis=1)
    res = np.dot(X, Y[:, :, np.newaxis]).sum(axis=2)/X_norm[:, np.newaxis]
    res /= Y_norm[np.newaxis, :]
    return 1 - res


if __name__ == "__main__":
    import numpy as np
    Y = np.array([
        [0, 0, 1],
        [2, 0, 3],
        [1, 0, 0]
    ])
    print(euclidean_distance(Y, Y))
