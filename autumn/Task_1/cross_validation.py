import numpy as np
from nearest_neighbors import KNNClassifier


def kfold(n, n_folds):
    lis = []
    n_1 = n % n_folds
    n_2 = n // n_folds + 1
    n_3 = n // n_folds
    for i in range(n_1):
        lis.append([x + i * n_2 for x in range(n_2)])
    for i in range(n_folds - n_1):
        lis.append([x for x in range(n_1 * n_2 + i * n_3, n_1 * n_2 + n_3 * (i + 1))])
    l_res = []
    for i in lis:
        l_help = []
        for j in range(n):
            if j not in i:
                l_help.append(j)
        l_res.append((np.array(l_help), np.array(i)))
    return l_res


def pred_weights(y, weigh, ind):
    ar_y = np.take(y, ind)  # отклики ближайших соседей
    labels = np.unique(ar_y.astype('<U22'), axis=1)  # уникальные отклики для каждой строки
    list_sum = []
    for row in labels.T:
        list_sum.append((weigh * np.equal(ar_y.astype('int'), row[:, np.newaxis].astype('int'))).sum(axis=1))
    ar_sum = np.array(list_sum)
    ar_sum = ar_sum.T
    return np.take_along_axis(labels, ar_sum.argmax(axis=1)[:, np.newaxis], axis=1).reshape((ar_y.shape[0],))


def pred_no_weights(y, ind):
    ar_y = np.take(y, ind)  # отклики ближайших соседей
    labels = np.unique(ar_y.astype('<U22'), axis=1)
    list_sum = []
    for row in labels.T:
        list_sum.append(np.equal(ar_y.astype('int'), row[:, np.newaxis].astype('int')).sum(axis=1))
    ar_sum = np.array(list_sum)
    ar_sum = ar_sum.T
    return np.take_along_axis(labels, ar_sum.argmax(axis=1)[:, np.newaxis], axis=1).reshape((ar_y.shape[0],))


def knn_cross_val_score(X, y, k_list, score, cv, **kwargs):
    dic = dict.fromkeys(k_list)
    for k in k_list:
        dic[k] = []
    if cv is None:
        cv = kfold(X.shape[0], n_folds=3)
    if score == 'accuracy':
        for i_cv in cv:
            clf = KNNClassifier(max(k_list), **kwargs)
            X_test = np.take(X, i_cv[0], axis=0)
            y_test = np.take(y, i_cv[0], axis=0)
            clf.fit(X_test, y_test)
            neigh, ind = clf.find_kneighbors(np.take(X, i_cv[1], axis=0), return_distance=True)
            if clf.weights:
                neigh += 1e-5
                neigh = 1 / neigh  # веса
            for k in k_list:
                if clf.weights:
                    y_pred = pred_weights(y_test, neigh[:, :k], ind[:, :k])
                else:
                    y_pred = pred_no_weights(y_test, ind[:, :k])
                dic[k].append(np.equal(y_pred.astype('int'), np.take(y, i_cv[1]).astype('int')).sum(axis=0) /
                              y_pred.shape[0])
    return dic
