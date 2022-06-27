from distances import euclidean_distance
from distances import cosine_distance
import numpy as np


class KNNClassifier:
    def __init__(self, k, strategy, metric, weights, test_block_size):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.size = test_block_size

    def fit(self, X, y):
        self.X = X
        self.y = y

    def find_kneighbors(self, X, return_distance):
        if self.strategy == 'my_own':
            if self.metric == 'euclidean':
                matr_dist = euclidean_distance(X, self.X)
            if self.metric == 'cosine':
                matr_dist = cosine_distance(X, self.X)
            matr_2 = np.argsort(matr_dist, axis=1)[:, :self.k]
            if return_distance:
                matr_1 = np.take_along_axis(matr_dist, matr_2, axis=1)[:, :self.k]
                return (matr_1, matr_2)
            else:
                return matr_2
        else:
            from sklearn.neighbors import NearestNeighbors
            knn_clf = NearestNeighbors(n_neighbors=self.k, algorithm=self.strategy, metric=self.metric)
            knn_clf.fit(self.X, self.y)
            return knn_clf.kneighbors(X=X, n_neighbors=self.k, return_distance=return_distance)

    def predict(self, X):
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

        if self.weights:
            dist, ind = self.find_kneighbors(X, return_distance=True)
            return pred_weights(self.y, dict, ind)
        else:
            ind = self.find_kneighbors(X, return_distance=False)
            return pred_no_weights(self.y, ind)
