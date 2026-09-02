import distance as dist
import hyperparameters as param
from sort import Sort
from datatypes import Data

class KNN:
    def __init__(self, train_datas:list[Data]):
        if not train_datas:
            raise ValueError('you must provide a training data')
        self.train_datas = train_datas
        self.test_data = None
        self.indices = []
        self.distances = []
    def set_test_data(self, test_data:Data):
        if test_data is None:
            raise ValueError('you must provide a test data')
        self.test_data = test_data
        self.indices = []
        self.distances = []
        self._compute_distance()
    def _compute_distance(self):
        distances = self.distances
        test_features = self.test_data.features
        for data in self.train_datas:
            train_features = data.features
            d = dist.unrooted_distance(test_features, train_features)
            distances.append(d)
        self._sort_indices()
    def _sort_indices(self):
        self.indices = list(range(len(self.train_datas)))
        Sort.quick_random(nums=self.indices, transform = lambda i: self.distances[i])
    def get_k_neighbors(self, k = param.K_NEIGHBORS):
        if self.test_data is None or not self.indices:
            raise RuntimeError('you must provide a test data and training data')
        neighbors = []
        k = min(k, len(self.indices))
        for i in range(k):
            idx = self.indices[i]
            neighbors.append(self.train_datas[idx])
        return neighbors
    def predict(self, k=param.K_NEIGHBORS):
        neighbors = self.get_k_neighbors(k)
        count = {}
        most_voted, most_count = None, -1
        for n in neighbors:
            count[n.label] = count.get(n.label, 0) + 1
        for label, freq in count.items():
            if freq > most_count:
                most_count = freq
                most_voted = label
        return most_voted


