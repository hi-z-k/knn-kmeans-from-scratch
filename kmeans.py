import random
from typing import Optional

import distance as dist
import hyperparameters as param
from datatypes import Data

class KMeans:
    def __init__(self, training_data: list[Data], k: int = param.K_CLUSTERS):
        if not training_data:
            raise ValueError("You must provide data in order to do clustering.")
        if k <= 0 or k > len(training_data):
            raise ValueError(f"k must be between 1 and {len(training_data)}")

        self.k = k
        self.training_data = training_data

        self.centroids: list[Data] = []
        self.clusters: dict[int, list[Data]] = {}
    def _get_starting_centroids(self):
        idxs = random.sample(range(len(self.training_data)), k=self.k)
        self.centroids = [
            Data(features=self.training_data[idx].features[:], label=None)
            for idx in idxs
        ]
    def _nearest_centroid(self, data: Data) -> int:
        near_pos = -1
        shortest_distance = float("inf")
        for idx, centroid in enumerate(self.centroids):
            distance = dist.unrooted_distance(data.features, centroid.features)
            if distance < shortest_distance:
                shortest_distance = distance
                near_pos = idx
        return near_pos
    def _cluster_by_centroid(self) -> dict[int, list[Data]]:
        curr_clusters: dict[int, list[Data]] = {i: [] for i in range(self.k)}
        for data in self.training_data:
            idx = self._nearest_centroid(data)
            curr_clusters[idx].append(data)
        return curr_clusters
    def _didnt_converge(self, prev_centroid:Data, curr_centroid:Data) -> bool:
        for prev, curr in zip(prev_centroid.features, curr_centroid.features):
            if abs(prev - curr) > param.TOLERANCE:
                return True
        return False
    def _handle_orphan_centroid(self, centroid_idx: int):
        replacement_data: Optional[Data] = None
        farthest_distance: float = float("-inf")

        for idx, cluster in self.clusters.items():
            curr_centroid = self.centroids[idx]
            for data in cluster:
                distance = dist.unrooted_distance(data.features, curr_centroid.features)
                if distance > farthest_distance:
                    farthest_distance = distance
                    replacement_data = data
        if replacement_data is not None:
            self.centroids[centroid_idx] = Data(features=replacement_data.features[:], label=None)

    def _update_centroid(self)->bool:
        dimension = len(self.training_data[0].features)
        did_centroid_changed = False
        for idx, cluster in self.clusters.items():
            prev_centroid = self.centroids[idx]
            if not cluster:
                self._handle_orphan_centroid(idx)
                did_centroid_changed = True
                continue

            new_features: list[float] = []
            for axis in range(dimension):
                total = sum(data.features[axis] for data in cluster)
                new_features.append(total / len(cluster))

            new_centroid = Data(features=new_features, label=None)
            if self._didnt_converge(prev_centroid, new_centroid):
                did_centroid_changed = True

            self.centroids[idx] = new_centroid
        return did_centroid_changed
    def fit(self)-> tuple[list[Data], dict[int, list[Data]]]:
        self._get_starting_centroids()
        self.clusters = self._cluster_by_centroid()

        for _ in range(param.MAX_ITERATIONS):
            centroid_changed = self._update_centroid()
            new_clusters = self._cluster_by_centroid()

            if not centroid_changed and new_clusters == self.clusters:
                break

            self.clusters = new_clusters

        return self.centroids, self.clusters
