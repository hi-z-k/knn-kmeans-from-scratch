import csv
import random
from abc import ABC, abstractmethod
from datatypes import Data


class DatasetLoader(ABC):
    expected_columns: int

    def load(self, path: str) -> list[Data]:
        dataset = []
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if not row:
                    continue
                while row and row[-1] == '':
                    row.pop()
                if len(row) != self.expected_columns:
                    raise ValueError(f"malformed row, expected {self.expected_columns} columns: {row}")
                dataset.append(self._row_to_data(row))
        return dataset

    @abstractmethod
    def _row_to_data(self, row: list[str]) -> Data:
        ...


class CropRecommendationLoader(DatasetLoader):
    expected_columns = 8

    def _row_to_data(self, row: list[str]) -> Data:
        feature_strs = row[:-1]
        label = row[-1]
        features = [float(x) for x in feature_strs]
        return Data(features=features, label=label)


class WholesaleCustomersLoader(DatasetLoader):
    expected_columns = 8

    def _row_to_data(self, row: list[str]) -> Data:
        feature_strs = row[2:]
        features = [float(x) for x in feature_strs]
        return Data(features=features, label=None)


def stratified_split(dataset: list[Data], test_ratio: float = 0.2, seed: int = None) -> tuple[list[Data], list[Data]]:
    rng = random.Random(seed)
    groups: dict[str, list[Data]] = {}
    for d in dataset:
        groups.setdefault(d.label, []).append(d)
    train, test = [], []
    for label, items in groups.items():
        items = items[:]
        rng.shuffle(items)
        split_point = int(len(items) * (1 - test_ratio))
        train.extend(items[:split_point])
        test.extend(items[split_point:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test