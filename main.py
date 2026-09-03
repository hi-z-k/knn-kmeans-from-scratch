from data_loader import CropRecommendationLoader, WholesaleCustomersLoader, stratified_split
from knn import KNN
from kmeans import KMeans
import hyperparameters as param


def run_knn():
    print("KNN: Crop Recommendation")
    loader = CropRecommendationLoader()
    data = loader.load("data/crop_recommendation.csv")
    train_data, test_data = stratified_split(data, test_ratio=0.2, seed=42)

    knn = KNN(train_data)
    correct = 0
    for test_point in test_data:
        knn.set_test_data(test_point)
        predicted = knn.predict(param.K_NEIGHBORS)
        if predicted == test_point.label:
            correct += 1

    print(f"k={param.K_NEIGHBORS} | accuracy={correct / len(test_data):.4f}\n")


def run_kmeans():
    print("K-Means: Wholesale Customers")
    loader = WholesaleCustomersLoader()
    data = loader.load("data/wholesale_customers.csv")

    kmeans = KMeans(data, k=param.K_CLUSTERS)
    centroids, clusters = kmeans.fit()

    for idx, cluster in clusters.items():
        print(f"Cluster {idx}: {len(cluster)} customers | centroid={centroids[idx].features}")


if __name__ == "__main__":
    run_knn()
    run_kmeans()