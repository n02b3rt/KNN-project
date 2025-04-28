import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

class KNN:
    def __init__(self, n_neighbors=1, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = np.array(y)

    def predict(self, X_test: np.ndarray):
        """
        Funkcja do predykcji dla wielu punktów testowych.
        """

        # Wybór odpowiedniej metryki
        if self.metric == 'euclidean':
            dists = euclidean_distances(X_test, self.X)
        elif self.metric == 'manhattan':
            dists = manhattan_distances(X_test, self.X)
        else:
            raise ValueError("Unknown metric: choose 'euclidean' or 'manhattan'")

        # Znajdowanie n najbliższych sąsiadów
        neighbors_indices = np.argsort(dists, axis=1)[:, :self.n_neighbors]

        # Predykcja - obliczanie średniej wartości z najbliższych sąsiadów
        predictions = np.mean(self.y[neighbors_indices], axis=1)

        return predictions


if __name__ == '__main__':
    # Dane treningowe (price, bed, bath, acreage, house_size)
    X_train = np.array([
        [105000, 3, 2, 0.12, 920],
        [80000, 4, 2, 0.08, 1527],
        [67000, 2, 1, 0.15, 748],
        [145000, 4, 2, 0.1, 1800],
        [65000, 6, 2, 0.05, 1200]
    ])

    # Ceny
    y_train = np.array([103378, 52707, 103379, 31239, 34632])

    # Skalowanie danych
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Tworzenie modelu z metryką Manhattan
    model = KNN(n_neighbors=3, metric='manhattan')
    model.fit(X_train_scaled, y_train)

    # Dane do testu modelu
    X_test = np.array([
        [105000, 3, 2, 0.12, 1000],
        [80000, 3, 2, 0.1, 1200],
    ])

    # Skalowanie danych testowych
    X_test_scaled = scaler.transform(X_test)

    predictions = model.predict(X_test_scaled)

    print(f"Predykcje: {predictions}")
