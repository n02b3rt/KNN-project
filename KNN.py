import numpy as np

class KNN:
    def __init__(self, n_neighbors=1):
        # inicjalizacja hiperparametrów modelu - liczba sąsiadów, metryka itp
        self.n_neighbors = n_neighbors

    def fit(self, X: np.ndarray, y: np.ndarray):
        # uczenie modelu - przyjęcie danych treningowych
        self.X = X
        self.y = np.array(y)

    def predict(self, x_test: np.ndarray):
        """
        TODO:
        Metoda predict powinna umożliwiać inferencję wielu obiektów na raz; niech
        przyjmuje macierz numpy o wymiarach 𝑛 × 𝑚, gdzie n oznacza liczbę obiektów
        testowych, a m liczbę atrybutów warunkowych.
        """

        # Wykonywanie predykcji na podstawie wiedzy z treningu i uwzględniając hiperparametry

        dists = []

        for x_test_point in x_test:
            distances = np.sqrt(((self.X - x_test_point) ** 2).sum(axis=1))
            dists.append(distances)

        dists = np.array(dists)

        # Znajdowanie n najbliższych sąsiadów
        neighbors_indices = np.argpartition(dists, self.n_neighbors, axis=1)[:, :self.n_neighbors]
        # print(neighbors_indices)

        # Predykcja - obliczanie średniej wartości z najbliższych sąsiadów
        predictions = np.mean(self.y[neighbors_indices], axis=1)
        # print(neighbors_indices)

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

    model = KNN(n_neighbors=3)
    model.fit(X_train, y_train)

    # Dane do testu modelu
    X_test = np.array([
        [105000, 3, 2, 0.12, 1000],
        [80000, 3, 2, 0.1, 1200],
    ])

    predictions = model.predict(X_test)

    print(f"Predykcje: {predictions}")