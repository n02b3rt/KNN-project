import numpy as np


def euclidean_distance(x1, x2):
    return (sum((x1 - x2) ** 2)) ** (1 / 2)


class KNN:
    def __init__(self, n_neighbors=1):
        # inicjalizacja hiperparametrów modelu - liczba sąsiadów, metryka itp
        self.n_neighnors = n_neighbors
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        # uczenie modelu - przyjęcie danych treningowych
        self.X = X
        self.y = y

        pass

    def predict(self, x_test: np.ndarray):
        # Wykonywanie predykcji na podstawie wiedzy z treningu i uwzględniając hiperparametry

        tablica_odleglosci = []

        for el in self.X:
            tablica_odleglosci.append(euclidean_distance(el, x_test))

        # print(tablica_odleglosci)

        indeksy_najmniejszch = []

        for _ in range(self.n_neighnors):
            # indeks_najmniejszej = -1
            # wartosc_najmniejszej = sys.maxsize
            # for i,e in enumerate(tablica_odleglosci):
            #      if e < wartosc_najmniejszej and i not in indeksy_najmniejszch:
            #         wartosc_najmniejszej = e
            #         indeks_najmniejszej = i
            #
            # indeksy_najmniejszch.append(indeks_najmniejszej)

            indeks = np.argmin(tablica_odleglosci)
            del tablica_odleglosci[indeks]
            indeksy_najmniejszch.append(int(indeks))

        print(indeksy_najmniejszch)

        decyzje = [int(self.y[i]) for i in indeksy_najmniejszch]
        print(decyzje)

        ucdecyzje = np.unique_counts(decyzje)
        maxucdecyzje = np.argmax(np.unique_counts(decyzje).counts)

        print(ucdecyzje)
        print(maxucdecyzje)

        print(ucdecyzje.values[maxucdecyzje])

        pass