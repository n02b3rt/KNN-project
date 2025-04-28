import pandas as pd

from utils import read_dataset


def main():
    # Załaduj dane
    df = read_dataset()

    # Sprawdzenie, czy kolumna 'state' istnieje
    if 'state' in df.columns:
        # Grupowanie danych po stanie i liczenie liczby rekordów w każdym stanie
        state_counts = df['state'].value_counts()

        # Tworzenie rankingu stanów po liczbie rekordów
        state_ranking = state_counts.sort_values(ascending=False)

        # Wyświetlanie rankingu stanów
        print("Ranking stanów według liczby rekordów:")
        print(state_ranking)

        # Opcjonalnie: wyświetlanie top 10 stanów
        print("\nTop 10 stanów:")
        print(state_ranking.head(10))
    else:
        print("Brak kolumny 'state' w danych.")

if __name__ == '__main__':
    main()
