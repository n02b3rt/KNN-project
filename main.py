"""

TODO:
- pobranie i wczytanie dataset z kagglehub (wyświetlić)
- usunąć recordy w których coś brakuje (uzupelnić)
- Należy zastosować standaryzację lub skalowanie min-max w celu sprowadzenia
danych do tej samej skali.
- Algorytm powinien zostać przetestowany metodą train and test lub metodą k
krotnej walidacji krzyżowej na samodzielnie wybranym, oryginalnym zbiorze
danych.
- Należy obliczyć metryki otrzymanego modelu, np. accuracy score i wyznaczyć
macierz pomyłek.
- Zmodyfikowany model powinien zostać porównany ze źródłowym modelem –
należy przedstawić różnice i podjąć próbę wyjaśnienia źródła różnic.
- Należy przedstawić krótkie podsumowanie i interpretację wyników.

"""

import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.impute import SimpleImputer

def main():
    # Wczytanie danych z Kaggle za pomocą kagglehub
    file_path = "realtor-data.zip.csv"
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "ahmedshahriarsakib/usa-real-estate-dataset",
        file_path,
    )

    # Ograniczenie do pierwszych 300 wierszy
    df = df.head(300)
    print("Pierwsze 5 rekordów z ograniczonego zbioru danych:")
    print(df.head())

    # Usuwanie brakujących wartości za pomocą imputera
    imputer = SimpleImputer(strategy='mean')

    # Wydzielenie kolumn numerycznych i kategorycznych
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Uzupełnienie braków
    df[numeric_columns] = pd.DataFrame(imputer.fit_transform(df[numeric_columns]), columns=numeric_columns)
    for column in categorical_columns:
        if not df[column].mode().empty:
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            print(f"Kolumna {column} jest pusta lub brak wartości do imputacji.")


if __name__ == '__main__':
    main()