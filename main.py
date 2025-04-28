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
import os
import subprocess
import zipfile

import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler


# Funkcja do wyboru najlepszych cech
def auto_select_features(X, y, k=4):
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_columns = X.columns[selector.get_support()]
    return X_selected, selected_columns

# Wczytanie danych z Kaggle za pomocą kagglehub
def read_dataset():
    zip_filename = "usa-real-estate-dataset.zip"
    csv_filename = "realtor-data.zip.csv"

    # Sprawdź, czy plik ZIP istnieje
    if not os.path.exists(zip_filename):
        print("Plik nie istnieje. Pobieranie...")
        curl_command = [
            "curl",
            "-L",
            "-o",
            zip_filename,
            "https://www.kaggle.com/api/v1/datasets/download/ahmedshahriarsakib/usa-real-estate-dataset"
        ]
        subprocess.run(curl_command, check=True)
    else:
        print("Plik ZIP już istnieje. Pomijam pobieranie.")

    # Wypakuj ZIP jeśli plik CSV nie istnieje
    if not os.path.exists(csv_filename):
        print("Rozpakowywanie archiwum ZIP...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(".")

        # Usuwanie pliku ZIP po rozpakowaniu
        print("Usuwanie pliku ZIP...")
        os.remove(zip_filename)
    else:
        print("Plik CSV już istnieje. Pomijam rozpakowywanie.")

    # Wczytaj CSV do DataFrame
    print("Wczytywanie danych...")
    df = pd.read_csv(csv_filename)

    return df


def main():
    df = read_dataset()
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

    # Przygotowanie cech i zmiennej docelowej
    X = df[['bed', 'bath', 'acre_lot', 'house_size']]
    y = df['price']

    # Skalowanie cech
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


    # Wybór najlepszych cech
    X_selected, selected_features = auto_select_features(X_scaled, y, k=4)
    print(f"Wybrane cechy: {list(selected_features)}")

if __name__ == '__main__':
    main()