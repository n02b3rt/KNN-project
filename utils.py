import os
import subprocess
import zipfile
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

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

    # Sprawdź, czy plik CSV istnieje
    if not os.path.exists(csv_filename):
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

def clean_data(df):
    # Wypełnij brakujące wartości w kolumnie 'bed' zerami
    df['bed'] = df['bed'].fillna(0)

    # Wypełnij brakujące wartości w kolumnie 'bath' zerami
    df['bath'] = df['bath'].fillna(0)

    # Wypełnij brakujące wartości w 'acre_lot' średnią
    df['acre_lot'] = df['acre_lot'].fillna(df['acre_lot'].mean())

    # Konwertuj 'street' na liczby (lub NaN jeśli nie da się)
    df['street'] = pd.to_numeric(df['street'], errors='coerce')

    # Zamień wartości kategorii 'city' i 'state' na kody liczbowe
    df['city'] = pd.Categorical(df['city']).codes
    df['state'] = pd.Categorical(df['state']).codes

    # Konwertuj 'zip_code' na liczby (lub NaN)
    df['zip_code'] = pd.to_numeric(df['zip_code'], errors='coerce')

    # Usuń niepotrzebne kolumny
    df = df.drop(columns=['prev_sold_date'])

    # Dodatkowe usuwanie NaN
    df = df.dropna()

    # Resetowanie indeksu po usunięciu wierszy
    df = df.reset_index(drop=True)

    return df