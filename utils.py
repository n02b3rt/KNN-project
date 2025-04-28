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