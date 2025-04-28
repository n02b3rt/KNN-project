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
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from utils import auto_select_features,read_dataset

from sklearn.model_selection import KFold
from KNN import KNN

import numpy as np

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

    # Walidacja k-krotna
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    best_k = None
    best_mse = float('inf')

    # Przeszukiwanie K
    for k in range(1, 21):
        model = KNN(n_neighbors=k)
        mse_scores = []

        for train_idx, test_idx in kfold.split(X_selected):
            X_train_kf, X_test_kf = X_selected[train_idx], X_selected[test_idx]
            y_train_kf, y_test_kf = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train_kf, y_train_kf)
            y_pred_kf = model.predict(X_test_kf)
            mse_scores.append(mean_squared_error(y_test_kf, y_pred_kf))

        mean_mse = np.mean(mse_scores)
        print(f"K: {k}, Średnie MSE (CV): {mean_mse:.2f}")

        if mean_mse < best_mse:
            best_k = k
            best_mse = mean_mse

    print(f"Najlepsze K: {best_k} z najniższym średnim MSE: {best_mse:.2f}")

if __name__ == '__main__':
    main()
