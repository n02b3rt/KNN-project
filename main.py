import os
import joblib
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from utils import auto_select_features,read_dataset
from sklearn.model_selection import KFold, train_test_split
from KNN import KNN

def main():
    # --- Wczytanie danych ---
    df = read_dataset()
    print("Oryginalny zbiór danych (przed czyszczeniem):")
    print(df.info())
    print(df.head())

    # ---  Czyszczenie danych ---
    # Zamiana wartości nieprawidłowych ('unset') w kolumnie 'price' na NaN
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # Usunięcie wierszy z brakami w kluczowych kolumnach
    df = df.dropna(subset=['acre_lot', 'price'])

    # Usunięcie wierszy, gdzie cena wynosi dokładnie 1
    df = df[df['price'] != 1]

    # Wypełnienie braków zerami tylko w kolumnach numerycznych
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Resetowanie indeksu po usunięciu wierszy
    df = df.reset_index(drop=True)

    print("\nZbiór danych po czyszczeniu:")
    print(df.info())
    print(df.head())

    # ---  (Opcjonalnie) Ograniczenie zbioru danych ---
    df = df.head(300)

    # ---  Przygotowanie cech (X) i zmiennej docelowej (y) ---
    X = df.select_dtypes(include=[np.number]).drop(columns=['price'])
    y = df['price']

    # ---  Skalowanie cech ---
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # ---  Wybór najlepszych cech ---
    X_selected, selected_features = auto_select_features(X_scaled, y, k=4)
    print(f"Wybrane cechy: {list(selected_features)}")

    # ---  Wyszukiwanie najlepszego K (KFold Cross Validation) ---
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    best_k = None
    best_mse = float('inf')

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
        print(f"K: {k}, Średnie MSE (CV): {mean_mse:.4f}")

        if mean_mse < best_mse:
            best_k = k
            best_mse = mean_mse

    print(f"\nNajlepsze K: {best_k} (Średnie MSE: {best_mse:.4f})")

    # ---  Podział danych na zbiór treningowy i testowy ---
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # ---  Trenowanie końcowego modelu ---
    final_model = KNN(n_neighbors=best_k)
    final_model.fit(X_train, y_train)

    # ---  Predykcja na zbiorze testowym ---
    y_pred = final_model.predict(X_test)

    # ---  Ocena modelu ---
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nMetryki końcowego modelu:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")

    # ---  Zapisanie modelu do pliku ---
    model_path = 'realtor_knn_model.pkl'
    if not os.path.exists(model_path):
        joblib.dump(final_model, model_path)
        print(f"\nModel zapisano w: {model_path}")

if __name__ == '__main__':
    main()
