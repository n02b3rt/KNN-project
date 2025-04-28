import os
import joblib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from utils import auto_select_features, read_dataset, clean_dataset
from sklearn.model_selection import KFold, train_test_split
from KNN import KNN


def main():
    df = read_dataset()

    # Ograniczenie do pierwszych 300 wierszy
    df_virginia = df[df['state'] == 'Virginia']
    df = df_virginia.sample(n=2000, random_state=42)

    # Obliczanie amplitudy cen (największa cena - najmniejsza cena)
    max_price = df['price'].max()  # Największa cena
    min_price = df['price'].min()  # Najmniejsza cena
    amplitude = max_price - min_price  # Amplituda cen

    #czyszczenie wyników
    df = clean_dataset(df)

    # Wyświetlanie wyników
    print(f"Największa cena: {max_price}")
    print(f"Najmniejsza cena: {min_price}")
    print(f"Amplituda cen: {amplitude}")

    print("Pierwsze 5 rekordów z ograniczonego zbioru danych:")
    print(df.head())

    # Przygotowanie cech i zmiennej docelowej
    X = df.select_dtypes(include=[np.number]).drop(columns=['price'])  # Usuwamy 'price' z numerycznych kolumn
    y = df['price']

    # Skalowanie cech
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


    # Wybór najlepszych cech
    X_selected, selected_features = auto_select_features(X_scaled, y, k=5)
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

    # Podział danych
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Trenowanie modelu
    model = KNN(n_neighbors=best_k)
    model.fit(X_train, y_train)

    # Predykcja
    y_pred = model.predict(X_test)

    # Ocena
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Metryki dla modelu: MSE = {mse:.2f}, MAE = {mae:.2f}, R2 = {r2:.2f}")

    # Obliczanie skuteczności
    y_mean = np.mean(y_test)  # Średnia wartość rzeczywistych danych
    y_pred_base = np.full_like(y_test, y_mean)  # Model bazowy (średnia wartość)

    # Obliczanie MSE bazowego modelu
    mse_base = np.mean((y_test - y_pred_base) ** 2)

    # Obliczanie skuteczności
    efficiency = 1 - (mse / mse_base)

    # Wyświetlanie skuteczności w procentach
    print(f"Skuteczność modelu: {efficiency * 100:.2f}%")

    # Obliczanie błędów predykcji
    errors = y_test - y_pred
    print(f"Błędy predykcji (rzeczywiste - przewidywane): {errors}")

    # Znalezienie największego i najmniejszego błędu
    max_error = errors.max()  # Największy błąd
    min_error = errors.min()  # Najmniejszy błąd

    # Wyświetlenie wyników
    print(f"Największy błąd predykcji: {max_error}")
    print(f"Najmniejszy błąd predykcji: {min_error}")

    # Tworzenie wykresu
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(x=max_error, color='red', linestyle='dashed', linewidth=2, label=f'Max Błąd: {max_error}')
    plt.axvline(x=min_error, color='green', linestyle='dashed', linewidth=2, label=f'Min Błąd: {min_error}')
    plt.title("Rozkład błędów predykcji")
    plt.xlabel("Błąd predykcji (rzeczywiste - przewidywane)")
    plt.ylabel("Liczba przypadków")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Zapisanie modelu
    model_path = 'realtor_knn_model.pkl'
    if not os.path.exists(model_path):
        joblib.dump(model, model_path)
        print(f"Model zapisano w: {model_path}")

if __name__ == '__main__':
    main()
