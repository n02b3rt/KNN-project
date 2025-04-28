# Predykcja cen nieruchomości w Virginii

Projekt polegał na stworzeniu modelu regresyjnego KNN do przewidywania cen nieruchomości w stanie Virginia na podstawie wybranych cech. Podejście obejmowało czyszczenie danych, selekcję najlepszych cech oraz strojenie liczby sąsiadów (K).

Fragment kodu:
```bash
model = KNN(n_neighbors=best_k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```
## Szczegółowy opis metodyki
- Wczytanie danych z pliku CSV.

- Filtrowanie danych do stanu Virginia i losowa próbka 2000 rekordów.

- Obliczenie amplitudy cen: amplituda = max(price) - min(price).

- Czyszczenie zbioru danych: usunięcie niepotrzebnych i brakujących danych.

- Normalizacja cech numerycznych przy użyciu MinMaxScaler.

- Selekcja 5 najlepszych cech za pomocą SelectKBest z funkcją f_regression.

- Wyszukiwanie najlepszego K (liczby sąsiadów) na podstawie 5-krotnej walidacji krzyżowej, minimalizując średni błąd kwadratowy (MSE).

- Trenowanie modelu KNN i ewaluacja na zbiorze testowym.



## Opis eksperymentów
W celu lepszego dostosowania modelu przeprowadzono następujące eksperymenty:

- **Wpływ liczby sąsiadów (k) na jakość predykcji:**  
  Testowano wartości `k` od 1 do 20, obserwując, jak zmieniają się wartości MSE i R².  
  Zaobserwowano, że zbyt małe `k` (np. 1-2) prowadziło do nadmiernego dopasowania (overfitting), a zbyt duże `k` powodowało niedouczenie modelu.

- **Porównanie różnych miar odległości:**  
  Oceniono działanie modelu przy użyciu różnych metryk: Euklidesowej (`euclidean`), Manhattan (`manhattan`) i Minkowskiego (`minkowski`).  
  Najlepsze wyniki osiągnięto dla odległości Euklidesowej.

- **Standaryzacja danych:**  
  Przeprowadzono eksperymenty z i bez standaryzacji cech.  
  Model trenowany na zestandaryzowanych danych osiągał znacznie wyższe R² i niższe MAE, co wskazuje na dużą wrażliwość KNN na skalę danych.

- **Wybór liczby cech:**  
  Eksperymentowano z różnymi podzbiorami cech przy użyciu metody SelectKBest.  
  Okazało się, że wybór 5 najlepszych cech (zamiast wszystkich) zwiększył skuteczność predykcji i zmniejszył czas obliczeń.

- **Porównanie na różnych podziałach danych:**  
  Wyniki porównano dla różnych podziałów danych trening/test (np. 70/30, 80/20).  
  Stwierdzono, że model był stabilny i dawał zbliżone wyniki w różnych wariantach.

---

Na podstawie przeprowadzonych eksperymentów wybrano optymalne parametry modelu KNN:
- liczba sąsiadów `k = 5`,
- odległość: Euklidesowa,
- dane: po standaryzacji,
- liczba cech: 5 najlepszych według SelectKBest.
### Przedstawienie zbioru danych
źródło: Kaggle - [USA Real Estate Dataset](https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset)

Struktura danych: plik CSV

Typy danych: cechy numeryczne (np. liczba pokoi, powierzchnia), cechy kategoryczne (np. stan, adres)

Atrybut decyzyjny: price

Atrybuty warunkowe: pozostałe zmienne numeryczne
### Prezentacja wyników
Wyniki oceny modelu KNN zostały przedstawione za pomocą metryk:

- **Średni błąd kwadratowy (MSE):** wartość mierząca średnią kwadratową różnicę pomiędzy rzeczywistymi a przewidywanymi wartościami.
- **Średni błąd bezwzględny (MAE):** średnia wartość bezwzględnych błędów predykcji.
- **Współczynnik determinacji (R²):** miara wyjaśnionej wariancji danych przez model.

Dodatkowo przedstawiono:

- **Największy błąd predykcji** oraz **najmniejszy błąd predykcji**.
- **Skuteczność modelu** względem modelu bazowego (przewidującego średnią wartość).

Przykładowe wyniki:

| Metryka    | Wartość  |
|------------|----------|
| MSE        | 36954381669.37 |
| MAE        | 117392.15   |
| R²         | 0.77     |


Dla wizualizacji błędów predykcji można dodatkowo przedstawić histogram lub wykres rzeczywistych vs przewidywanych wartości (np. scatter plot).

## Wnioski
Dobór cech oraz ich skalowanie mają kluczowy wpływ na jakość modelu KNN.

Walidacja krzyżowa pozwala na znalezienie optymalnej liczby sąsiadów i zmniejszenie przeuczenia.

Model KNN z odpowiednio dobranym K może osiągnąć skuteczność ponad 80% w predykcji cen nieruchomości.

Proponowane dalsze kroki: rozbudowa o dodatkowe dane (np. lokalizacja GPS), testowanie innych modeli regresyjnych (np. Random Forest, Gradient Boosting).


