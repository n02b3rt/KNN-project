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
  Oceniono działanie modelu przy użyciu różnych metryk: Euklidesowej (`euclidean`) i Manhattan (`manhattan`).  
  Najlepsze wyniki osiągnięto dla odległości Euklidesowej.

- **Standaryzacja danych:**  
  Przeprowadzono eksperymenty z i bez standaryzacji cech.  
  Model trenowany na zestandaryzowanych danych osiągał znacznie wyższe R² i niższe MAE, co wskazuje na dużą wrażliwość KNN na skalę danych.
  Testowaliśmy dwie metody standaryzacji: MinMaxScaler i StandardScaler. Okazało się, że MinMaxScaler dał lepsze wyniki,
- **Wybór liczby cech:**  
  Eksperymentowano z różnymi podzbiorami cech przy użyciu metody **SelectKBest**. Okazało się, że wybór 5 najlepszych cech (zamiast wszystkich) zwiększył skuteczność predykcji i zmniejszył czas obliczeń. Testowano również różne metody selekcji cech, w tym: **f_regression**, **mutual_info_regression**

- **Porównanie na różnych podziałach danych:**  
  Wyniki porównano dla różnych podziałów danych trening/test (np. 70/30, 80/20).  
  Stwierdzono, że model był stabilny i dawał zbliżone wyniki w różnych wariantach.

---

Na podstawie przeprowadzonych eksperymentów wybrano optymalne parametry modelu KNN:
- liczba sąsiadów `k = 5`,
- odległość: Euklidesowa,
- dane: po standaryzacji,
- liczba cech: 5 najlepszych według SelectKBest (używając **f_regression**).
### Przedstawienie zbioru danych
źródło: Kaggle - [USA Real Estate Dataset](https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset)

Struktura danych: plik CSV

| Cecha           | Opis                                                            |
|-----------------|-----------------------------------------------------------------|
| brokered_by     | Broker / Agencja sprzedająca nieruchomość                       |
| status          | Status sprzedaży nieruchomości                                  |
| price           | Cena nieruchomości                                              |
| bed             | Liczba sypialni                                                 |
| bath            | Liczba łazienek                                                 |
| acre_lot        | Powierzchnia działki / lotu w akrach                            |
| street          | Adres nieruchomości (zakodowany)                                |
| city            | Miasto, w którym znajduje się nieruchomość                       |
| state           | Stan, w którym znajduje się nieruchomość                         |
| zip_code        | Kod pocztowy nieruchomości  

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

Najlepszy otrzymany przez nas wynik:

| Metryka    | Wartość  |
|------------|----------|
| MSE        | 36954381669.37 |
| MAE        | 117392.15   |
| R²         | 0.77     |


Dla wizualizacji błędów predykcji dodatkowo przedstawiono histogram rzeczywistych - przewidywanych wartości.
![image](https://github.com/user-attachments/assets/18815b06-1b57-4adb-8d55-d20232a58788)


## Wnioski
Dobór cech: Selekcja najlepszych cech, przeprowadzona za pomocą metod jak SelectKBest (z f_regression oraz mutual_info_regression), ma istotny wpływ na skuteczność modelu. Zastosowanie odpowiedniego zestawu cech (np. tylko numeryczne zmienne) przyczyniło się do poprawy wyników predykcji.

Optymalizacja liczby cech (k): Wybór liczby cech w metodzie SelectKBest okazał się kluczowy dla optymalizacji wyników. Wybranie mniejszej liczby cech, np. 5 zamiast wszystkich, zmniejszyło czas obliczeń i poprawiło skuteczność modelu.

Skalowanie cech: Użycie MinMaxScaler do skalowania cech przed wprowadzeniem do modelu KNN zwiększyło jego skuteczność. Niezsynchronizowane cechy mogą prowadzić do słabszych wyników, dlatego odpowiednia normalizacja jest kluczowa.

Optymalizacja K (liczby sąsiadów): Przeprowadzenie walidacji krzyżowej pokazało, że dla najlepszej wartości K=6 uzyskano najniższe średnie MSE. To sugeruje, że przy większych wartościach K model zaczyna tracić dokładność, co może prowadzić do przeuczenia.

Skuteczność modelu: Model KNN z optymalnymi parametrami osiągnął skuteczność predykcji cen nieruchomości na poziomie ponad 70%, co pokazuje solidność tego algorytmu w kontekście tak dużego zbioru danych.

Błędy predykcji: Błędy predykcji wykazały, że model dobrze radzi sobie w przypadku większości próbek, jednak błędy są nadal obecne, zwłaszcza dla skrajnych wartości. Większe odchylenia mogły być spowodowane dużymi różnicami w cenach nieruchomości.

Wykresy i analiza błędów: Wizualizacja rozkładu błędów predykcji wskazała na obecność skrajnych przypadków, które mogą wpływać na jakość modelu. Warto zwrócić uwagę na te przypadki i rozważyć dalsze usprawnienia.

Proponowane dalsze kroki: Rozbudowa modelu oraz przetestowanie innych algorytmów regresyjnych, takich jak Random Forest czy Gradient Boosting, mogłoby poprawić skuteczność modelu. Również rozważenie analizy interakcji między cechami może dostarczyć dodatkowych informacji.
