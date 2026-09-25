# DataScience_Unige

# California Housing Price Classification - Progetto Finale Data Science

Progetto sviluppato per il corso di **Introduzione alla Data Science** (Anno Accademico 2025/2026) presso l'Università degli Studi di Genova (UniGe).

## 📋 Panoramica del Progetto
Questo repository contiene la pipeline end-to-end di Data Science per la classificazione multi-classe dei prezzi delle case in California. L'obiettivo è prevedere la fascia di prezzo (`median_house_value`, suddivisa in 5 classi discrete) di un set di test inedito[cite: 13], partendo da informazioni geografiche, demografiche e strutturali delle abitazioni.

Il lavoro è suddiviso in due componenti principali richieste dalla consegna:
1. **Un Jupyter Notebook (`.ipynb`)**: contenente l'analisi esplorativa, la pulizia dei dati, la gestione dei valori mancanti, la trasformazione delle feature (log-transform) e la valutazione comparativa di diversi modelli di machine learning tramite Cross-Validation.
2. **Uno script Python eseguibile (`S6520241.py`)**: configurato per addestrare il modello migliore sul training set completo ed esportare le predizioni sul file di test.

---

## ⚙️ Pipeline e Algoritmi Implementati
Il progetto confronta diversi approcci di classificazione tramite la libreria `scikit-learn`:
* **Pre-processing e Pulizia:** Gestione dei valori mancanti tramite imputazione basata sulla media (calcolata rigorosamente sul training set per evitare *data leakage*), standardizzazione delle feature (`StandardScaler`) e applicazione di trasformazioni logaritmiche (`np.log1p`) per ridurre l'asimmetria delle variabili numeriche.
* **Loss Quadratica (Ridge Classifier):** Implementato tramite strategia *One-vs-Rest* per gestire la classificazione multi-classe, analizzando i coefficienti associati alle feature geografiche e socio-economiche.
* **Loss Logistica (Logistic Regression):** Classificatore lineare regolarizzato (sempre in modalità *One-vs-Rest*), studiando l'effetto del parametro di regolarizzazione $C$ sull'accuratezza di generalizzazione.
* **k-Nearest Neighbors (k-NN):** Algoritmo di *lazy learning*, valutando l'accuratezza al variare del parametro $k$ per bilanciare overfitting e underfitting.
* **Alberi Decisionali (Decision Tree Classifier):** Ottimizzato tramite `GridSearchCV` (analizzando i parametri di profondità massima `max_depth` e campioni minimi per foglia `min_samples_leaf`) per prevenire l'overfitting. Si è rivelato il modello più performante del confronto.

---

## 📊 Risultati e Modello Migliore
Dall'analisi di Cross-Validation (K-Fold con $k=5$), il modello basato sugli **Alberi Decisionali** ha ottenuto le prestazioni di generalizzazione più elevate (accuratezza media vicina al 58.8%), sfruttando in modo ottimale feature chiave come il reddito mediano (`median_income`), la distanza dalla costa (`distance_to_coast`) e le coordinate geografiche.

---

## 🚀 Istruzioni per l'Esecuzione

### Requisiti di Sistema
Lo script di inferenza è progettato per girare con le librerie standard di riferimento specificate dalla committenza:
* `python` >= 3.8
* `pandas`
* `numpy`
* `scikit-learn`

### Esecuzione dello Script di Test
Lo script rispetta rigorosamente i parametri da riga di comando richiesti per la valutazione automatica:

```bash
python S6520241.py --train houses_data.csv --test houses_test.csv
```

L'esecuzione genererà automaticamente il file di output S6520241.txt contenente una predizione di classe per ogni riga del dataset di test

## 👤 Autore
Francesco Giuseppino (Matricola: 6520241)  
Corso di Laurea in Informatica – Università degli Studi di Genova (UniGe)   
