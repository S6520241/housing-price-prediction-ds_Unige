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

## 📄 Documentazione del Progetto
La specifica dettagliata del progetto fornita dai docenti è inclusa nel file **`descrizione_progetto.pdf`** presente nella repository.

## 👤 Autore
Francesco Giuseppino (Matricola: 6520241)  
Corso di Laurea in Informatica – Università degli Studi di Genova (UniGe)   


_________________________________________________________

# DataScience_Unige

# California Housing Price Classification - Data Science Final Project

Project developed for the **Introduction to Data Science** course (Academic Year 2025/2026) at the University of Genoa (UniGe).

## 📋 Project Overview
This repository contains the end-to-end Data Science pipeline for the multi-class classification of California housing prices. The goal is to predict the price tier (`median_house_value`, divided into 5 discrete classes) of an unseen test set, starting from geographical, demographic, and structural housing information.

The work is divided into two main components required by the assignment:
1. **A Jupyter Notebook (`.ipynb`)**: containing exploratory data analysis, data cleaning, handling missing values, feature transformation (log-transform), and comparative evaluation of various machine learning models using Cross-Validation.
2. **An executable Python script (`S6520241.py`)**: configured to train the best model on the complete training set and export predictions on the test file.

---

## ⚙️ Pipeline and Implemented Algorithms
The project compares different classification approaches using the `scikit-learn` library:
* **Pre-processing and Cleaning:** Handling missing values via mean-based imputation (strictly calculated on the training set to prevent *data leakage*), feature standardization (`StandardScaler`), and logarithmic transformations (`np.log1p`) to reduce numerical feature skewness.
* **Quadratic Loss (Ridge Classifier):** Implemented via a *One-vs-Rest* strategy to handle multi-class classification, analyzing the coefficients associated with geographical and socio-economic features.
* **Logistic Loss (Logistic Regression):** Regularized linear classifier (also in *One-vs-Rest* mode), studying the effect of the regularization parameter $C$ on generalization accuracy.
* **k-Nearest Neighbors (k-NN):** A *lazy learning* algorithm, evaluating accuracy across different values of $k$ to balance overfitting and underfitting.
* **Decision Trees (Decision Tree Classifier):** Optimized using `GridSearchCV` (analyzing max depth `max_depth` and minimum samples per leaf `min_samples_leaf`) to prevent overfitting. It proved to be the best-performing model in the comparison.

---

## 📊 Results and Best Model
From the Cross-Validation analysis (K-Fold with $k=5$), the **Decision Tree** model achieved the highest generalization performance (average accuracy close to 58.8%), effectively leveraging key features such as median income (`median_income`), distance to the coast (`distance_to_coast`), and geographical coordinates.

---

## 🚀 Execution Instructions

### System Requirements
The inference script is designed to run with the standard reference libraries specified by the course instructors:
* `python` >= 3.8
* `pandas`
* `numpy`
* `scikit-learn`

### Running the Test Script
The script strictly complies with the command-line parameters required for automated evaluation:

```bash
python S6520241.py --train houses_data.csv --test houses_test.csv
```
Execution will automatically generate the output file S6520241.txt containing one class prediction per row of the test dataset

## 📄 Project Documentation
The detailed project specifications provided by the instructors are included in the descrizione_progetto.pdf file located in the repository.

## 👤 Author
Francesco Giuseppino (Student ID: 6520241)

Bachelor's Degree in Computer Science – University of Genoa (UniGe)
