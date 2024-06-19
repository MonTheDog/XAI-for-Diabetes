import pandas as pd
import numpy as np
import os
# Per evitare problemi con la libreria tensorflow, disabilitiamo le ottimizzazioni per DNN e settiamo il livello di log a 1
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import random as rn
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from keras import Sequential
from keras.src.layers import Dense
import tensorflow as tf
import pickle

# Settiamo il seed per la riproducibilità
os.environ["PYTHONHASHSEED"] = "25"
np.random.seed(25)
rn.seed(25)
tf.random.set_seed(25)

# Ignoriamo i warning
warnings.filterwarnings("ignore")

# Cambiamo le impostazioni di pandas per visualizzare tutte le colonne
pd.set_option('display.max_columns', None)


def dataset_informations(dataset):
    print("\n", "-*-" * 10, "Informazioni sul dataset", "-*-" * 10, "\n")
    print(dataset.info())  # 768 righe, 9 colonne
    # Notiamo che non abbiamo valori nan e non abbiamo problemi con i tipi di dati, quindi proseguiamo

    # Visualizziamo le informazioni statistiche del dataset, per capire se ci sono valori anomali
    print("\n", "-*-" * 10, "Informazioni sulle colonne", "-*-" * 10, "\n")
    print(dataset.describe().T)
    # Notiamo che le colonne "Glucose", "Blood Pressure", "SkinThickness", "Insulin" e "BMI" hanno valori minimi pari
    # a 0, il che è impossibile per queste feature, quindi possiamo dedurre che questi siano in realtà valori mancanti


def find_missing_values(dataset):
    # Sostituiamo i valori 0 con nan, per evitare che il classificatore sbagli
    dataset[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = (
        dataset[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.nan))

    # Controlliamo quanti sono i valori mancanti
    print("\n", "-*-"*10, "Valori mancanti", "-*-"*10, "\n")
    print(dataset.isnull().sum())

    # Glucose 5 valori mancanti
    # BloodPressure 35 valori mancanti
    # SkinThickness 227 valori mancanti
    # Insulin 374 valori mancanti
    # BMI 11 valori mancanti


def fix_missing_values(dataset):
    # Andiamo a studiare le distribuzioni per comprendere cosa fare con questi valori
    dataset.hist(figsize=(20, 20), color='skyblue', bins=15, grid=False)
    plt.savefig("images/histograms_pre_imputation.png")
    plt.show()
    plt.close("all")

    # Notiamo che Glucose e Blood Pressure seguono una distribuzione normale, quindi possiamo
    # sostituire i valori mancanti con la media
    dataset["Glucose"].fillna(dataset["Glucose"].mean(), inplace=True)
    dataset["BloodPressure"].fillna(dataset["BloodPressure"].mean(), inplace=True)

    # Notiamo poi che Skin Thickness, Insulin e BMI seguono una distribuzione asimmetrica, quindi possiamo sostituire i
    # valori mancanti con la mediana
    dataset["SkinThickness"].fillna(dataset["SkinThickness"].median(), inplace=True)
    dataset["Insulin"].fillna(dataset["Insulin"].median(), inplace=True)
    dataset["BMI"].fillna(dataset["BMI"].median(), inplace=True)

    # Controlliamo che le distribuzioni non siano cambiate sensibilmente
    dataset.hist(figsize=(20, 20), color='skyblue', bins=15, grid=False)
    plt.savefig("images/histograms_post_imputation.png")
    plt.show()
    plt.close("all")

    # Poiché le distribuzioni non sono cambiate sensibilmente possiamo procedere


def find_duplicated_rows(dataset):
    print("\n", "-*-" * 10, "Righe duplicate", "-*-" * 10, "\n")
    print("Valori duplicati nel dataset: ", dataset.duplicated().sum())
    # 0 righe duplicate, quindi proseguiamo


def is_dataset_balanced(dataset):
    plt.pie(dataset["Outcome"].value_counts(), labels=["Non Diabetico", "Diabetico"],
            explode=(0, 0.05), autopct="%0.2f", colors=["#f77189", "#36ada4"])
    plt.savefig("images/dataset_balancement.png")
    plt.show()
    plt.close("all")

    print("\n", "-*-" * 10, "Bilanciamento del dataset", "-*-" * 10, "\n")
    print(dataset["Outcome"].value_counts(normalize=True) * 100)
    # 65% non diabetici, 35% diabetici
    print("\n", "-*-" * 10, "Numero di istanze di outcome", "-*-" * 10, "\n")
    print(dataset["Outcome"].value_counts())
    # 500 non diabetici, 268 diabetici

    # Il dataset non è bilanciato, quindi proseguiamo con il bilanciamento


def dataset_balancement(dataset):
    # Dividiamo il dataset in feature e target
    X = dataset.drop("Outcome", axis=1)
    y = dataset["Outcome"]

    # Bilanciamo il dataset con SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print("\n", "-*-" * 10, "Numero di istanze dopo SMOTE", "-*-" * 10, "\n")
    print(y_resampled.value_counts())
    # 500 non diabetici, 500 diabetici

    resampled_dataset = pd.concat([X_resampled, y_resampled], axis=1)

    return resampled_dataset

    # Il dataset è ora bilanciato, quindi proseguiamo con l'analisi


def generate_pairplot(dataset):
    sns.pairplot(dataset, hue='Outcome', palette="husl", corner=True, vars=["Pregnancies", "Glucose", "BloodPressure",
                                                                            "SkinThickness", "Insulin", "BMI",
                                                                            "DiabetesPedigreeFunction", "Age", "Outcome"])
    plt.savefig("images/pairplot.png")
    plt.show()
    plt.close("all")
    # Notiamo che "Glucose", "BMI" e "Age" sono le feature che sembrano avere una correlazione maggiore con l'outcome
    # quindi potrebbero essere le feature più importanti per il classificatore


def generate_heatmap(dataset):
    plt.figure(figsize=(14, 12))
    sns.heatmap(dataset.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.savefig("images/heatmap.png")
    plt.show()
    plt.close("all")
    # Notiamo che le feature non sono molto correlate tra loro, e confermiamo che "Glucose" e "BMI" e sono le feature
    # più correlate con l'outcome


def scale_dataset(dataset):
    scaler = StandardScaler()
    scaler.fit(dataset.drop(["Outcome"], axis=1))
    X = scaler.transform(dataset.drop(["Outcome"], axis=1))
    y = dataset["Outcome"]

    # Salviamo lo scaler per poter riportare i dati alla scala originale in un secondo momento
    with open("scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    return X, y


def create_neural_network():
    # Creiamo la rete neurale
    # La seguente architettura è stata scelta basandosi su un approccio trial and error
    # Essa è composta da 3 layers, rispettivamente di 32, 16 e 8 neuroni
    model = Sequential()
    model.add(Dense(32, activation="relu", kernel_initializer="normal", input_dim=8))
    model.add(Dense(16, activation="relu", kernel_initializer="normal", input_dim=32))
    model.add(Dense(8, activation="relu", kernel_initializer="normal", input_dim=16))
    model.add(Dense(1, activation="sigmoid"))

    # Compiliamo il modello
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


def train_models(models, X_train, y_train):
    # Effettuiamo il train di tutti i modelli
    for key in models.keys():
        if key == "Neural Network":
            print("\n", "-*-" * 10, "Training rete Neurale", "-*-" * 10, "\n")
            models[key].fit(X_train, y_train, epochs=100, batch_size=64)
        else:
            models[key].fit(X_train, y_train)


def test_models(models, X_test, y_test, filename="scores.csv"):
    # Prepariamo le liste per le metriche
    accuracy_list, precision_list, recall_list, f1_list, auc_list = [], [], [], [], []

    # Per ogni modello andiamo a calcolare le metriche e a visualizzare la confusion matrix e la ROC curve
    for key in models.keys():

        y_pred = models[key].predict(X_test)

        if key == "Neural Network" or key == "Tuned Neural Network":
            y_pred = (y_pred > 0.5).astype(int)

        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        precision = precision_score(y_true=y_test, y_pred=y_pred)
        recall = recall_score(y_true=y_test, y_pred=y_pred)
        f1 = f1_score(y_true=y_test, y_pred=y_pred)
        auc = roc_auc_score(y_true=y_test, y_score=y_pred)

        fpr, tpr, _ = roc_curve(y_test, y_pred)

        # Codice per generare la ROC Curve
        plt.plot(fpr, tpr, color="darkorange", lw=2)
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve " + key)
        plt.savefig("images/" + key + "_roc_curve.png")
        plt.show()
        plt.close("all")

        # Codice per generare la confusion matrix
        matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
        sns.heatmap(matrix, annot=True, cmap="Blues", fmt="d")
        plt.title("Confusion Matrix " + key)
        plt.savefig("images/" + key + "_confusion_matrix.png")
        plt.show()
        plt.close("all")

        accuracy_list.append(round(accuracy, 4))
        precision_list.append(round(precision, 4))
        recall_list.append(round(recall, 4))
        f1_list.append(round(f1, 4))
        auc_list.append(round(auc, 4))

    # Visualizziamo le performance sotto forma di tabella
    scores = pd.DataFrame({"Accuracy": accuracy_list, "Precision": precision_list, "Recall": recall_list,
                           "F1": f1_list, "AUC": auc_list}, index=models.keys())
    print("\n", "-*-" * 10, "Risultati dei modelli", "-*-" * 10, "\n")
    print(scores)

    # Salviamo i risultati su un file csv
    scores.to_csv(filename)


def create_tuned_random_forest(X_train, y_train):
    # Creiamo il modello
    model = RandomForestClassifier(random_state=25)

    # Definiamo i parametri da testare
    param_grid = {
        "n_estimators": [1, 3, 5, 7, 9, 11, 13],
        "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "min_samples_split": [2, 6, 10],
        "min_samples_leaf": [1, 3, 5],
        "bootstrap": [True, False]
    }

    # Creiamo il GridSearchCV
    tuned_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring="accuracy", n_jobs=-1, cv=5, refit="True", verbose=2)
    tuned_model.fit(X_train, y_train)

    print("Parametri ottimi per random forest: \n", tuned_model.best_params_)

    return tuned_model


def create_tuned_svm(X_train, y_train):
    # Creiamo il modello
    model = SVC()

    # Definiamo i parametri da testare
    param_grid = {
        "C": [0.1, 1, 10, 100, 1000],
        "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
        "kernel": ["rbf"]
    }

    # Creiamo il GridSearchCV
    tuned_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring="accuracy", n_jobs=-1, cv=5, refit="True", verbose=2)
    tuned_model.fit(X_train, y_train)

    print("Parametri ottimi per SVM: \n", tuned_model.best_params_)

    return tuned_model


def create_tuned_xgboost(X_train, y_train):
    # Creiamo il modello
    model = xgb.XGBClassifier(objective="binary:logistic")

    # Definiamo i parametri da testare
    param_grid = {
        "min_child_weight": [1, 5, 10],
        "gamma": [0.5, 1, 1.5, 2, 5],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "max_depth": [3, 4, 5]
    }

    # Creiamo il GridSearchCV
    tuned_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring="accuracy", n_jobs=-1, cv=5, refit="True", verbose=2)
    tuned_model.fit(X_train, y_train)

    print("Parametri ottimi per XGBoost: \n", tuned_model.best_params_)

    return tuned_model


# Questa funzione è di fatto identica a create_neural_network, con la differenza che aggiunge
# il training del modello. Questo è stato fatto per coerenza con le altre funzioni dei modelli tunati
def create_tuned_neural_network(X_train, y_train):

    model = Sequential()
    model.add(Dense(32, activation="relu", kernel_initializer="normal", input_dim=8))
    model.add(Dense(16, activation="relu", kernel_initializer="normal", input_dim=32))
    model.add(Dense(8, activation="relu", kernel_initializer="normal", input_dim=16))
    model.add(Dense(1, activation="sigmoid"))

    # Compiliamo il modello
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=100, batch_size=64)

    return model


def save_models(models):
    for key in models.keys():
        with open("models/" + key + ".pkl", "wb") as file:
            pickle.dump(models[key], file)


if __name__ == "__main__":

    # Importiamo il dataset
    diabetes_dataset = pd.read_csv("diabetes.csv")

    # Andiamo a vedere le prime informazioni sul dataset
    dataset_informations(diabetes_dataset)

    # Andiamo a cercare i valori mancanti
    find_missing_values(diabetes_dataset)

    # Andiamo a sostituire i valori mancanti attraverso imputazione statistica
    fix_missing_values(diabetes_dataset)

    # Controlliamo se abbiamo righe duplicate
    find_duplicated_rows(diabetes_dataset)

    # Controlliamo se il dataset è bilanciato
    is_dataset_balanced(diabetes_dataset)

    # Bilanciamo il dataset
    diabetes_dataset_balanced = dataset_balancement(diabetes_dataset)

    # Visualizziamo la correlazione tra le feature e l'outcome attraverso un pairplot
    generate_pairplot(diabetes_dataset_balanced)

    # Visualizziamo la correlazione tra le feature attraverso una heatmap
    generate_heatmap(diabetes_dataset_balanced)

    # Salviamo il dataset corretto
    diabetes_dataset_balanced.to_csv("diabetes_corrected.csv", index=False)

    # Normalizziamo le feature usando uno StandardScaler e prepariamo i dati per lo split
    X, y = scale_dataset(diabetes_dataset_balanced)

    # Dividiamo il dataset in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Ora possiamo procedere con la creazione dei modelli

    # Testeremo i seguenti modelli: Random Forest, Support Vector Machine, XGBoost e una rete neurale
    models = {
        "Random Forest": RandomForestClassifier(random_state=25),
        "Support Vector Machine": SVC(),
        "XGBoost": xgb.XGBClassifier(objective="binary:logistic"),
        "Neural Network": create_neural_network()
    }

    # Andiamo a fare il training dei modelli
    train_models(models, X_train, y_train)

    # Andiamo a fare il testing dei modelli
    test_models(models, X_test, y_test)

    # Andiamo ora a fare il fine-tuning dei modelli e a testarli nuovamente
    # tuned_models = {
    #     "Tuned Random Forest": create_tuned_random_forest(X_train, y_train),
    #     "Tuned Support Vector Machine": create_tuned_svm(X_train, y_train),
    #     "Tuned XGBoost": create_tuned_xgboost(X_train, y_train),
    #     "Tuned Neural Network": create_tuned_neural_network(X_train, y_train)
    # }
    # test_models(tuned_models, X_test, y_test, filename="tuned_scores.csv")

    # Questa operazione si è rivelata inutile, in quanto i modelli non hanno migliorato le proprie performance

    # Andiamo infine a salvare i modelli
    save_models(models)
