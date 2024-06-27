import os
import pandas as pd
import numpy as np
import multiprocessing
import pickle
import random as rn
from anchor import anchor_tabular

# Settiamo il seed per la riproducibilità
os.environ["PYTHONHASHSEED"] = "25"
np.random.seed(25)
rn.seed(25)

# Per evitare problemi con la libreria tensorflow, disabilitiamo le ottimizzazioni per DNN e settiamo il livello di log a 1
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Definizione il numero di thread per la parallelizzazione
THREAD_NO = 5


def load_models(models_path):
    for model_name, model_path in models_path.items():
        with open(model_path, "rb") as f:
            models[model_name] = pickle.load(f)


def get_row_to_explain(dataset, scaler, target_row_index):
    # Otteniamo la riga target e la trasformiamo in un Dataframe di una sola riga, per poi rimuovere l'Outcome
    row_to_explain = dataset.iloc[target_row_index]
    row_to_explain_df = row_to_explain.to_frame().T
    row_to_explain_df = row_to_explain_df.drop(["Outcome"], axis=1)

    # Scaliamo la riga con lo scaler, in quanto i modelli sono stati addestrati sul dataset scalato
    # Inoltre abbiamo bisogno sia della riga sotto forma di array numpy che di Dataframe, in quanto
    # i vari algoritmi potrebbero richiedere o uno o l'altro.
    scaled_row_array = scaler.transform(row_to_explain_df)
    scaled_row_df = pd.DataFrame(scaled_row_array, columns=dataset.columns[:-1])
    return scaled_row_array, scaled_row_df


def anchor_precision_and_coverage_test(models, diabetes_dataset, X_scaled, scaled_row_array, precision_test, coverage_test):

    # Definiamo una funzione wrapper per la Rete Neurale, per impostare la verbosità a 0 e rispettare la firma
    # di anchor, che richiede che la previsione sia un array unidimensionale (quindi dobbiamo fare il flatten)
    def flatten_prediction(scaled_row):
        result = models["Neural Network"].predict(scaled_row, verbose=0)
        return (result > 0.5).astype(int).flatten()

    for model in models.keys():
        # Creiamo l'explainer e otteniamo la spiegazione
        explainer = anchor_tabular.AnchorTabularExplainer(
            class_names=["No Diabetes", "Diabetes"],
            feature_names=diabetes_dataset.columns[:-1],
            train_data=X_scaled,
            categorical_names={}
        )

        if model == "Neural Network":
            explanation = explainer.explain_instance(scaled_row_array.flatten(), flatten_prediction, threshold=0.95)
        else:
            explanation = explainer.explain_instance(scaled_row_array.flatten(), models[model].predict, threshold=0.95)

        # Inseriamo precision e coverage per studiarle
        precision_test[model].append(explanation.precision())
        coverage_test[model].append(explanation.coverage())


def multithreading_test(i, models, diabetes_dataset, scaler, X_scaled, precision_test, coverage_test):
    # Definiamo le righe che ogni thread deve processare
    thread_range = range(i * int(1000 / THREAD_NO), (i + 1) * int(1000 / THREAD_NO))

    for i in thread_range:
        scaled_row_array, scaled_row_df = get_row_to_explain(diabetes_dataset, scaler, i)
        anchor_precision_and_coverage_test(models, diabetes_dataset, X_scaled, scaled_row_array, precision_test, coverage_test)

    return precision_test, coverage_test


if __name__ == "__main__":

    # Inizializiamo i dizionari per i risultati
    precision_test = {
        "Random Forest": [],
        "SVM": [],
        "XGBoost": [],
        "Neural Network": []
    }
    coverage_test = {
        "Random Forest": [],
        "SVM": [],
        "XGBoost": [],
        "Neural Network": []
    }

    # Importiamo gli oggetti necessari per la valutazione
    models = {
        "Random Forest": "models/Random Forest.pkl",
        "SVM": "models/Support Vector Machine.pkl",
        "XGBoost": "models/XGBoost.pkl",
        "Neural Network": "models/Neural Network.pkl"
    }
    load_models(models)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    diabetes_dataset = pd.read_csv("diabetes_corrected.csv")
    X_scaled = scaler.transform(diabetes_dataset.drop(["Outcome"], axis=1))

    # Carichiamo i risultati precedenti, se esistono
    with open("results.pkl", "rb") as f:
        result = pickle.load(f)

    # Se non esistono, eseguiamo il codice parallelizzato per produrre i risultati
    if result is None:
        with multiprocessing.Pool(THREAD_NO) as p:
            # Ritorna una lista di tuple, in ogni tupla ci sono due dizionari con quattro campi, ogni campo è una lista
            result = p.starmap(multithreading_test, [(0, models, diabetes_dataset, scaler, X_scaled, precision_test, coverage_test),
                                                     (1, models, diabetes_dataset, scaler, X_scaled, precision_test, coverage_test),
                                                     (2, models, diabetes_dataset, scaler, X_scaled, precision_test, coverage_test),
                                                     (3, models, diabetes_dataset, scaler, X_scaled, precision_test, coverage_test),
                                                     (4, models, diabetes_dataset, scaler, X_scaled, precision_test, coverage_test)])
        # Salviamo i risultati su file
        with open("results.pkl", "wb") as file:
            pickle.dump(result, file)
    # Altrimenti, processiamo i risultati per ottenere le medie
    else:
        # Ricomponiamo i risultati dei vari thread in un unico dizionario
        precision_test["Random Forest"] = result[0][0]["Random Forest"] + result[1][0]["Random Forest"] + result[2][0][
            "Random Forest"] + result[3][0]["Random Forest"] + result[4][0]["Random Forest"]
        coverage_test["Random Forest"] = result[0][1]["Random Forest"] + result[1][1]["Random Forest"] + result[2][1][
            "Random Forest"] + result[3][1]["Random Forest"] + result[4][1]["Random Forest"]

        precision_test["SVM"] = result[0][0]["SVM"] + result[1][0]["SVM"] + result[2][0]["SVM"] + result[3][0]["SVM"] + \
                                result[4][0]["SVM"]
        coverage_test["SVM"] = result[0][1]["SVM"] + result[1][1]["SVM"] + result[2][1]["SVM"] + result[3][1]["SVM"] + \
                               result[4][1]["SVM"]

        precision_test["XGBoost"] = result[0][0]["XGBoost"] + result[1][0]["XGBoost"] + result[2][0]["XGBoost"] + \
                                    result[3][0]["XGBoost"] + result[4][0]["XGBoost"]
        coverage_test["XGBoost"] = result[0][1]["XGBoost"] + result[1][1]["XGBoost"] + result[2][1]["XGBoost"] + \
                                   result[3][1]["XGBoost"] + result[4][1]["XGBoost"]

        precision_test["Neural Network"] = result[0][0]["Neural Network"] + result[1][0]["Neural Network"] + \
                                           result[2][0]["Neural Network"] + result[3][0]["Neural Network"] + \
                                           result[4][0]["Neural Network"]
        coverage_test["Neural Network"] = result[0][1]["Neural Network"] + result[1][1]["Neural Network"] + \
                                          result[2][1]["Neural Network"] + result[3][1]["Neural Network"] + \
                                          result[4][1]["Neural Network"]

        # Calcoliamo le medie
        precision_test_means = {
            "Random Forest": np.round(np.mean(precision_test["Random Forest"]), 3),
            "SVM": np.round(np.mean(precision_test["SVM"]), 3),
            "XGBoost": np.round(np.mean(precision_test["XGBoost"]), 3),
            "Neural Network": np.round(np.mean(precision_test["Neural Network"]), 3)
        }

        coverage_test_means = {
            "Random Forest": np.round(np.mean(coverage_test["Random Forest"]), 3),
            "SVM": np.round(np.mean(coverage_test["SVM"]), 3),
            "XGBoost": np.round(np.mean(coverage_test["XGBoost"]), 3),
            "Neural Network": np.round(np.mean(coverage_test["Neural Network"]), 3)
        }

        # Creiamo un dataframe con i risultati e lo mostriamo, dopodiché lo salviamo su file
        results_df = pd.DataFrame([precision_test_means, coverage_test_means], index=["Precision", "Coverage"])
        print(results_df)
        results_df.to_csv("explanation_result.csv")

        # Da questi risultati si evince che SVM e Neural Network sono i modelli che hanno una spiegabilità maggiore
        # con Anchor. Tra i due, SVM ha una coverage leggermente migliore (Circa il 10% rispetto alla Neural Network),
        # e visto che le metriche di valutazione sono molto simili (in particolare la Recall è uguale), si può affermare
        # che SVM è il modello migliore per spiegare le previsioni con Anchor.
