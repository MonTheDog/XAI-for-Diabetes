import warnings
import os
import pandas as pd
import numpy as np
import shap
import pickle
import random as rn
import logging
from lime import lime_tabular
import ciu
from anchor import anchor_tabular

# Settiamo il seed per la riproducibilità
os.environ["PYTHONHASHSEED"] = "25"
np.random.seed(25)
rn.seed(25)

# Per evitare problemi con la libreria tensorflow, disabilitiamo le ottimizzazioni per DNN e settiamo il livello di log a 1
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Ignoriamo i warning
warnings.filterwarnings("ignore")
logging.getLogger("shap").setLevel(logging.CRITICAL)

# Cambiamo le impostazioni di pandas per visualizzare tutte le colonne
pd.set_option('display.max_columns', None)

# Definiamo come formattare i numeri nei dataframe, senza notazione esponenziale
pd.set_option('display.float_format', lambda x: '%.8f' % x)


def load_models(models_path):
    for model_name, model_path in models_path.items():
        with open(model_path, "rb") as f:
            models[model_name] = pickle.load(f)


def get_row_to_explain(dataset, target_row_index):
    # Otteniamo la riga target e la trasformiamo in un Dataframe di una sola riga, per poi rimuovere l'Outcome
    row_to_explain = dataset.iloc[target_row_index]
    row_to_explain_df = row_to_explain.to_frame().T
    row_to_explain_df = row_to_explain_df.drop(["Outcome"], axis=1)

    # Mostriamo la riga selezionata dall'utente
    print("\n", "-*-" * 10, "Riga Selezionata", "-*-" * 10, "\n")
    print(row_to_explain_df)

    # Scaliamo la riga con lo scaler, in quanto i modelli sono stati addestrati sul dataset scalato
    # Inoltre abbiamo bisogno sia della riga sotto forma di array numpy che di Dataframe, in quanto
    # i vari algoritmi potrebbero richiedere o uno o l'altro.
    scaled_row_array = scaler.transform(row_to_explain_df)
    scaled_row_df = pd.DataFrame(scaled_row_array, columns=dataset.columns[:-1])
    return scaled_row_array, scaled_row_df


def has_diabetes(diabetes_value):
    print("\n", "-*-" * 10, "Diagnosi Reale", "-*-" * 10, "\n")
    if diabetes_value == 0:
        print("La persona non ha il diabete")
    else:
        print("La persona ha il diabete")


def make_predictions(models):
    print("\n", "-*-" * 10, "Risultati dei Modelli", "-*-" * 10, "\n")

    predictions = dict()

    for model in models.keys():

        # Effettuiamo una distinzione con la rete neurale per il suo diverso funzionamento, infatti
        # non vogliamo la probabilità, ma la classe
        if model == "Neural Network":
            predictions[model] = models[model].predict(scaled_row_array, verbose=0)
            predictions[model] = (predictions[model] > 0.5).astype(int)
        else:
            predictions[model] = models[model].predict(scaled_row_array)

        # Mostriamo se ogni modello ha indovinato la previsione
        if predictions[model] == diabetes_value:
            print(f"Il modello {model} ha predetto correttamente")
        else:
            print(f"Il modello {model} ha predetto in modo errato")

    return predictions


def shap_explanation(models, explainability_score, X_scaled, scaled_row_df):
    # I valori SHAP sono da considerarsi additivi verso una delle due previsioni in un problema di classificazione binaria:
    # se negativi tenderanno verso lo classe negativa (assenza di diabete), altrimenti verso quella positiva (diabete).
    # Il valore assoluto rappresenterà quanto una colonna pesa nella previsione finale
    # SHAP eccelle nella spiegabilità globale di un modello, ma in questo studio siamo principalmente interessati
    # alla spiegabilità locale.

    for model in models.keys():
        # Gestiamo ogni modello singolarmente, poiché SHAP funziona diversamente in base al classificatore usato

        shap_values = None

        # Nel caso di Random Forest usiamo un Tree Explainer e prendiamo la riga degli shap_values corrispondente
        # alla previsione (SHAP di default le restituisce entrambe, ma sono una l'opposto dell'altra)
        if model == "Random Forest":
            explainer = shap.TreeExplainer(models[model])
            shap_values = explainer(scaled_row_df)
            if predictions[model] == 0:
                shap_values = shap_values[:, :, 0].values
            else:
                shap_values = shap_values[:, :, 1].values

        # Nel caso di SVM e XGBoost c'è poco da fare se non usare la sottoclasse adeguata dell'Explainer
        if model == "SVM":
            explainer = shap.KernelExplainer(models[model].predict, X_scaled)
            shap_values = explainer.shap_values(scaled_row_df, silent=True)

        if model == "XGBoost":
            explainer = shap.TreeExplainer(models[model])
            shap_values = explainer.shap_values(scaled_row_df)

        # Nel caso della rete Neurale dobbiamo selezionare il primo indice del tensore, che contiene gli SHAP values
        if model == "Neural Network":
            explainer = shap.KernelExplainer(models[model], X_scaled)
            shap_values = explainer.shap_values(scaled_row_df, silent=True)
            shap_values = shap_values[:, :, 0]

        # Aggiungiamo la riga al dataframe
        explainability_score.loc[model + " SHAP"] = shap_values.flatten()

        # Interazione UI per mostrare il progresso
        print(".", end="")


def lime_explanation(models, explainability_score, diabetes_dataset, X_scaled, scaled_row_array, predictions):
    # I valori LIME misurano il peso di una caratteristica all'interno di una previsione. Se il segno è positivo allora
    # la caratteristica aumenta la probabilità della classe predetta, altrimenti se il segno è negativo aumenta la
    # probabilità della classe opposta. Il valore assoluto rappresenta quanto la caratteristica pesa nella previsione.

    # Definiamo una funzione wrapper per la Rete Neurale, per impostare la verbosità a 0 così da evitare di riempire la console
    def non_verbose_prediction(scaled_row):
        result = models["Neural Network"].predict(scaled_row, verbose=0)
        return result

    for model in models.keys():
        # Creiamo l'oggetto explainer di LIME
        explainer = lime_tabular.LimeTabularExplainer(X_scaled, mode="regression")

        # Andiamo a spiegare la riga per ottenere i valori di LIME
        # Se usiamo la Rete Neurale passiamo la funzione wrapper, altrimenti usiamo la predict normale
        if model == "Neural Network":
            explanation = explainer.explain_instance(scaled_row_array.flatten(), non_verbose_prediction, num_features=8)
        else:
            explanation = explainer.explain_instance(scaled_row_array.flatten(), models[model].predict, num_features=8)

        # Poiché LIME restituisce le spiegazioni in un formato poco leggibile e comprensibile (utilizza degli indici
        # numerati al posto dei nomi delle feature e mette in disordine le colonne), mappiamo i valori in un dizionario ordinato

        feature_names = diabetes_dataset.columns[:-1]
        mapped_explanation = dict()

        # Per prima cosa convertiamo gli indici nei nomi delle colonne
        if predictions[model] == 0:
            for index, feature in enumerate(feature_names):
                for column_index in explanation.local_exp[0]:
                    if index == column_index[0]:
                        mapped_explanation[feature] = column_index[1]
        else:
            for index, feature in enumerate(feature_names):
                for column_index in explanation.local_exp[1]:
                    if index == column_index[0]:
                        mapped_explanation[feature] = column_index[1]

        # Ordiniamo le colonne per come dovranno essere inserite nel dataset
        sorted_row = []
        for column in feature_names:
            for feature in mapped_explanation.keys():
                if column == feature:
                    sorted_row.append(mapped_explanation[feature])

        # Aggiungiamo la riga al dataframe
        explainability_score.loc[model + " LIME"] = sorted_row

        # Interazione UI per mostrare il progresso
        print(".", end="")


def ciu_explanation(models, explainability_score, dataframe_scaled, scaled_row_df, predictions):
    # In CIU abbiamo due valori: CI e CU
    # CI è la Contextual Importance, e misura l'importanza relativa di una caratteristica nel contesto di una particolare
    # istanza. Un valore CI più alto indica che la caratteristica è importante per la previsione di quella istanza.
    # CU è la Contextual Utility, e misura l'utilità della caratteristica nel contesto di una particolare istanza. Valuta
    # quindi quanto il valore della caratteristica sia favorevole o sfavorevole per la classe predetta

    # Definiamo una funzione wrapper per la Rete Neurale, per impostare la verbosità a 0 così da evitare di riempire la console
    def non_verbose_prediction(scaled_row):
        result = models["Neural Network"].predict(scaled_row, verbose=0)
        return result

    for model in models.keys():
        # Creiamo l'oggetto CIU e otteniamo la spiegazione
        # Se il modello è la Rete Neurale, dobbiamo passare la funzione wrapper, altrimenti passiamo la predict normale
        if model == "Neural Network":
            CIU = ciu.CIU(non_verbose_prediction, ["Diabetes"], data=dataframe_scaled)
            CIU_explanation = CIU.explain(scaled_row_df)
        else:
            CIU = ciu.CIU(models[model].predict_proba, ["No Diabetes", "Diabetes"], data=dataframe_scaled)
            CIU_explanation = CIU.explain(scaled_row_df, output_inds=predictions[model])

        # Formattiamo la spiegazione in due righe del dataframe, una per CI e una per CU
        explainability_score.loc[model + " CI"] = CIU_explanation.T[0:1].values.flatten()
        explainability_score.loc[model + " CU"] = CIU_explanation.T[1:2].values.flatten()

        # Interazione UI per mostrare il progresso
        print(".", end="")


def anchor_explanation(models, diabetes_dataset, scaler, X_scaled, scaled_row_array):
    # Anchor non ha dei valori numerici, ma permette di produrre delle regole che spiegano la previsione. Queste regole
    # vengono poi utilizzate su istanze "vicine" a quella spiegata, per controllare quanto siano affidabili, attraverso
    # la Precision (La percentuale di istanze vicine per cui le regole sono vere) e il Coverage (La percentuale di istanze
    # del dataset che sono coperte da quelle regole)

    anchor_result = ""

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

        # La spiegazione contiene però i valori scalati, quindi dobbiamo riportarli alla scala originale

        # Prepariamo un dataframe con tutti zero, dove inseriremo i valori da convertire
        scaled_values = pd.DataFrame(np.zeros(shape=(1, 8)), columns=diabetes_dataset.columns[:-1])

        # Prepariamo le HashMap per mappare i valori e il simbolo (maggiore, minore, uguale) per ogni colonna
        value_map = {}
        symbol_map = {}

        # Le seguenti variabili servono per i casi in cui abbiamo una regola sottoforma di range, quindi abbiamo due valori e due simboli
        is_auxiliary_needed = False
        auxiliary_symbol_map = {}
        auxiliary_value_map = {}
        auxiliary_scaled_values = pd.DataFrame(np.zeros(shape=(1, 8)), columns=diabetes_dataset.columns[:-1])
        auxiliary_actual_values = pd.DataFrame(np.zeros(shape=(1, 8)), columns=diabetes_dataset.columns[:-1])

        # Andiamo a splittare ogni regola Anchor per inserirla nel dataframe
        for string in explanation.names():
            split = string.split()
            # Se la regola è composta da tre elementi, allora è una regola normale, altrimenti è una regola con range
            if len(split) == 3:
                value_map[split[0]] = split[2]
                symbol_map[split[0]] = split[1]
            # Nel caso di una regola con range andiamo a inserire il simbolo e il valore aggiuntivo nelle mappe ausiliarie
            else:
                is_auxiliary_needed = True
                auxiliary_value_map[split[2]] = split[0]
                auxiliary_symbol_map[split[2]] = split[1]
                value_map[split[2]] = split[4]
                symbol_map[split[2]] = split[3]

        # Andiamo a inserire i valori scalati all'interno del dataframe, per poterli poi riportare alla scala originale
        for column in diabetes_dataset.columns[:-1]:
            if column in value_map.keys():
                scaled_values[column] = value_map[column]
            if column in auxiliary_value_map.keys():
                auxiliary_scaled_values[column] = auxiliary_value_map[column]

        # Riportiamo i valori scalati alla scala originale
        actual_values_array = scaler.inverse_transform(scaled_values)
        actual_values = pd.DataFrame(actual_values_array, columns=diabetes_dataset.columns[:-1])
        if is_auxiliary_needed:
            auxiliary_actual_values_array = scaler.inverse_transform(auxiliary_scaled_values)
            auxiliary_actual_values = pd.DataFrame(auxiliary_actual_values_array, columns=diabetes_dataset.columns[:-1])

        # Andiamo a comporre la lista di regole anchor con i valori alla scala originale
        rules_list = []
        for key in value_map.keys():
            # Caso in cui abbiamo una regola con range
            if key in auxiliary_value_map.keys():
                rule = (str(np.round(auxiliary_actual_values.loc[0, key], 3)) + " " + auxiliary_symbol_map[key]
                        + " " + key + " " + symbol_map[key] + " " + str(np.round(actual_values.loc[0, key], 3)))
            # Caso in cui abbiamo una regola normale
            else:
                rule = key + " " + symbol_map[key] + " " + str(np.round(actual_values.loc[0, key], 3))

            rules_list.append(rule)

        # Inseriamo il risultato del modello nel risultato finale
        anchor_result += "--- " + model + " ---" + "\n"
        anchor_result += 'Anchor: %s' % (' AND '.join(rules_list)) + "\n"
        anchor_result += 'Precision: %.2f' % explanation.precision() + "\n"
        anchor_result += 'Coverage: %.2f' % explanation.coverage() + "\n\n"

        print(".", end="")

    return anchor_result


if __name__ == '__main__':

    # Carichiamo i modelli
    models = {
        "Random Forest": "models/Random Forest.pkl",
        "SVM": "models/Support Vector Machine.pkl",
        "XGBoost": "models/XGBoost.pkl",
        "Neural Network": "models/Neural Network.pkl"
    }
    load_models(models)

    # Carichiamo lo scaler
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Carichiamo il dataset
    diabetes_dataset = pd.read_csv("diabetes_corrected.csv")

    # All'interno del codice necessitiamo dei dati scalati sia sottoforma di dataframe che sottoforma di array numpy,
    # quindi li costruiamo in questo momento
    X_scaled = scaler.transform(diabetes_dataset.drop(["Outcome"], axis=1))
    dataframe_scaled = pd.DataFrame(X_scaled, columns=diabetes_dataset.columns[:-1])

    # Iniziamo il ciclo di interazione con l'utente
    while True:

        # Chiediamo all'utente quale riga vuole spiegare (Righe interessanti = 1, 93)
        # 93 ha un errore di previsione da parte di tutti i modelli, ma i valori di explainability sono particolari
        # oltre che avere un Anchor particolarmente complesso
        target_row = int(input(f"Inserisci l'indice della riga da spiegare (0-{len(diabetes_dataset) - 1}) oppure "
                               f"un indice non valido per uscire: "))

        if target_row < 0 or target_row >= len(diabetes_dataset):
            break

        # Prendiamo la riga da spiegare, sotto forma di array e di dataframe
        scaled_row_array, scaled_row_df = get_row_to_explain(diabetes_dataset, target_row)

        # Mostriamo se la persona ha o meno il diabete
        diabetes_value = diabetes_dataset["Outcome"][target_row]
        has_diabetes(diabetes_value)

        # Creiamo un dataframe vuoto per inserire i valori di explainability
        # È bene specificare che ogni valore va interpretato sulla base della tecnica utilizzata, quindi un valore
        # di una colonna non significa intrinsecamente nulla se non viene rapportato al nome della riga, che indica
        # il modello e la tecnica utilizzata. La scelta di inserire queste informazioni in un dataframe è stata fatta
        # prettamente per comodità e non per indicare un' omogeneità dei dati.
        explainability_score = pd.DataFrame(columns=diabetes_dataset.columns[:-1])

        # Effettuiamo la previsione per ogni modello e individuiamo quali modelli hanno indovinato e quali hanno sbagliato
        predictions = make_predictions(models)

        print("\n", "-*-" * 10, "Computazione Explanations", "-*-" * 10, "\n")

        # Otteniamo l'explanation con SHAP
        print("SHAP in progress", end="")
        shap_explanation(models, explainability_score, X_scaled, scaled_row_df)
        print(" Done")

        # Otteniamo l'explanation con LIME
        print("LIME in progress", end="")
        lime_explanation(models, explainability_score, diabetes_dataset, X_scaled, scaled_row_array, predictions)
        print(" Done")

        # Otteniamo l'explanation con CIU
        print("CIU in progress", end="")
        ciu_explanation(models, explainability_score, dataframe_scaled, scaled_row_df, predictions)
        print(" Done")

        # Otteniamo l'explanation con Anchor
        print("ANCHOR in progress", end="")
        anchor_result = anchor_explanation(models, diabetes_dataset, scaler, X_scaled, scaled_row_array)
        print(" Done")

        print("\n", "-*-" * 10, "Risultati SHAP, LIME e CIU", "-*-" * 10, "\n")

        # Mostriamo i risultati dei primi tre algoritmi
        print(explainability_score)

        print("\n", "-*-" * 10, "Risultato Anchor", "-*-" * 10, "\n")

        # Mostriamo le regole generate da SHAP
        print(anchor_result)

