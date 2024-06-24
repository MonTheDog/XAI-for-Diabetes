import warnings
import os
import pandas as pd
import numpy as np
import shap
import pickle
import random as rn
import lime
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
# Cambiamo le impostazioni di pandas per visualizzare tutte le colonne
pd.set_option('display.max_columns', None)
# Definiamo come formattare i numeri nei dataframe, senza notazione esponenziale
pd.set_option('display.float_format', lambda x: '%.8f' % x)


def load_models(models_path):
    # Carichiamo i modelli
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
    print("\n", "-*-" * 10, "Risultato effettivo", "-*-" * 10, "\n")
    if diabetes_value == 0:
        print("La persona non ha il diabete")
    else:
        print("La persona ha il diabete")


def make_predictions(models):
    print("\n", "-*-" * 10, "Risultati dei Modelli", "-*-" * 10, "\n")

    predictions = dict()
    for model in models.keys():

        # Effettuiamo una distinzione con la rete neurale per il suo diverso funzionamento, infatti
        # non vogliamo la probabilità, ma il valore della classe, e per evitare di flooddare la console
        # andiamo a impsotare la verbosità a 0
        if model == "Neural Network":
            predictions[model] = models[model].predict(scaled_row_array, verbose=0)
            predictions[model] = (predictions[model] > 0.5).astype(int)
        else:
            predictions[model] = models[model].predict(scaled_row_array)

        # Mostriamo se ogni modello ha indovinato la previsione a fini di studio
        if predictions[model] == diabetes_value:
            print(f"Il modello {model} ha predetto correttamente")
        else:
            print(f"Il modello {model} ha predetto in modo errato")

    return predictions


def shap_explanation(models, explainability_score, X_scaled, scaled_row_df):
    # TODO rivedere
    # I valori sono da considerarsi additivi verso una delle due previsioni (Se negativi tenderanno verso lo classe
    # negativa, altrimenti verso quella positiva). Il valore assoluto rappresenterà quanto una colonna pesa
    # nella previsione finale
    for model in models.keys():
        # Gestiamo ogni modello singolarmente, poiché SHAP funziona diversamente in base al classificatore usato

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
            shap_values = explainer.shap_values(scaled_row_df)

        if model == "XGBoost":
            explainer = shap.TreeExplainer(models[model])
            shap_values = explainer.shap_values(scaled_row_df)

        # Nel caso della rete Neurale dobbiamo selezionare il primo indice del tensore, che contiene gli
        # SHAP values effettivi
        if model == "Neural Network":
            explainer = shap.KernelExplainer(models[model], X_scaled)
            shap_values = explainer.shap_values(scaled_row_df)
            shap_values = shap_values[:, :, 0]

        # Aggiungiamo la riga al dataframe
        explainability_score.loc[model + " SHAP"] = shap_values.flatten()


def lime_explanation(models, explainability_score, diabetes_dataset, X_scaled, scaled_row_array, predictions):
    for model in models.keys():
        explainer = lime.lime_tabular.LimeTabularExplainer(X_scaled, mode="regression")
        explanation = explainer.explain_instance(scaled_row_array.flatten(), models[model].predict, num_features=8)

        # Convertiamo gli indici nei nomi delle colonne
        feature_names = diabetes_dataset.columns[:-1]
        mapped_explanation = dict()
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

        # Ordiniamo le colonne per come devono essere inserite nel dataset
        sorted_row = []
        for column in feature_names:
            for feature in mapped_explanation.keys():
                if column == feature:
                    sorted_row.append(mapped_explanation[feature])

        # Aggiungiamo la riga al dataframe
        explainability_score.loc[model + " LIME"] = sorted_row

def ciu_explanation(models, explainability_score, dataframe_scaled, scaled_row_df, predictions):
    for model in models.keys():
        if model == "Neural Network":
            CIU = ciu.CIU(models[model].predict, ["Diabetes"], data=dataframe_scaled)
            CIU_explanation = CIU.explain(scaled_row_df)
        else:
            CIU = ciu.CIU(models[model].predict_proba, ["No Diabetes", "Diabetes"], data=dataframe_scaled)
            CIU_explanation = CIU.explain(scaled_row_df, output_inds=predictions[model])

        explainability_score.loc[model + " CI"] = CIU_explanation.T[0:1].values.flatten()
        explainability_score.loc[model + " CU"] = CIU_explanation.T[1:2].values.flatten()

def anchor_explanation(models, diabetes_dataset, scaler, X_scaled, scaled_row_array):

    # Wrapper per ottenere la previsione per la Rete Neurale, così da rispettare la firma della funzione di Anchor
    def flatten_prediction(scaled_row):
        result = models["Neural Network"].predict(scaled_row, verbose=0)
        return (result > 0.5).astype(int).flatten()

    for model in models.keys():
        explainer = anchor_tabular.AnchorTabularExplainer(
            class_names=["No Diabetes", "Diabetes"],
            feature_names=diabetes_dataset.columns[:-1],
            train_data=X_scaled,
            categorical_names={}
        )

        if model == "Neural Network":
            # Perché serve un array unidimensionale e la Neural Network restituisce un array bidimensionale
            explanation = explainer.explain_instance(scaled_row_array.flatten(), flatten_prediction, threshold=0.95)
        else:
            explanation = explainer.explain_instance(scaled_row_array.flatten(), models[model].predict, threshold=0.95)

        # Riscaliamo i risultati
        inversed_values = pd.DataFrame(np.zeros(shape=(1, 8)), columns=diabetes_dataset.columns[:-1])

        value_map = {}
        symbol_map = {}

        for string in explanation.names():
            split = string.split()
            value_map[split[0]] = split[2]
            symbol_map[split[0]] = split[1]

        for column in diabetes_dataset.columns[:-1]:
            if column in value_map.keys():
                inversed_values[column] = value_map[column]

        actual_values_array = scaler.inverse_transform(inversed_values)
        actual_values = pd.DataFrame(actual_values_array, columns=diabetes_dataset.columns[:-1])

        rules_list = []
        for key in value_map.keys():
            rule = key + " " + symbol_map[key] + " " + str(np.round(actual_values.loc[0, key], 3))
            rules_list.append(rule)

        print(model)
        print('Anchor: %s' % (' AND '.join(rules_list)))
        print('Precision: %.2f' % explanation.precision())
        print('Coverage: %.2f' % explanation.coverage())


if __name__ == '__main__':

    models = {
        "Random Forest": "models/Random Forest.pkl",
        "SVM": "models/Support Vector Machine.pkl",
        "XGBoost": "models/XGBoost.pkl",
        "Neural Network": "models/Neural Network.pkl"
    }

    # Carichiamo i modelli
    load_models(models)

    # Carichiamo lo scaler
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Carichiamo il dataset e selezioniamo l'indice della riga che vogliamo spiegare
    target_row = 1
    diabetes_dataset = pd.read_csv("diabetes_corrected.csv")

    # Poichè abbiamo bisogno sia di un array numpy che del dataframe scalato li costruiamo
    X_scaled = scaler.transform(diabetes_dataset.drop(["Outcome"], axis=1))
    dataframe_scaled = pd.DataFrame(X_scaled, columns=diabetes_dataset.columns[:-1])

    # Prendiamo la riga da spiegare, sotto forma di array e di dataframe
    scaled_row_array, scaled_row_df = get_row_to_explain(diabetes_dataset, target_row)

    # Mostriamo se la persona ha o meno il diabete a fini di studio
    diabetes_value = diabetes_dataset["Outcome"][target_row]
    has_diabetes(diabetes_value)

    # Creiamo un dataframe vuoto per inserire poi i valori di explainability
    # E' bene specificare che ogni valore va interpretato sulla base della tecnica utilizzata, quindi un valore
    # di una colonna non significa intrinsecamente nulla se non viene rapportato al nome della riga, che indica
    # il modello e la tecnica utilizzata. La scelta di inserire queste informazioni in un dataframe è stata fatta
    # prettamente per comodità e non per indicare una omogeneità dei dati.
    explainability_score = pd.DataFrame(columns=diabetes_dataset.columns[:-1])

    # Effettuiamo la previsione per ogni modello
    predictions = make_predictions(models)

    print("SHAP in progress")
    # Otteniamo l'explanation con SHAP
    shap_explanation(models, explainability_score, X_scaled, scaled_row_df)

    print("LIME in progress")
    # Otteniamo l'explanation con LIME
    lime_explanation(models, explainability_score, diabetes_dataset, X_scaled, scaled_row_array, predictions)

    print("CIU in progress")
    # Otteniamo l'explanation con CIU
    ciu_explanation(models, explainability_score, dataframe_scaled, scaled_row_df, predictions)

    print("ANCHOR in progress")
    # Otteniamo l'explanation con Anchor
    anchor_explanation(models, diabetes_dataset, scaler, X_scaled, scaled_row_array)

    print(explainability_score)
