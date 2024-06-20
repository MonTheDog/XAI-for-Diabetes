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



if __name__ == '__main__':

    models = {
        "Random Forest": "models/Random Forest.pkl",
        "SVM": "models/Support Vector Machine.pkl",
        "XGBoost": "models/XGBoost.pkl",
        "Neural Network": "models/Neural Network.pkl"
    }

    # Carichiamo i modelli
    for model_name, model_path in models.items():
        with open(model_path, "rb") as f:
            models[model_name] = pickle.load(f)

    # Carichiamo lo scaler
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Carichiamo il dataset e prendiamone una riga
    target_row = 1

    diabetes_dataset = pd.read_csv("diabetes.csv")
    X_scaled = scaler.transform(diabetes_dataset.drop(["Outcome"], axis=1))
    y = diabetes_dataset["Outcome"]
    row_to_explain = diabetes_dataset.iloc[target_row]
    row_to_explain_df = row_to_explain.to_frame().T
    row_to_explain_df = row_to_explain_df.drop(["Outcome"], axis=1)
    print(row_to_explain_df)
    scaled_row = scaler.transform(row_to_explain_df)
    scaled_row_df = pd.DataFrame(scaled_row, columns=diabetes_dataset.columns[:-1])
    print(scaled_row_df)

    if y[target_row] == 0:
        print("La persona non ha il diabete")
    else:
        print("La persona ha il diabete")

    # Creiamo un dataframe vuoto per inserire poi i valori di explainability
    explainability_score = pd.DataFrame(columns=diabetes_dataset.columns[:-1])
    predictions = dict()

    # Effettuiamo la previsione per ogni modello
    for model in models.keys():

        if model == "Neural Network":
            predictions[model] = models[model].predict(scaled_row, verbose=0)
        else:
            predictions[model] = models[model].predict(scaled_row)

        if model == "Neural Network":
            predictions[model] = (predictions[model] > 0.5).astype(int)

        is_prediction_correct = predictions[model] == y[target_row]

        if is_prediction_correct:
            print(f"Il modello {model} ha predetto correttamente")
        else:
            print(f"Il modello {model} ha predetto in modo errato")

    # Otteniamo l'explanation con SHAP
    for model in models.keys():

        if model == "Random Forest":
            explainer = shap.TreeExplainer(models[model])
            shap_values = explainer(scaled_row_df)
            if predictions[model] == 0:
                shap_values = shap_values[:,:,0].values
            else:
                shap_values = shap_values[:,:,1].values

        if model == "SVM":
            explainer = shap.KernelExplainer(models[model].predict, X_scaled)
            shap_values = explainer.shap_values(scaled_row_df)

        if model == "XGBoost":
            explainer = shap.TreeExplainer(models[model])
            shap_values = explainer.shap_values(scaled_row_df)

        if model == "Neural Network":
            explainer = shap.KernelExplainer(models[model], X_scaled)
            shap_values = explainer.shap_values(scaled_row_df)
            shap_values = shap_values[:,:,0]

        # Aggiungiamo la riga al dataframe
        explainability_score.loc[model + " SHAP"] = shap_values.flatten()

    # Otteniamo l'explanation con LIME
    for model in models.keys():
        explainer = lime.lime_tabular.LimeTabularExplainer(X_scaled, mode="regression")

        explanation = explainer.explain_instance(scaled_row.flatten(), models[model].predict, num_features=8)

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

    # Ricostruiamo il dataframe ma scalato
    dataframe_scaled = pd.DataFrame(X_scaled, columns=diabetes_dataset.columns[:-1])

    # Otteniamo l'explanation con CIU
    for model in models.keys():
        if model == "Neural Network":
            CIU = ciu.CIU(models[model].predict, ["Diabetes"], data=dataframe_scaled)
            CIU_explanation = CIU.explain(scaled_row_df)
        else:
            CIU = ciu.CIU(models[model].predict_proba, ["No Diabetes","Diabetes"], data=dataframe_scaled)
            CIU_explanation = CIU.explain(scaled_row_df, output_inds=predictions[model])

        explainability_score.loc[model + " CI"] = CIU_explanation.T[0:1].values.flatten()
        explainability_score.loc[model + " CU"] = CIU_explanation.T[1:2].values.flatten()

    # Wrapper per ottenere la previsione per la Rete Neurale, così da rispettare la firma della funzione di Anchor
    def flatten_prediction(scaled_row):
        result = models["Neural Network"].predict(scaled_row, verbose=0)
        return (result > 0.5).astype(int).flatten()

    # Otteniamo l'explanation con Anchor
    for model in models.keys():
        explainer = anchor_tabular.AnchorTabularExplainer(
            class_names=["No Diabetes", "Diabetes"],
            feature_names=diabetes_dataset.columns[:-1],
            train_data=X_scaled,
            categorical_names={}
        )

        if model == "Neural Network":
            # Perché serve un array unidimensionale e la Neural Network restituisce un array bidimensionale
            explanation = explainer.explain_instance(scaled_row.flatten(), flatten_prediction, threshold=0.95)
        else:
            explanation = explainer.explain_instance(scaled_row.flatten(), models[model].predict, threshold=0.95)

        print(model)
        print('Anchor: %s' % (' AND '.join(explanation.names())))
        print('Precision: %.2f' % explanation.precision())
        print('Coverage: %.2f' % explanation.coverage())

    print(explainability_score)


    # inversed_scale = scaler.inverse_transform(dataframe_scaled)
    # inversed_dataframe = pd.DataFrame(inversed_scale, columns=diabetes_dataset.columns[:-1])



