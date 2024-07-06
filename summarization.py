import pickle
import numpy as np
import pandas as pd
from openai import OpenAI
import streamlit as st
from anchor import anchor_tabular


def get_data():
    # Carichiamo il modello selezionato per la predizione e lo scaler
    with open("models/Support Vector Machine.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Carichiamo il dataset e generiamo i valori scalati
    diabetes_dataset = pd.read_csv("diabetes_corrected.csv")
    X_scaled = scaler.transform(diabetes_dataset.drop(["Outcome"], axis=1))

    return model, scaler, diabetes_dataset, X_scaled


def create_title():
    st.markdown("""
        <style>
        .parent-div {
            display: flex;
        }
        .div2 {
            padding-top: 24px;
            padding-left: 10px;
        }
        .color {
            color: #08a4a7;
        }
        .big-font {
            font-size:70px !important;
        }
        .small-font {
            font-size:20px !important;
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown(
        '<div class="parent-div"> <div> <span class="big-font">D<span class="color">IA </span> </span> </div> <div class="div2"> <div> '
        '<span class="small-font"> betes </span> </div> <div> <span class="small-font"> gnoser </span> </div> </div> </div>',
        unsafe_allow_html=True)


def create_form(model_input):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        help_txt = '''
                # Number of Pregnancies
                Suggested Value: (0 - 17)  
                Median Diabetic: 4  
                Median Non-Diabetic: 3
                '''
        model_input["Pregnancies"] = st.number_input("Pregnancies", value=model_input["Pregnancies"], help=help_txt,
                                                     min_value=0)

    with col2:
        help_txt = '''
                # Glucose Level
                Suggested Value: (44 - 199)  
                Median Diabetic: 142  
                Median Non-Diabetic: 110
                '''
        model_input["Glucose"] = st.number_input("Glucose", value=model_input["Glucose"], help=help_txt, min_value=0)

    with col3:
        help_text = '''
                # Blood Pressure
                Suggested Value: (24 - 122)  
                Median Diabetic: 75  
                Median Non-Diabetic: 71
                '''
        model_input["BloodPressure"] = st.number_input("BloodPressure", value=model_input["BloodPressure"],
                                                       help=help_text, min_value=0)

    with col4:
        help_text = '''
                # Skin Thickness
                Suggested Value: (7 - 99)  
                Median Diabetic: 32  
                Median Non-Diabetic: 27
                '''
        model_input["SkinThickness"] = st.number_input("SkinThickness", value=model_input["SkinThickness"],
                                                       help=help_text, min_value=0)

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        help_text = '''
                # Insulin Level
                Suggested Value: (14 - 846)  
                Median Diabetic: 164  
                Median Non-Diabetic: 128
                '''
        model_input["Insulin"] = st.number_input("Insulin", value=model_input["Insulin"], help=help_text, min_value=0)

    with col6:
        help_text = '''
                # BMI
                Suggested Value: (18.5 - 67.5)  
                Median Diabetic: 35.5  
                Median Non-Diabetic: 30.8
                '''
        model_input["BMI"] = st.number_input("BMI", value=model_input["BMI"], help=help_text, min_value=0.0, step=0.1, format="%.1f")

    with col7:
        help_text = '''
                # Diabetes Pedigree Function
                Suggested Value: (0.078 - 2.420)  
                Median Diabetic: 0.565  
                Median Non-Diabetic: 0.429
                '''
        model_input["DiabetesPedigreeFunction"] = st.number_input("DiabetesPedigreeFunc", value=model_input["DiabetesPedigreeFunction"],
                                                                  help=help_text, min_value=0.0, step=0.001,
                                                                  format="%.3f")

    with col8:
        help_text = '''
                # Age
                Suggested Value: (21 - 81)  
                Median Diabetic: 37  
                Median Non-Diabetic: 31
                '''
        model_input["Age"] = st.number_input("Age", value=model_input["Age"], help=help_text, min_value=0)


def create_submit(model_input, scaler, model, diabetes_dataset, X_scaled):
    prediction = None
    explanation = ""
    col9, col10 = st.columns([7, 1])
    with col10:
        if st.button("Submit"):
            # Prendiamo i dati inseriti e li scaliamo
            input_data = pd.DataFrame([model_input])
            scaled_input = scaler.transform(input_data)
            # Effettuiamo la previsione
            prediction = model.predict(scaled_input)
            # Otteniamo l'input da dare al LLM
            gpt_input = anchor(model, diabetes_dataset, scaler, X_scaled, scaled_input)
            # Usiamo il LLM per ottenere una spiegazione human-readable
            # TODO summarization using GPT
            explanation = gpt_input # explanation = ask_gpt(gpt_input)

    st.divider()

    return prediction, explanation


def anchor(model, diabetes_dataset, scaler, X_scaled, scaled_row_array):
    # Anchor non ha dei valori numerici, ma permette di produrre delle regole che spiegano la previsione. Queste regole
    # vengono poi utilizzate su istanze "vicine" a quella spiegata, per controllare quanto siano affidabili, attraverso
    # la Precision (La percentuale di istanze vicine per cui le regole sono vere) e il Coverage (La percentuale di istanze
    # del dataset che sono coperte da quelle regole)

    anchor_result = ""

    # Creiamo l'explainer e otteniamo la spiegazione
    explainer = anchor_tabular.AnchorTabularExplainer(
        class_names=["No Diabetes", "Diabetes"],
        feature_names=diabetes_dataset.columns[:-1],
        train_data=X_scaled,
        categorical_names={}
    )

    explanation = explainer.explain_instance(scaled_row_array.flatten(), model.predict, threshold=0.95)

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
    anchor_result += 'Rule: %s' % (' AND '.join(rules_list)) + "\n"
    anchor_result += 'Coverage: %.2f' % explanation.coverage()

    return anchor_result


#TODO
def ask_gpt():
    client = OpenAI(
        api_key=""
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Say this is a test!"
            }
        ],
        model="gpt-3.5-turbo",
    )

    print(chat_completion.choices[0].message.content)


def show_results(prediction, explanation):
    if prediction != None:
        st.subheader("Diagnosis")
        if prediction == 0:
            st.success("The patient doesn't have diabetes")
        else:
            st.error("The patient has diabetes")
        st.divider()

    if explanation:
        st.subheader("Explanation")
        st.markdown(explanation)
        st.divider()


def show_signature():
    st.markdown("<p style='text-align: right;'>Created by MonTheDog 🌠 </p>", unsafe_allow_html=True)


if __name__ == "__main__":

    # Definiamo i valori di default per il modello (Riga 0 del dataset) e la struttura dati per l'input
    model_input = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 125,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }

    # Carichiamo ciò che ci serve
    model, scaler, diabetes_dataset, X_scaled = get_data()

    # Creiamo il titolo
    create_title()

    # Creiamo il form per l'inserimento dei valori
    create_form(model_input)

    # Creiamo il bottone per la predizione e la spiegazione
    prediction, explanation = create_submit(model_input, scaler, model, diabetes_dataset, X_scaled)

    # Mostriamo la diagnosi e la spiegazione
    show_results(prediction, explanation)

    # Mostrare la firma
    show_signature()
