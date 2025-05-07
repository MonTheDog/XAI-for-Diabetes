# XAI for Diabetes
Questa repo contiene tutto il codice, i grafici e i risultati utilizzati per la scrittura della Tesi Triennale in Informatica "DIA: Un Nuovo Approccio per Spiegare l’Output di Modelli di Predizione del Diabete" presso l'Università degli Studi di Salerno in collaborazione col SeSaLab.

## Struttura
- Le cartelle anchor e ciu sono librerie python importate per utilizzare le suddette tecniche
- La cartella images contiene i grafici utilizzati nella tesi
- La cartella models contiene i file pickle dei modelli addestrati
- Il file diabetes.csv contiene il dataset utilizzato per l'addestramento, in particolare è il Pima Indians Diabetes Database (https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Il file diabetes_corrected.csv contiene il dataset pulito da evenutali problemi
- Il file classifier.py contiene lo script utilizzato per il Data Cleaning, il training e la valutazione dei modelli (non è strettamente necessario rieseguirlo in quanto i modelli sono già presenti nella repository)
- Il file scores.csv contiene i risultati della valutazione dei modelli
- Il file explainability.py contiene lo script di esecuzione delle varie tecniche di explainability prese in considerazione
- Il file evaluation.py contiene lo script di valutazione della miglior coppia classificatore-algoritmo di explainability
- Il file explanation_result.csv contiene i risultati della valutazione delle coppie classificatore-algoritmo di explainability
- Il file summarization.py contiene lo script di summarization utilizzando LLMs e lo script per l'interfaccia grafica utente
- Il file requirements.txt contiene tutte le dipendenze python per eseguire il progetto
- I file results.pkl e scaler.pkl contengono oggetti pickle per eseguire alcune logiche negli script

## Setup
```bash 
  # Clona la repository 
  git clone https://github.com/MonTheDog/XAI-for-Diabetes.git 
  cd XAI-for-Diabetes 
  # Crea un virtual environment (opzionale)
  python3 -m venv venv 
  source venv/bin/activate 
  # Installa le dipendenze 
  pip install -r requirements.txt 
```

## Come eseguire
Per lanciare l'interfaccia utente bisogna avere una chiave API di OpenAI e inserirla nella variabile api_key a riga 237 dello script summarization.py.
Dopodiché da bash bisognerà lanciare il seguente comando
```bash 
  streamlit run summarization.py
```
A questo punto si aprira una finestra nel browser con l'interfaccia utente pronta all'utilizzo

## Note di riproducibilità
- Tutti i random seed sono fissi per garantire riproducibilità
- Tutti gli output sono già salvati nel progetto per evitare la necessità di rieseguire tutti gli script
- Le risposte di ChatGPT possono chiaramente divergere da quelle indicate a causa della natura del modello
