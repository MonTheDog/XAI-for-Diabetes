# XAI for Diabetes
This repository contains all the code, plots, and results used in the Bachelor’s Thesis in Computer Science titled "DIA: Un Nuovo Approccio per Spiegare l’Output di Modelli di Predizione del Diabete" ("DIA: A New Approach to Explain the Output of Diabetes Prediction Models"), presented at the University of Salerno in collaboration with SeSaLab.

## Project Structure
- The anchor and ciu folders contain Python libraries that implement the respective XAI techniques.
- The images folder contains all the plots and figures used in the thesis.
- The models folder contains the pickle files of the trained models.
- The diabetes.csv file contains the original dataset, that is the Pima Indians Diabetes Database (https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).
- The diabetes_corrected.csv file contains the cleaned version of the dataset, with all the detected issues resolved, which was used for training.
- The classifier.py script performs data cleaning, model training, and evaluation (re-running it is not strictly necessary since models are already included in the repository).
- The scores.csv file contains the evaluation results of the trained models.
- The explainability.py script runs the different explainability techniques considered in the project.
- The evaluation.py script evaluates the best classifier–explainability technique pair.
- The explanation_result.csv file contains the results of the classifier–explainability technique pair evaluations.
- The summarization.py script includes the LLM-based summarization logic as well as the user interface built with Streamlit.
- The requirements.txt file lists all the required Python dependencies to run the project.
- The results.pkl and scaler.pkl files contain serialized Python objects used within the scripts.
- The prompt hisotry.txt file contains the prompt iterations in order.

## Setup
```bash
  # Clone the repository
  git clone https://github.com/MonTheDog/XAI-for-Diabetes.git
  cd XAI-for-Diabetes

  # (Optional) Create a virtual environment
  python3 -m venv venv
  source venv/bin/activate

  # Install the dependencies
  pip install -r requirements.txt
```

## How to Run
To launch the user interface, you will need an OpenAI API key.
Insert your API key into the api_key variable at line 237 in the summarization.py script.

Then run the following command:
```bash 
streamlit run summarization.py
```

This will open a new browser window with the user interface ready to use.

## Reproducibility Notes
- All random seeds are fixed to ensure reproducibility.
- All outputs are already saved in the repository, so re-running all scripts is not strictly required.
- ChatGPT responses may vary from those included due to the probabilistic nature of the model.
