# Auto Judge  
### Predicting Programming Problem Difficulty

Auto Judge is a machine learning–based system that automatically predicts the **difficulty level** and **difficulty score** of programming problems using only their textual descriptions.

Online coding platforms such as Codeforces, CodeChef, and Kattis usually rely on human judgment and user feedback to label problems as *Easy*, *Medium*, or *Hard*.  
This project demonstrates how **Natural Language Processing (NLP)** and **machine learning** can be used to automate this process.

---

## 📖 Project Objectives

The system predicts:

- **Problem Class** → Easy / Medium / Hard *(Classification)*
- **Problem Score** → Numerical difficulty value *(Regression)*

Predictions are based **only on text**, including:
- Problem description
- Input description
- Output description

No metadata, user submissions, or solution code is used.

---

## 📑 Dataset

Each data sample contains:
- `title`
- `description`
- `input_description`
- `output_description`
- `problem_class` (Easy / Medium / Hard)
- `problem_score` (numerical)

The dataset is provided and **no manual labeling is required**.

---

## 📜 Approach

### 1. Data Preprocessing
- Combined description, input, and output into a single text field
- Cleaned text by removing HTML, URLs, and extra whitespace
- Converted text to lowercase

### 2. Feature Engineering
**Text Features**
- TF-IDF vectors (unigrams + bigrams, max 30,000 features)

**Handcrafted Numeric Features**
- Log-transformed text length
- Log-transformed count of mathematical symbols
- Constraint indicators (constraints, large N, time limits)
- Keyword frequency for algorithm categories:
  - Dynamic Programming
  - Graph Algorithms
  - Data Structures
  - Math
  - Geometry
  - Strings
  - Greedy Techniques

TF-IDF features and numeric features are concatenated to form the final input.

---

## 🗃️ Models Used

### Classification
- Logistic Regression *(final model)*
- Tried: Random Forest, SVM

### Regression
- Gradient Boosting Regressor *(final model)*
- Tried: Linear Regression, Random Forest

Deep learning was **not used**, as per project guidelines.

---

## 📓 Evaluation Results

### Classification (Logistic Regression)
- **Test Accuracy:** ~54%
- Hard problems were predicted most reliably
- Medium problems were hardest to classify due to overlap

### Regression (Gradient Boosting)
- **Test RMSE:** ~2.01  
- **Test MAE:** ~1.68  

> Note: Gradient Boosting smooths predictions by averaging multiple weak learners, so exact score matching is not expected. Small deviations are normal in NLP-based regression tasks.

---

## 🌐 Web Interface (Streamlit)

A simple Streamlit web app allows users to:

1. Paste:
   - Problem description
   - Input description
   - Output description
2. Click **Predict**
3. View:
   - Predicted difficulty class
   - Predicted difficulty score

No authentication or database is required.

---

## 🖼️ Web Interface Screenshots

![before_1](sample_web_interface_images/before_1.png)
![before_2](sample_web_interface_images/before_2.png)
![after_1](sample_web_interface_images/after_1.png)
![after_2](sample_web_interface_images/after_2.png)


---

## How to Run the Web App (Optional)

```bash
pip install streamlit scikit-learn pandas numpy scipy joblib
streamlit run streamlit_app.py
```

---

## 📃 Project Structure

AutoJudge/
│
├── AutoJudge.ipynb
│   └── Model training, feature engineering, evaluation, and model saving (Colab)
│
├── streamlit_app.py
│   └── Streamlit web application for difficulty prediction
│
├── models/
│   ├── tfidf.pkl
│   ├── logreg_classifier.pkl
│   ├── gb_regressor.pkl
│   └── label_encoder.pkl
│   └── Saved models and preprocessing objects used by the web app
│
├── sample_web_interface_images/
│   └── Screenshots of the Streamlit web interface
│
├── data.jsonl
│   └── Dataset used for training and evaluation
│
└── README.md
    └── Project documentation
