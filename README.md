# Student Depression Prediction

## 📌 Project Overview
This project applies **machine learning models** to predict student depression based on academic, lifestyle, and personal factors.  
The workflow includes:
- Data preprocessing (handling missing values, encoding categorical features, scaling)
- Training multiple ML models
- Selecting the best model based on F1-score
- Generating a classification report
- Saving the trained model and scaler for future use

---

## 📊 Dataset Description
The dataset contains information about students, including demographics, academic performance, lifestyle habits, and mental health indicators.

### Columns
- **id** – Unique identifier (dropped during preprocessing)
- **Gender** – Male/Female
- **Age** – Age of student
- **City** – City of residence
- **Profession** – Student/Other
- **Academic Pressure** – Numeric scale
- **Work Pressure** – Numeric scale
- **CGPA** – Cumulative Grade Point Average
- **Study Satisfaction** – Numeric scale
- **Job Satisfaction** – Numeric scale
- **Sleep Duration** – e.g., "5-6 hours", "7-8 hours"
- **Dietary Habits** – Healthy/Moderate/Unhealthy
- **Degree** – Academic degree pursued
- **Have you ever had suicidal thoughts?** – Yes/No
- **Work/Study Hours** – Numeric scale
- **Financial Stress** – Numeric scale
- **Family History of Mental Illness** – Yes/No
- **Depression** – Target variable (0 = No, 1 = Yes)

---

## ⚙️ Code Workflow
1. **Load Data** – Reads CSV file into a Pandas DataFrame.
2. **Handle Missing Values**  
   - Numeric columns → filled with median  
   - Categorical columns → filled with mode
3. **Encode Categorical Variables** – One-hot encoding with `pd.get_dummies`.
4. **Split Features & Target** – Separates predictors (X) and target (y).
5. **Feature Scaling** – Standardizes features using `StandardScaler`.
6. **Train-Test Split** – Stratified split (80% train, 20% test).
7. **Model Training** – Logistic Regression, Random Forest, SVM, Decision Tree, Naive Bayes.
8. **Evaluation** – Accuracy and F1-score for each model.
9. **Best Model Selection** – Chooses model with highest F1-score.
10. **Final Report** – Prints classification report.
11. **Save Model** – Saves best model and scaler using `pickle`.

---

## 🚀 How to Run
1. Clone or download the project folder.
2. Place the dataset file as **`Student Depression Dataset.csv`** in the same directory as `app.py`.
3. Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate
   ```
4. Install dependencies:
 ```
pip install pandas scikit-learn
 ```
5. Run the script:
```
python app.py
 ```
## 📈 Output
- Prints accuracy and F1-score for all models.
- Displays classification report for the best model.
- Saves:
  - `best_depression_model.pkl` (trained model)
  - `scaler.pkl` (feature scaler)

---

## ✅ Notes
- Ensure Python and required libraries are installed.
- The warning about `select_dtypes(include=["object"])` has been fixed by explicitly including `"object", "string"`.
- The project is designed to be extendable — you can add more models or tune hyperparameters.

---

## 🔮 Future Improvements
- Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
- Cross-validation for more robust evaluation
- Integration of deep learning models (e.g., neural networks)
- Deployment as a web app using Flask/Django

## 📈 Output screenshot
<img width="612" height="590" alt="image" src="https://github.com/user-attachments/assets/69a9047a-d81f-437b-a451-318fed4b07db" />

<img width="602" height="391" alt="image" src="https://github.com/user-attachments/assets/5c11fa95-a8f4-487d-ade2-a6e6320c87e5" />


