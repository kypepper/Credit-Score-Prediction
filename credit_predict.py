import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# --- Load dataset ---
def load_dataset(file_path):
    return pd.read_csv(file_path, index_col=0)

# --- Preprocess for credit model ---
def preprocess_credit_data(data):
    X = data.drop(columns=['Rating'])
    y = data['Rating']

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return X, y, preprocessor

# --- Train model for credit scoring ---
def train_credit_model(X, y, preprocessor):
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', GradientBoostingRegressor(random_state=42))])

    param_grid = {
        'regressor__n_estimators': [100],
        'regressor__learning_rate': [0.1],
        'regressor__max_depth': [3]
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
    grid_search.fit(X, y)

    return grid_search.best_estimator_

# --- Predict credit score ---
def predict_credit_score(model, user_input):
    user_data = pd.DataFrame(user_input, index=[0])
    return model.predict(user_data)[0]

# --- Display credit graphs ---
def display_credit_graphs(X, y, user_input, predicted_score):
    fig, axs = plt.subplots(2, len(X.columns) // 2 + 1, figsize=(20, 10))
    axs = axs.flatten()

    sns.histplot(y, kde=True, color='blue', ax=axs[0])
    score_to_plot = min(predicted_score, axs[0].get_xlim()[1])
    axs[0].axvline(score_to_plot, color='red', linestyle='--', linewidth=2, label='Your Score')
    axs[0].set_title("Your Credit Score vs Distribution")
    axs[0].legend()

    for i, col in enumerate(X.columns):
        sns.histplot(X[col], kde=True, ax=axs[i + 1], color='green')
        if col in user_input:
            val = user_input[col]
            x_min, x_max = axs[i + 1].get_xlim()
            val = min(max(val, x_min), x_max)
            axs[i + 1].axvline(val, color='red', linestyle='--', linewidth=2)
        axs[i + 1].set_title(f"{col} Distribution")

    plt.tight_layout()
    st.pyplot(fig)

# --- Visualize user CSV ---
def visualize_uploaded_csv(df):
    st.subheader("Custom Dataset Exploration")
    st.write("Basic Statistics")
    st.write(df.describe(include='all'))

    st.subheader("Correlation Matrix")
    if df.select_dtypes(include='number').shape[1] > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    st.subheader("Distribution Plots")
    for col in df.select_dtypes(include='number').columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    st.subheader("Optional Regression (First Column as Target)")
    if df.select_dtypes(include='number').shape[1] > 1:
        target = df.columns[0]
        features = df.columns[1:]

        X = df[features]
        y = df[target]

        X = X.fillna(X.median(numeric_only=True))
        model = GradientBoostingRegressor().fit(X, y)
        score = model.score(X, y)

        st.write(f"R² score of basic model: {score:.2f}")

# --- Streamlit App ---
def main():
    st.title("Credit Score Predictor and Data Explorer")

    # Sidebar user input
    st.sidebar.header("Enter Your Information for Credit Score Prediction")
    user_input = {
        'Income': float(st.sidebar.text_input("Income", "50000")),
        'Limit': float(st.sidebar.text_input("Credit Limit", "10000")),
        'Cards': int(st.sidebar.text_input("Number of Credit Cards", "3")),
        'Age': int(st.sidebar.text_input("Age", "30")),
        'Education': int(st.sidebar.text_input("Education Level (Years)", "16")),
        'Gender': st.sidebar.selectbox("Gender", ["Male", "Female"]),
        'Student': st.sidebar.selectbox("Are you a student?", ["Yes", "No"]),
        'Married': st.sidebar.selectbox("Are you married?", ["Yes", "No"]),
        'Ethnicity': st.sidebar.selectbox("Ethnicity", ["Caucasian", "Asian", "African American"]),
        'Balance': float(st.sidebar.text_input("Avg. Credit Card Balance", "1500")),
    }

    # Default credit model
    data = load_dataset("Credit.csv")
    X, y, preprocessor = preprocess_credit_data(data)
    model = train_credit_model(X, y, preprocessor)
    prediction = predict_credit_score(model, user_input)

    st.success(f"Your predicted credit score is: {prediction:.2f}")
    display_credit_graphs(X, y, user_input, prediction)

    # Section for uploading custom CSV
    st.markdown("---")
    st.subheader("📁 Upload Your Own Dataset")
    st.write("Please drop your own CSV if you want to visualize the correlation and perform regressions on a different topic.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        visualize_uploaded_csv(df)

if __name__ == "__main__":
    main()
