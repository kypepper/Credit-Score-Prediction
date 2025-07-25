import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load default dataset
def load_default_dataset():
    return pd.read_csv("Credit.csv", index_col=0)

# Preprocessing
def preprocess_data(data):
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

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return X, y, preprocessor

# Train model
def train_model(X, y, preprocessor):
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', GradientBoostingRegressor(random_state=42))])
    model.fit(X, y)
    return model

# Predict score
def predict_credit_score(model, user_input_df):
    return model.predict(user_input_df)[0]

# Graphs
def display_all_graphs(X, y, predicted_credit_score, user_input):
    fig, axs = plt.subplots(2, len(X.columns) // 2 + 1, figsize=(20, 10))
    axs = axs.flatten()

    # Plot distribution of target variable (Credit Score)
    sns.histplot(y, kde=True, color='blue', ax=axs[0])
    axs[0].axvline(predicted_credit_score, color='red', linestyle='dashed', linewidth=2, label='Your Predicted Score')
    axs[0].set_xlabel('Rating')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Your Credit Score vs Population')
    axs[0].legend()

    # Plot distribution of each feature with user input line
    for i, feature in enumerate(X.columns):
        sns.histplot(X[feature], kde=True, color='green', ax=axs[i+1])
        axs[i+1].set_xlabel(feature)
        axs[i+1].set_ylabel('Frequency')
        axs[i+1].set_title(f'Distribution of {feature}')

        if feature in user_input:
            try:
                val = float(user_input[feature])
                min_val = X[feature].min()
                max_val = X[feature].max()

                # Clamp the value to axis limits
                if val < min_val:
                    axs[i+1].axvline(min_val, color='red', linestyle='dashed', linewidth=2)
                    axs[i+1].text(min_val, axs[i+1].get_ylim()[1]*0.9, 'Your Input\n(too low)', color='red')
                elif val > max_val:
                    axs[i+1].axvline(max_val, color='red', linestyle='dashed', linewidth=2)
                    axs[i+1].text(max_val, axs[i+1].get_ylim()[1]*0.9, 'Your Input\n(too high)', color='red', ha='right')
                else:
                    axs[i+1].axvline(val, color='red', linestyle='dashed', linewidth=2, label='Your Input')
                    axs[i+1].legend()
            except:
                pass

    plt.tight_layout()
    st.pyplot(fig)

# Visualize custom CSV
def visualize_custom_csv(data):
    st.subheader("Correlation Heatmap")
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    corr = numeric_data.corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.subheader("Build a Simple Regression")
    features = list(numeric_data.columns)
    if len(features) < 2:
        st.warning("Not enough numeric columns for regression.")
        return

    x_feature = st.selectbox("Select X feature", features, key="x_feat")
    y_feature = st.selectbox("Select Y feature", features, index=1, key="y_feat")

    if x_feature != y_feature:
        sns.lmplot(data=data, x=x_feature, y=y_feature)
        st.pyplot(plt.gcf())

# Streamlit App
def main():
    st.title("📊 Credit Score Predictor + Custom CSV Visualizer")

    # Default model training
    st.header("Default Credit Score Prediction")
    try:
        data = load_default_dataset()
        X, y, preprocessor = preprocess_data(data)
        model = train_model(X, y, preprocessor)

        st.sidebar.header("Enter Your Info")

        user_input = {
            'Income': st.sidebar.number_input("Income (Thousands)", min_value=0.0, value=50000.0),
            'Limit': st.sidebar.number_input("Credit Limit", min_value=0.0, value=10000.0),
            'Cards': st.sidebar.number_input("Number of Credit Cards", min_value=0, value=3),
            'Age': st.sidebar.number_input("Age", min_value=18, value=30),
            'Education': st.sidebar.number_input("Education Level (Years)", min_value=0, value=16),
            'Gender': st.sidebar.selectbox("Gender", ['Male', 'Female']),
            'Student': st.sidebar.selectbox("Student Status", ['Yes', 'No']),
            'Married': st.sidebar.selectbox("Marital Status", ['Yes', 'No']),
            'Ethnicity': st.sidebar.selectbox("Ethnicity", ['Caucasian', 'Asian', 'African American']),
            'Balance': st.sidebar.number_input("Average Credit Card Balance", min_value=0.0, value=1500.0)
        }

        user_df = pd.DataFrame(user_input, index=[0])
        predicted_score = predict_credit_score(model, user_df)
        st.subheader(f"📈 Predicted Credit Score: **{predicted_score:.2f}**")
        display_all_graphs(X, y, predicted_score, user_input)

    except FileNotFoundError:
        st.error("Default file 'Credit.csv' not found in working directory.")

    # Custom CSV section
    st.markdown("---")
    st.header("📂 Please drop your own CSV if you want to visualize the correlation and perform regressions on a different topic.")
    custom_file = st.file_uploader("Upload your CSV file", type="csv")

    if custom_file is not None:
        try:
            custom_data = pd.read_csv(custom_file)
            st.success("Custom dataset loaded successfully!")
            st.write("Preview of uploaded data:")
            st.dataframe(custom_data.head())
            visualize_custom_csv(custom_data)
        except Exception as e:
            st.error(f"Error reading file: {e}")

if __name__ == "__main__":
    main()
