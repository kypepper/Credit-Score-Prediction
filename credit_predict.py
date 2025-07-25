import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(layout="wide")

# ------------------- Helper Functions -------------------
def preprocess_data(data, target_col):
    X = data.drop(columns=[target_col])
    y = data[target_col]

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'bool', 'category']).columns

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

def train_model(X, y, preprocessor):
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', GradientBoostingRegressor(random_state=42))])

    param_grid = {
        'regressor__n_estimators': [100],
        'regressor__learning_rate': [0.1],
        'regressor__max_depth': [3]
    }

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2')
    grid_search.fit(X, y)

    return grid_search.best_estimator_

def predict_score(model, user_input_df):
    return model.predict(user_input_df)[0]

def clamp_to_range(val, min_val, max_val):
    return max(min(val, max_val), min_val)

def plot_distributions(X, y, user_input_df, predicted_value, target_col):
    num_plots = len(X.columns) + 1
    cols = 3
    rows = int(np.ceil(num_plots / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(20, rows * 4))
    axs = axs.flatten()

    # Target variable
    sns.histplot(y, kde=True, ax=axs[0], color='blue')
    line = clamp_to_range(predicted_value, y.min(), y.max())
    axs[0].axvline(line, color='red', linestyle='--', label='Your Prediction')
    axs[0].set_title(f'Distribution of {target_col}')
    axs[0].legend()

    for i, col in enumerate(X.columns):
        sns.histplot(X[col], kde=True, ax=axs[i + 1], color='green')
        val = user_input_df[col].values[0]
        clamped_val = clamp_to_range(val, X[col].min(), X[col].max())
        axs[i + 1].axvline(clamped_val, color='red', linestyle='--')
        axs[i + 1].set_title(f'Distribution of {col}')

    for ax in axs[num_plots:]:
        ax.axis('off')

    st.pyplot(fig)

# ------------------- Streamlit App -------------------
st.title("📊 Regression & Distribution Visualizer")

# ---------- SECTION 1: Default Credit Score Model ----------
st.header("💳 Credit Score Prediction")

credit_data = pd.read_csv("Credit.csv", index_col=0)
X_credit, y_credit, credit_preprocessor = preprocess_data(credit_data, 'Rating')
credit_model = train_model(X_credit, y_credit, credit_preprocessor)

st.subheader("🔧 Input Your Credit Info Below")
user_input = {
    'Income': st.slider('Income', 0.0, 300.0, 50.0),
    'Limit': st.slider('Credit Limit', 0.0, 1000.0, 200.0),
    'Cards': st.slider('Number of Credit Cards', 1, 10, 2),
    'Age': st.slider('Age', 18, 100, 30),
    'Education': st.slider('Education (Years)', 1, 5, 2),
    'Gender': st.selectbox('Gender', ['Male', 'Female']),
    'Student': st.selectbox('Student?', ['Yes', 'No']),
    'Married': st.selectbox('Married?', ['Yes', 'No']),
    'Ethnicity': st.selectbox('Ethnicity', ['Caucasian', 'Asian', 'African American']),
    'Balance': st.slider('Avg Credit Card Balance', 0.0, 2000.0, 500.0),
}
user_df = pd.DataFrame([user_input])
predicted_score = predict_score(credit_model, user_df)

st.success(f"✅ Predicted Credit Score: **{predicted_score:.2f}**")

st.subheader("📉 Distribution Comparison")
plot_distributions(X_credit, y_credit, user_df, predicted_score, "Rating")

# ---------- SECTION 2: Upload Your Own CSV ----------
st.markdown("---")
st.header("📤 Custom Dataset Analysis")
st.markdown("Please drop your own CSV if you want to visualize the correlation and perform regressions on a different topic.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    custom_data = pd.read_csv(uploaded_file)
    st.subheader("Correlation Matrix")
    numeric_df = custom_data.select_dtypes(include=['int64', 'float64'])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    st.subheader("Target Selection for Regression")
    all_columns = custom_data.columns.tolist()
    target_col = st.selectbox("Select Target Column (what to predict)", all_columns)
    feature_cols = st.multiselect("Select Feature Columns (inputs)", [col for col in all_columns if col != target_col])

    if feature_cols and target_col:
        try:
            regression_df = custom_data[feature_cols + [target_col]].dropna()
            X_custom, y_custom, custom_preprocessor = preprocess_data(regression_df, target_col)
            custom_model = train_model(X_custom, y_custom, custom_preprocessor)

            st.subheader("🔢 Enter Your Input Values")

            custom_input_dict = {}
            for col in feature_cols:
                if regression_df[col].dtype == 'object':
                    unique_vals = regression_df[col].dropna().unique().tolist()
                    if unique_vals:
                        custom_input_dict[col] = st.selectbox(f"{col}", unique_vals)
                    else:
                        custom_input_dict[col] = ""
                else:
                    min_val, max_val = float(regression_df[col].min()), float(regression_df[col].max())
                    default_val = float(regression_df[col].mean())
                    custom_input_dict[col] = st.slider(f"{col}", min_val, max_val, default_val)

            custom_input_df = pd.DataFrame([custom_input_dict])
            predicted_val = predict_score(custom_model, custom_input_df)

            st.success(f"✅ Predicted {target_col}: **{predicted_val:.2f}**")

            st.subheader("📉 Distribution Comparison")
            plot_distributions(X_custom, y_custom, custom_input_df, predicted_val, target_col)

        except Exception as e:
            st.error(f"Error in model training or prediction: {e}")
