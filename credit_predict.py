# Import necessary libraries
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

# Function to load the dataset
def load_dataset(file_path):
    return pd.read_csv(file_path, index_col=0)  # Specify index column

# Function to preprocess the dataset
def preprocess_data(data):
    # Separate features (X) and target variable (y)
    X = data.drop(columns=['Rating'])  # Update to use 'Rating' as target variable
    y = data['Rating']  # Update to use 'Rating' as target variable

    # Define preprocessing steps
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Replace missing values with median
        ('scaler', StandardScaler())  # Scale numerical features
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Replace missing values with most frequent value
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical variables
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return X, y, preprocessor

# Function to train the model
def train_model(X, y, preprocessor):
    # Define the model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', GradientBoostingRegressor(random_state=42))])

    # Define hyperparameters to tune
    param_grid = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__learning_rate': [0.05, 0.1, 0.2],
        'regressor__max_depth': [3, 4, 5]
    }

    # Perform grid search cross-validation to find the best hyperparameters
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
    grid_search.fit(X, y)  # Train on entire dataset for better accuracy

    # Get the best model
    best_model = grid_search.best_estimator_

    return best_model

# Function to predict credit score based on user input
def predict_credit_score(model, user_input):
    # Convert user input into DataFrame
    user_data = pd.DataFrame(user_input, index=[0])
    # Make predictions using the model
    predicted_score = model.predict(user_data)
    return predicted_score[0]

# Function to display all the graphs on the same tab
def display_all_graphs(X, y, predicted_credit_score):
    fig, axs = plt.subplots(2, len(X.columns) // 2 + 1, figsize=(20, 10))
    axs = axs.flatten()

    # Plot distribution of target variable (Credit Scores)
    sns.histplot(y, kde=True, color='blue', ax=axs[0])
    axs[0].axvline(predicted_credit_score, color='red', linestyle='dashed', linewidth=2, label='Your Credit Score')
    axs[0].set_xlabel('Credit Score')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Comparison of Your Credit Score to Others')
    axs[0].legend()

    # Plot distribution of each feature
    for i, feature in enumerate(X.columns):
        sns.histplot(X[feature], kde=True, color='green', ax=axs[i+1])
        axs[i+1].set_xlabel(feature)
        axs[i+1].set_ylabel('Frequency')
        axs[i+1].set_title(f'Distribution of {feature}')

    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Load the dataset
    data = load_dataset(r"C:\Users\User\Desktop\PROJECTS\Credit.csv")

    # Preprocess the data
    X, y, preprocessor = preprocess_data(data)

    # Train the model
    best_model = train_model(X, y, preprocessor)

    # User input
    user_input = {
        'Income': float(input("Enter your income: ")),
        'Limit': float(input("Enter your credit limit: ")),
        'Cards': int(input("Enter the number of credit cards you have: ")),
        'Age': int(input("Enter your age: ")),
        'Education': int(input("Enter your education level (in years): ")),
        'Gender': input("Enter your gender (Male/Female): "),
        'Student': input("Are you a student? (Yes/No): "),
        'Married': input("Are you married? (Yes/No): "),
        'Ethnicity': input("Enter your ethnicity (Caucasian/Asian/African American): "),
        'Balance': float(input("Enter your average credit card balance: ")),
    }

    # Predict credit score based on user input
    predicted_credit_score = predict_credit_score(best_model, user_input)
    print("Predicted Credit Score:", predicted_credit_score)

    # Display all the graphs on the same tab
    display_all_graphs(X, y, predicted_credit_score)

# Entry point of the program
if __name__ == "__main__":
    main()
