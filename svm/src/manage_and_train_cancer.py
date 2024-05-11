# src/train_model.py
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd

def save_dataset_to_csv():
    # Load dataset
    cancer_data = load_breast_cancer()
    X = pd.DataFrame(data=cancer_data.data, columns=cancer_data.feature_names)
    y = pd.DataFrame(data=cancer_data.target, columns=['target'])

    # Concatenate features and target into a single DataFrame
    df = pd.concat([X, y], axis=1)

    # Save the DataFrame to a CSV file
    df.to_csv('../data/breast_cancer.csv', index=False)

    print("Breast Cancer dataset saved to breast_cancer.csv")

def train_model():
    # Load dataset
    df = pd.read_csv('../data/breast_cancer.csv')

    # Split data into features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train classifier
    model = SVC()
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print("Model accuracy:", accuracy)

if __name__ == "__main__":
    save_dataset_to_csv()
    train_model()
