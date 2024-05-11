# src/train_model.py
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

def save_dataset_to_csv():
    # Load dataset
    wine_data = load_wine()
    X = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
    y = pd.DataFrame(data=wine_data.target, columns=['target'])

    # Concatenate features and target into a single DataFrame
    df = pd.concat([X, y], axis=1)

    # Save the DataFrame to a CSV file
    df.to_csv('../data/wine.csv', index=False)

    print("Wine dataset saved to wine.csv")

def train_model():
    # Load dataset
    df = pd.read_csv('../data/wine.csv')

    # Split data into features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train classifier
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print("Model accuracy:", accuracy)

if __name__ == "__main__":
    save_dataset_to_csv()
    train_model()
