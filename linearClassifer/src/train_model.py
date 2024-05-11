# src/train_model.py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def train():
    # Load dataset
    data = load_iris()
    X, y = data.data, data.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Initialize and train classifier
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print("Model accuracy:", accuracy)

if __name__ == "__main__":
    train()
