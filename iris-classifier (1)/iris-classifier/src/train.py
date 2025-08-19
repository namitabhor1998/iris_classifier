import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def main(test_size: float = 0.2, random_state: int = 42):
    """Train a Decision Tree on the Iris dataset, print accuracy,
    and save the confusion matrix plot to outputs/confusion_matrix.png.
    Returns:
        float: accuracy score
    """
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Decision Tree on Iris dataset.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of test set size (e.g., 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for train/test split and model")
    args = parser.parse_args()
    main(test_size=args.test_size, random_state=args.random_state)
