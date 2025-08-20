import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Plot confusion matrix directly from model
disp = ConfusionMatrixDisplay.from_estimator(
    model, X_test, y_test, display_labels=iris.target_names, cmap="Blues", values_format="d"
)

plt.title("Confusion Matrix - Decision Tree (Iris)")
plt.savefig("outputs/confusion_matrix.png")
plt.close()
print("Confusion matrix saved to outputs/confusion_matrix.png")
