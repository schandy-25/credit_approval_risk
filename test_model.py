from src.modeling import train_model

clf, (X_test, y_test, s_test, y_pred, y_proba, metrics) = train_model()

print("Accuracy:", metrics["accuracy"])
print("ROC AUC:", metrics["roc_auc"])
print("Confusion matrix:\n", metrics["confusion_matrix"])
