avg_train_acc = sum(history.history['accuracy']) / len(history.history['accuracy'])
print(f"Average Training Accuracy: {avg_train_acc:.4f}")
