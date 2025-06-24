confident_preds = (predictions > 0.6) | (predictions < 0.4)
pseudo_accuracy = np.sum(confident_preds) / len(predictions)
