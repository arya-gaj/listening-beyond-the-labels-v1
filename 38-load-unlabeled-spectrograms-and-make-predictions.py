if os.path.exists(unlabeled_data_dir):
    files = [f for f in os.listdir(unlabeled_data_dir) if f.endswith('.npy')]
    print(f"Found {len(files)} .npy files in the directory.")

    X_unlabeled = []

    for file in files:
        file_path = os.path.join(unlabeled_data_dir, file)
        data = np.load(file_path)

        if data.shape == (128, 157):
            X_unlabeled.append(data[..., np.newaxis])

    X_unlabeled = np.array(X_unlabeled)
    print(f"Shape of X_unlabeled: {X_unlabeled.shape}")

    if X_unlabeled.shape[0] > 0:
        predictions = model.predict(X_unlabeled)
        print(f"Predictions: {predictions}")

    else:
        print("No valid data found to make predictions.")

else:
    print("Directory not found. Please check the path.")
