def load_spectrogram_data(dementia_dir, control_dir):
    X = []
    y = []

    for file in os.listdir(dementia_dir):

        if file.endswith('.npy'):
            spec = np.load(os.path.join(dementia_dir, file))

            if spec.shape[1] < 157:
                pad_width = 157 - spec.shape[1]
                spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')

            X.append(spec)
            y.append(1)

    for file in os.listdir(control_dir):

        if file.endswith('.npy'):
            spec = np.load(os.path.join(control_dir, file))

            if spec.shape[1] < 157:
                pad_width = 157 - spec.shape[1]
                spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')

            X.append(spec)
            y.append(0)

    X = np.array(X)
    X = np.expand_dims(X, -1)
    y = np.array(y)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
