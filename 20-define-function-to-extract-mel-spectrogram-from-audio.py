def extract_mel_spec(file_path, sr=16000, n_mels=128, duration=5):
    y, _ = librosa.load(file_path, sr=sr, duration=duration)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db
