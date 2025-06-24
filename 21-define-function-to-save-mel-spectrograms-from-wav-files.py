def extract_all_mels(src_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)

    for file in tqdm(os.listdir(src_dir)):

        if not file.endswith('.wav'):

          continue
        full_path = os.path.join(src_dir, file)
        mel = extract_mel_spec(full_path)
        out_path = os.path.join(dest_dir, file.replace('.wav', '.npy'))
        np.save(out_path, mel)
