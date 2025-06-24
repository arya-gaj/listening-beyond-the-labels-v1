def convert_all_mp3s(src_dir, dest_dir, max_threads=8):
    os.makedirs(dest_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.endswith(".mp3")]
    file_infos = [(f, src_dir, dest_dir) for f in files]

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        list(tqdm(executor.map(convert_single, file_infos), total=len(file_infos)))
