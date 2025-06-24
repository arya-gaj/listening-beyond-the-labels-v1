def count_audio_files(path, exts=(".mp3")):

    return len([f for f in os.listdir(path) if f.endswith(exts)])
