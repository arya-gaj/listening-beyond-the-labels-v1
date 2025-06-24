def convert_single(file_info):
    src_file, src_dir, dest_dir = file_info

    try:
        src_path = os.path.join(src_dir, src_file)
        dest_path = os.path.join(dest_dir, src_file.replace('.mp3', '.wav'))

        if not os.path.exists(dest_path):
            audio = AudioSegment.from_mp3(src_path)
            audio.export(dest_path, format='wav')

    except Exception as e:
        print(f"Error with {src_file}: {e}")
