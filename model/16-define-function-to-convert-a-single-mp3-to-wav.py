"""
Listening Beyond the Labels – A scalable and non-invasive speech-based machine learning model for early Alzheimer's detection using mel-spectrograms and lightweight semi-supervised CNN with no transcription or neuroimaging needed.

Copyright (C) 2025 Aryaman Gajrani

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Contact: arya-gaj@proton.me
"""

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
