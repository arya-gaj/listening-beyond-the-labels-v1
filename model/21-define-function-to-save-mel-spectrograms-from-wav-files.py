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

def extract_all_mels(src_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)

    for file in tqdm(os.listdir(src_dir)):

        if not file.endswith('.wav'):

          continue
        full_path = os.path.join(src_dir, file)
        mel = extract_mel_spec(full_path)
        out_path = os.path.join(dest_dir, file.replace('.wav', '.npy'))
        np.save(out_path, mel)
