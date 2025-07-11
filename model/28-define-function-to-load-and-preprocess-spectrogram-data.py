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
