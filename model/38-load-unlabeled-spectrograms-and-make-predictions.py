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
