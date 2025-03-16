import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from tqdm import tqdm

# Set main data directory
data_dir = './data'
file_paths = []
labels = []

# Traverse through each actor's directory and get file paths and labels
for actor_dir in sorted(os.listdir(data_dir)):
    actor_path = os.path.join(data_dir, actor_dir)
    for file_name in sorted(os.listdir(actor_path)):
        file_path = os.path.join(actor_path, file_name)
        emotion = int(file_name.split("-")[2])
        labels.append(emotion)
        file_paths.append(file_path)

# Feature extraction function
def extract_features(file_path, sample_rate=48000):  
    attributes = file_path.split("-")
    actor_id = int(attributes[-1].split(".")[0])
    gender = actor_id % 2 

    # Load audio and trim silence
    signal, sr = librosa.load(file_path, sr=sample_rate)
    signal, _ = librosa.effects.trim(signal)

    n_fft = min(512, len(signal))

    mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, n_fft=n_fft).T, axis=0)
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=signal, sr=sr, n_fft=n_fft), axis=1)
    mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, n_mels=40), axis=1)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=n_fft))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr, n_fft=n_fft))
    tonnetz = np.mean(librosa.feature.tonnetz(y=signal, sr=sr), axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=signal))
    
    chroma_cqt = np.mean(librosa.feature.chroma_cqt(y=signal, sr=sr), axis=1)
    chroma_cens = np.mean(librosa.feature.chroma_cens(y=signal, sr=sr), axis=1)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=signal, sr=sr, n_fft=n_fft), axis=1)
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=signal, n_fft=n_fft))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr, n_fft=n_fft))
    rms = np.mean(librosa.feature.rms(y=signal, frame_length=n_fft))

    # Combine selected features
    features = np.hstack([
        mfccs, chroma_stft, mel_spectrogram, spectral_centroid, spectral_bandwidth, 
        tonnetz, zcr, chroma_cqt, chroma_cens, spectral_contrast, 
        spectral_flatness, spectral_rolloff, rms
    ])

    return features

# Parallelize feature extraction
features = Parallel(n_jobs=-1)(delayed(extract_features)(file_path) for file_path in tqdm(file_paths, desc="Extracting Features"))

X = np.array(features)
y = np.array(labels)

# Scale the features. Mean of 0 and Variance of 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the SVM model
svm_model = SVC(kernel='rbf', C=25)
svm_model.fit(X_train, y_train)

# Evaluate the model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

