import numpy as np
from pathlib import Path
import tensorflow as tf
from nptdms import TdmsFile
import pywt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

SAMPLE_RATE = 51200
WINDOW_DURATION = 3
SAMPLES_PER_WINDOW = SAMPLE_RATE * WINDOW_DURATION

SAMPLES_PER_BLOCK = 600
BLOCKS_PER_WINDOW = 256
WINDOWS_PER_FILE = 40
PREVIEW_SAMPLES = 5
TARGET_CROP_SECONDS = 120
EPOCHS = 100
BATCH_SIZE = 32


def load_tdms_file(filepath):
    tdms_file = TdmsFile.read(filepath)
    data = {}
    for group in tdms_file.groups():
        for channel in group.channels():
            channel_name = channel.name
            data[channel_name] = channel.data
    return data


def format_preview(values, max_items=PREVIEW_SAMPLES):
    preview = np.asarray(values[:max_items])
    return np.array2string(preview, precision=4, separator=', ')



def load_all_data():
    """Load all TDMS files and extract features."""
    data_dir = Path(__file__).resolve().parent / 'data'
    labels = ['L0', 'L1', 'L2', 'L3']

    loaded_data = []
    loaded_labels = []
    for tdms_file in sorted(data_dir.glob('*.tdms')):
        label = tdms_file.name.split('_', 1)[0]
        if label not in labels:
            continue

        data = load_tdms_file(tdms_file)
        loaded_data.append({
            'file': tdms_file.name,
            'channels': data,
        })
        loaded_labels.append(label)
    
    return loaded_data, loaded_labels

def print_data_summary(data, labels):
    """Print a summary of the loaded data."""
    print(f"Total files loaded: {len(data)}")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} files")

    for item, label in zip(data, labels):
        file_path = Path(__file__).resolve().parent / 'data' / item['file']
        file_size_kb = file_path.stat().st_size / 1024
        print(f"\n{item['file']} ({label})")
        print(f"  Size: {file_size_kb:.1f} KB")

        for channel_name, channel_data in item['channels'].items():
            sample_count = len(channel_data)
            preview = format_preview(channel_data)
            print(f"  {channel_name}: {sample_count} samples, first {PREVIEW_SAMPLES} = {preview}")


def crop_data(data, crop_seconds=TARGET_CROP_SECONDS):
    """Crop the centered middle section to a fixed duration in seconds."""

    target_samples = int(crop_seconds * SAMPLE_RATE)
    cropped_data = []
    for item in data:
        cropped_item = {
            'file': item['file'],
            'channels': {}
        }
        for channel_name, channel_data in item['channels'].items():
            channel_data = np.asarray(channel_data)
            sample_count = len(channel_data)

            window_samples = min(sample_count, target_samples)
            start_index = max((sample_count - window_samples) // 2, 0)
            end_index = start_index + window_samples
            cropped_item['channels'][channel_name] = channel_data[start_index:end_index]
        cropped_data.append(cropped_item)
    return cropped_data


def wavelet_denoise_signal(signal, wavelet='db4', level=None):
    """Denoise a 1D signal using wavelet thresholding."""

    signal = np.asarray(signal)
    if signal.size < 4:
        return signal.copy()

    max_level = pywt.dwt_max_level(signal.size, pywt.Wavelet(wavelet).dec_len)
    if max_level <= 0:
        return signal.copy()

    if level is None:
        level = min(4, max_level)
    else:
        level = min(level, max_level)

    coeffs = pywt.wavedec(signal, wavelet, level=level)
    if len(coeffs) < 2:
        return signal.copy()

    detail_coeffs = coeffs[1:]
    sigma = np.median(np.abs(detail_coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(signal.size))

    coeffs[1:] = [pywt.threshold(coeff, threshold, mode='soft') for coeff in detail_coeffs]
    denoised_signal = pywt.waverec(coeffs, wavelet)
    return denoised_signal[:signal.size]


def wavelet_denoise_data(data, wavelet='db4', level=None):
    """Apply wavelet denoising to every channel in every file."""

    denoised_data = []
    for item in data:
        denoised_item = {
            'file': item['file'],
            'channels': {},
        }
        for channel_name, channel_data in item['channels'].items():
            denoised_item['channels'][channel_name] = wavelet_denoise_signal(
                channel_data,
                wavelet=wavelet,
                level=level,
            )
        denoised_data.append(denoised_item)
    return denoised_data


def extract_block_features(block):
    """Compute the 7 requested time-domain features for one block."""

    block = np.asarray(block)
    n = block.size
    if n == 0:
        return np.zeros(7, dtype=float)

    eps = 1e-12
    mean_val = np.mean(block)
    abs_mean = np.mean(np.abs(block))
    rms = np.sqrt(np.mean(np.square(block)))
    sigma = np.std(block, ddof=1) if n > 1 else 0.0

    centered = block - mean_val
    denom = max(n - 1, 1)
    skewness = np.sum(centered ** 3) / (denom * (sigma ** 3 + eps))
    kurtosis = np.sum(centered ** 4) / (denom * (sigma ** 4 + eps))

    max_abs = np.max(np.abs(block))
    shape_factor = rms / (abs_mean + eps)
    crest_factor = max_abs / (rms + eps)
    impulse_factor = max_abs / (abs_mean + eps)
    clearance = max_abs / (np.mean(np.sqrt(np.abs(block))) ** 2 + eps)

    return np.array([
        rms,
        skewness,
        kurtosis,
        shape_factor,
        crest_factor,
        impulse_factor,
        clearance,
    ], dtype=float)


def windowing(data, window_size=SAMPLES_PER_WINDOW, block_size=SAMPLES_PER_BLOCK, blocks_per_window=BLOCKS_PER_WINDOW):
    """Split signals into windows and return per-window tensors of shape (256, 7)."""

    if block_size * blocks_per_window != window_size:
        raise ValueError("window_size must equal block_size * blocks_per_window")

    windowed_data = []
    for item in data:
        windowed_item = {
            'file': item['file'],
            'channels': {},
        }

        for channel_name, channel_data in item['channels'].items():
            channel_data = np.asarray(channel_data)
            num_windows = len(channel_data) // window_size
            channel_window_features = []

            for window_idx in range(num_windows):
                start = window_idx * window_size
                end = start + window_size
                window_signal = channel_data[start:end]
                blocks = window_signal.reshape(blocks_per_window, block_size)

                block_features = np.vstack([extract_block_features(block) for block in blocks])
                channel_window_features.append(block_features)

            if channel_window_features:
                windowed_item['channels'][channel_name] = np.stack(channel_window_features, axis=0)
            else:
                windowed_item['channels'][channel_name] = np.empty((0, blocks_per_window, 7), dtype=float)

        windowed_data.append(windowed_item)

    return windowed_data

    






    


def build_lstm_model(input_shape):
    """Build and compile the LSTM model."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(4, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def build_sequence_split_dataset(windowed_data, labels, train_ratio=0.6):
    """Build train/validation sets by splitting each file sequence temporally."""

    label_names = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}

    channel_sets = [set(item['channels'].keys()) for item in windowed_data]
    common_channels = sorted(set.intersection(*channel_sets))
    if not common_channels:
        raise ValueError('No common channels across all files for sequence building.')

    x_train, y_train = [], []
    x_val, y_val = [], []

    for item, label in zip(windowed_data, labels):
        per_channel = [item['channels'][channel] for channel in common_channels]
        num_windows = min(channel_tensor.shape[0] for channel_tensor in per_channel)
        if num_windows < 2:
            continue

        sequence_tensor = np.concatenate(
            [channel_tensor[:num_windows] for channel_tensor in per_channel],
            axis=2,
        )

        split_idx = int(np.floor(num_windows * train_ratio))
        split_idx = min(max(split_idx, 1), num_windows - 1)

        x_train.extend(sequence_tensor[:split_idx])
        y_train.extend([label_to_idx[label]] * split_idx)

        x_val.extend(sequence_tensor[split_idx:])
        y_val.extend([label_to_idx[label]] * (num_windows - split_idx))

    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    x_val = np.asarray(x_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int32)

    if x_train.size == 0 or x_val.size == 0:
        raise ValueError('Not enough windowed data to build train/validation sets.')

    return x_train, y_train, x_val, y_val, label_names, common_channels


def standardize_sequence_features(x_train, x_val):
    """Standardize features using train-set statistics only."""

    n_train, t_steps, n_features = x_train.shape
    n_val = x_val.shape[0]

    scaler = StandardScaler()
    x_train_2d = x_train.reshape(-1, n_features)
    x_val_2d = x_val.reshape(-1, n_features)

    x_train_scaled = scaler.fit_transform(x_train_2d).reshape(n_train, t_steps, n_features)
    x_val_scaled = scaler.transform(x_val_2d).reshape(n_val, t_steps, n_features)
    return x_train_scaled, x_val_scaled, scaler


def save_confusion_matrix_artifacts(y_true, y_pred, label_names, output_dir):
    """Save confusion matrix image and CSV to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_names)))

    cm_csv_path = output_dir / 'confusion_matrix.csv'
    np.savetxt(cm_csv_path, cm, delimiter=',', fmt='%d')

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('LSTM Confusion Matrix')
    plt.tight_layout()
    cm_png_path = output_dir / 'confusion_matrix.png'
    plt.savefig(cm_png_path)
    plt.close()

    return cm, cm_csv_path, cm_png_path


def compute_per_class_accuracy(cm, label_names):
    """Compute per-class accuracy as recall for each class."""

    per_class = {}
    for idx, label in enumerate(label_names):
        class_total = np.sum(cm[idx, :])
        if class_total == 0:
            per_class[label] = 0.0
        else:
            per_class[label] = cm[idx, idx] / class_total
    return per_class


if __name__ == '__main__':

    print("Loading data...")
    data, labels = load_all_data()  
    print_data_summary(data, labels)

    
    denoised_data = wavelet_denoise_data(data)

    print("\nDenoised data summary:")
    print_data_summary(denoised_data, labels)


    cropped_data = crop_data(denoised_data)

    print("\nCropped data summary:")
    print_data_summary(cropped_data, labels)

    windowed_feature_data = windowing(cropped_data)
    print("\nWindow feature tensor shapes (num_windows, 256, 7):")
    for item in windowed_feature_data:
        channel_name = next(iter(item['channels']))
        print(f"{item['file']} - {channel_name}: {item['channels'][channel_name].shape}")

    x_train, y_train, x_val, y_val, label_names, used_channels = build_sequence_split_dataset(
        windowed_feature_data,
        labels,
        train_ratio=0.6,
    )
    print(f"\nUsing channels for training: {used_channels}")
    print(f"Train samples: {x_train.shape[0]}, Validation samples: {x_val.shape[0]}")

    x_train, x_val, scaler = standardize_sequence_features(x_train, x_val)

    model = build_lstm_model(input_shape=x_train.shape[1:])
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    val_pred_prob = model.predict(x_val, verbose=0)
    y_pred = np.argmax(val_pred_prob, axis=1)
    val_acc = accuracy_score(y_val, y_pred)
    print(f"\nValidation accuracy: {val_acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_val, y_pred, target_names=label_names, digits=4))

    output_dir = Path(__file__).resolve().parent / 'output'
    cm, cm_csv_path, cm_png_path = save_confusion_matrix_artifacts(
        y_val,
        y_pred,
        label_names,
        output_dir,
    )

    print("\nConfusion matrix:")
    print(cm)
    per_class_acc = compute_per_class_accuracy(cm, label_names)
    print("\nPer-class accuracy:")
    for label in label_names:
        print(f"{label}: {per_class_acc[label]:.4f}")
    print(f"Saved confusion matrix CSV to: {cm_csv_path}")
    print(f"Saved confusion matrix image to: {cm_png_path}")


