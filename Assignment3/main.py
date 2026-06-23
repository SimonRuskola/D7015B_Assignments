import numpy as np
from pathlib import Path
import tensorflow as tf
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

# Try importing TdmsFile
try:
    from nptdms import TdmsFile
except ImportError:
    print("Installing nptdms...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'nptdms'])
    from nptdms import TdmsFile

# ============================================================================
# STEP 1: EXTRACT 7 FEATURES PER BLOCK
# ============================================================================

def extract_features(block):
    """
    Extract 7 time-domain features from a block of 600 samples.
    
    Features:
    1. RMS - Root Mean Square
    2. Skewness
    3. Kurtosis
    4. Shape factor - RMS / mean(|x|)
    5. Crest factor - max(|x|) / RMS
    6. Impulse factor - max(|x|) / mean(|x|)
    7. Clearance - max(|x|) / (mean(sqrt(|x|)))^2
    """
    N = len(block)
    m = np.mean(block)  # mean
    sigma = np.std(block, ddof=1)  # standard deviation
    
    # 1. RMS
    rms = np.sqrt(np.mean(block**2))
    
    # 2. Skewness
    skewness = np.sum((block - m)**3) / ((N - 1) * sigma**3) if sigma != 0 else 0
    
    # 3. Kurtosis
    kurtosis = np.sum((block - m)**4) / ((N - 1) * sigma**4) if sigma != 0 else 0
    
    # 4. Shape factor
    shape_factor = rms / np.mean(np.abs(block)) if np.mean(np.abs(block)) != 0 else 0
    
    # 5. Crest factor
    crest_factor = np.max(np.abs(block)) / rms if rms != 0 else 0
    
    # 6. Impulse factor
    impulse_factor = np.max(np.abs(block)) / np.mean(np.abs(block)) if np.mean(np.abs(block)) != 0 else 0
    
    # 7. Clearance (margin)
    sqrt_mean = np.mean(np.sqrt(np.abs(block)))
    clearance = np.max(np.abs(block)) / (sqrt_mean**2) if sqrt_mean != 0 else 0
    
    return np.array([rms, skewness, kurtosis, shape_factor, crest_factor, impulse_factor, clearance])


def load_tdms_file(filepath):
    """Load TDMS file and return accelerometer data."""
    tdms_file = TdmsFile.read(filepath)
    
    # TDMS file structure: groups and channels
    # Extract channels from the group 'Untitled'
    data = {}
    for group in tdms_file.groups():
        for channel in group.channels():
            channel_name = channel.name
            data[channel_name] = channel.data
    return data


def extract_bogie_window(data, sampling_rate=51200, window_duration=3):
    """
    Extract 3-second windows and reshape into 256 blocks of 600 samples.
    
    Returns:
    - features: (n_windows, 256, 7) - 256 time-steps with 7 features each
    """
    # Use Acceleration Mod1Ai0 (accelerometer A, X direction)
    ax = data['Acceleration Mod1Ai0']
    
    # Extract windows
    n_complete_windows = len(ax) // SAMPLES_PER_WINDOW
    
    all_window_features = []
    
    for w in range(n_complete_windows):
        start_idx = w * SAMPLES_PER_WINDOW
        end_idx = start_idx + SAMPLES_PER_WINDOW
        window_data = ax[start_idx:end_idx]
        
        # Reshape into 256 blocks of 600 samples
        window_data = window_data[:BLOCKS_PER_WINDOW * SAMPLES_PER_BLOCK]
        blocks = window_data.reshape((BLOCKS_PER_WINDOW, SAMPLES_PER_BLOCK))
        
        # Extract features for each block
        window_features = np.array([extract_features(block) for block in blocks])
        all_window_features.append(window_features)
    
    return np.array(all_window_features)  # Shape: (n_windows, 256, 7)


def select_assignment_windows(window_features, max_windows=WINDOWS_PER_FILE):
    """Keep the middle pass segment, matching the assignment's S1S2 crop."""
    n_windows = window_features.shape[0]
    if n_windows <= max_windows:
        return window_features

    start = (n_windows - max_windows) // 2
    end = start + max_windows
    return window_features[start:end]


def load_all_data():
    """Load all TDMS files and extract features."""
    data_dir = Path('Assignment3/data')
    labels = ['L0', 'L1', 'L2', 'L3']
    
    all_features = []
    all_labels = []
    
    for label in labels:
        files = sorted(list(data_dir.glob(f'{label}*.tdms')))
        print(f"Found {len(files)} files for {label}")
        
        for filepath in files:
            print(f"  Processing {filepath.name}...")
            data = load_tdms_file(filepath)
            features = extract_bogie_window(data)  # Shape: (n_windows, 256, 7)
            features = select_assignment_windows(features)
            
            all_features.append(features)
            # Each file contributes n_windows labels
            all_labels.extend([label] * features.shape[0])
    
    # Combine all data
    X = np.vstack(all_features)  # Shape: (total_windows, 256, 7)
    y = np.array(all_labels)
    
    # Convert labels to numeric (L0=0, L1=1, L2=2, L3=3)
    label_map = {'L0': 0, 'L1': 1, 'L2': 2, 'L3': 3}
    y_numeric = np.array([label_map[label] for label in y])
    
    return X, y_numeric


# ============================================================================
# STEP 2: BUILD THE LSTM
# ============================================================================

def build_lstm_model(input_shape):
    """
    Build LSTM model for wear classification with focal loss for class imbalance.
    
    Input shape: (256 time-steps, 7 features)
    Output: 4 classes (L0, L1, L2, L3)
    """
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(4, activation='softmax')  # 4 classes
    ])
    
    # Use Adam with moderate learning rate
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    # Use focal loss from tf_addons for better class imbalance handling
    # Fallback to weighted cross-entropy if focal loss not available
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("Loading data...")
    X, y = load_all_data()
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    # Train/test split (60/40)
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Normalize using training data only to avoid leaking test statistics.
    print("\nNormalizing features using the training set only...")
    scaler = StandardScaler()
    X_train_shape = X_train.shape
    X_test_shape = X_test.shape
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train_shape[-1])).reshape(X_train_shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test_shape[-1])).reshape(X_test_shape)
    
    # Build and train LSTM
    print("\nBuilding LSTM model...")
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.summary()
    
    # Count samples per class
    class_counts = np.bincount(y_train)
    print(f"\nTraining class distribution: {class_counts}")
    
    # Add early stopping
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
        callbacks=[early_stop]
    )
    
    # Save model
    print("\nSaving model...")
    model.save('Assignment3/output/wear_lstm_model.h5')
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Generate predictions and confusion matrix
    print("\nGenerating predictions...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Compute per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    class_names = ['L0', 'L1', 'L2', 'L3']
    print("\nPer-class accuracy:")
    for i, cls in enumerate(class_names):
        print(f"  {cls}: {per_class_accuracy[i]:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix (Overall Accuracy: {test_accuracy:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('Assignment3/output/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as 'Assignment3/output/confusion_matrix.png'")
    plt.close()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    print("\nDone! Model saved as 'Assignment3/output/wear_lstm_model.h5'")