
'''
File for training original CNN and also running inference on any .h5 model
'''

import pickle
import pprint
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, Conv1D, Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle
import matplotlib
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import butter, lfilter, get_window
from scipy import signal as sg
import subprocess
import csv
import os


## TRAINING DATASET 1 - preprocessing PPG-Dalia
def preprocess_data_dalia(session_name):
    session_file = '/Users/jamborghini/Documents/PYTHON/TRAINING DATA - PPG Dalia/' + session_name + '.pkl'

    with open(session_file, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    # pprint.pprint(data)

    # from scientific paper where PPG_Dalia comes from
    fs_ppg = 64  # Hz
    fs_acc = 32  # Hz
    fs_ecg = 0.5 # effective sampling rate given measured every 2 seconds

    time_analyse_seconds = 7900
    num_ppg = int(time_analyse_seconds * fs_ppg)          # number of datapoints
    num_acc = int(time_analyse_seconds * fs_acc)
    num_ecg = int(time_analyse_seconds * fs_ecg)

    # extract data
    ppg_data = data['signal']['wrist']['BVP']
    ppg_data = ppg_data[:num_ppg]
    ppg_data = np.squeeze(ppg_data)
    #ppg_data = np.array(ppg_data)
    #print(ppg_data.shape)

    # make sure it's in the same form
    def unpack(signal):
        data = signal.reshape(-1, 1)
        data = np.squeeze(data)
        return data

    acc_data = data['signal']['wrist']['ACC']
    x_data = acc_data[:, 0][:num_acc]
    x_data = unpack(x_data)
    y_data = acc_data[:, 1][:num_acc]
    y_data = unpack(y_data)
    z_data = acc_data[:, 2][:num_acc]
    z_data = unpack(z_data)

    ecg_ground_truth = data['label']
    ecg_ground_truth = ecg_ground_truth[:num_ecg-3]   # -3 because of windowing. Make this dynamic eventually

    return ppg_data, x_data, y_data, z_data, ecg_ground_truth, fs_ppg, fs_acc, num_ppg, num_acc

## TRAINING DATASET 2 - preprocessing Wrist PPG
def preprocess_data_wrist_ppg(session_name):
    file_path = '/Users/jamborghini/Documents/PYTHON/TESTING DATA - Wrist PPG Dataset/'
    file_name = file_path + session_name
    record = wfdb.rdrecord(file_name)

    # from .hea file: 0 = ecg, 1 = ppg, 2-4 = gyro, 5-7 = 2g accelerometer, 8-10 = 16g accelerometer
    df = pd.DataFrame({
        'ppg': record.adc()[:, 0],
        'x': record.adc()[:, 5],
        'y': record.adc()[:, 6],
        'z': record.adc()[:, 7]
    })
    ppg_data = df['ppg']
    x_data = df['x']
    y_data = df['y']
    z_data = df['z']

    with open('/Users/jamborghini/Documents/PYTHON/Fatigue Model/' + session_name +'_heart_rate_wrist_ppg.pkl', 'rb') as file:
        ecg_ground_truth = pickle.load(file, encoding='latin1')

    fs_ppg = 256  # from the paper
    fs_acc = 256
    num_ppg = len(ppg_data)
    num_acc = len(x_data)
    time_analyse_seconds = len(ppg_data) / fs_ppg  # this is the total time in seconds

    return ppg_data, x_data, y_data, z_data, ecg_ground_truth, fs_ppg, fs_acc, num_ppg, num_acc

## TRAINING DATASET 3 - preprocessing WESAD
def preprocess_data_wesad(session_name):

    ppg_file = '/Users/jamborghini/Documents/PYTHON/TRAINING DATA - WESAD/'+session_name+'/'+session_name+'_E4_Data/BVP.csv'
    acc_file = '/Users/jamborghini/Documents/PYTHON/TRAINING DATA - WESAD/'+session_name+'/'+session_name+'_E4_Data/ACC.csv'

    df_ppg = pd.read_csv(ppg_file)
    ppg_data = df_ppg.iloc[1:, 0]
    df_acc = pd.read_csv(acc_file)
    x_data = df_acc.iloc[1:, 0]
    y_data = df_acc.iloc[1:, 1]
    z_data = df_acc.iloc[1:, 2]
    print(len(ppg_data))
    print(len(x_data))

    with open('/Users/jamborghini/Documents/PYTHON/Fatigue Model/' + session_name +'_heart_rate_wesad.pkl', 'rb') as file:
        ecg_ground_truth = pickle.load(file, encoding='latin1')

    fs_ppg = 64  # Hz
    fs_acc = 32  # Hz
    num_ppg = len(ppg_data)
    num_acc = len(x_data)
    time_analyse_seconds = len(ppg_data) / fs_ppg  # this is the total time in seconds
    print(time_analyse_seconds)

    return ppg_data, x_data, y_data, z_data, ecg_ground_truth, fs_ppg, fs_acc, num_ppg, num_acc

## DATASET 4 - preprocessing Pulse Transit
def preprocess_data_pulse_transit(csv_file):

    def band_pass_filter(signal, fs, order=5):
        lowcut = 0.5  # this is to get rid of the low frequency 'hovering' to centre around 0
        highcut = 50

        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist

        b, a = butter(order, [low, high], btype='band')
        filtered = lfilter(b, a, signal)
        return filtered

    # NOTE - this dataset is recorded on finger not wrist like every other one. Seeing if it can be used well. Hopefully it works
    df = pd.read_csv(csv_file)
    ppg_data = df['pleth_6']  # chose 6 because it's proximal part of finger and green light. But open to trying different columns
    x_data = df['a_x']
    y_data = df['a_y']
    z_data = df['a_z']
    peaks = df['peaks']
    # ecg = df['ecg']

    fs = 500
    fs_ppg = 500
    fs_acc = 500
    num_ppg = len(ppg_data)
    num_acc = len(x_data)
    time_analyse_seconds = num_ppg / fs_ppg
    #print(len(peaks))

    # NORMALISATION
    # shift ppg & accel data down to around 0
    ppg_data = ppg_data - np.mean(ppg_data)
    x_data = x_data - np.mean(x_data)
    y_data = y_data - np.mean(y_data)
    z_data = z_data - np.mean(z_data)

    # get rid of edge effects (big peaks) at the start owing to the butterworth filter
    window = get_window('tukey', len(ppg_data))

    windowed_ppg = ppg_data * window
    windowed_x = x_data * window
    windowed_y = y_data * window
    windowed_z = z_data * window

    ppg_data = band_pass_filter(windowed_ppg, fs)
    x_data = band_pass_filter(windowed_x, fs)
    y_data = band_pass_filter(windowed_y, fs)
    z_data = band_pass_filter(windowed_z, fs)

    # find the ground truth heartrate using the CSV
    peak_indices = np.where(df['peaks'].to_numpy() == 1)[0]  # indices in the peak column where the peaks are
    window_size = 8 * fs  # 8-second window size
    step_size = 2 * fs  # 2-second step size
    ecg_ground_truth = []


    for i in range(0, len(peaks) - window_size + 1, step_size):
        window_r_peaks = [peak for peak in peak_indices if i <= peak < i + window_size]
        # ^ 'list comprehension' - remember this format

        if len(window_r_peaks) >= 2:
            window_bpm = (60 * fs) / np.average(np.diff(
                window_r_peaks))  # average of diff function is key to showing how HR changes, rather than just doing *60 calc
            ecg_ground_truth.append(window_bpm)

    # plt.plot(ecg_ground_truth)
    # plt.show()

    return ppg_data, x_data, y_data, z_data, ecg_ground_truth, fs_ppg, fs_acc, num_ppg, num_acc

def process_for_cnn(ppg_data, x_data, y_data, z_data, ecg_ground_truth, fs_ppg, fs_acc, num_ppg, num_acc):
## Need to turn this into a class / self ASAP

    def calculate_fft(data):
        result = np.fft.fft(data)  # take full FFT not just the positive part
        return result

    def trim_fft(data, num_segments):
        # only keep 0-4Hz
        trim_index = 256 * num_segments +1         # make dynamic when you set classes
        result = data[:trim_index]
        return result

    # get the original signal into segments + take FFT
    def segment_fft(data, fs, num_data):
        segment_size_seconds = 8
        segment_step_seconds = 2
        segment_size = segment_size_seconds * fs       # better to keep in terms of indices not time
        segment_step = segment_step_seconds * fs
        num_windows = ((num_data - segment_size) // segment_step) +1                  # easy to visualise this formula

        num_segments = 1
        segments = []

        for i in range(num_windows):
            # now running as (i-1, i, i+1)
            # if i == 0:
            #     start_idx = i
            #     end_idx = start_idx + (segment_size * (num_segments - 1))
            # else:
            #     start_idx = (i-1) * segment_step
            #     end_idx = start_idx + (segment_size * (num_segments - 1))

            start_idx = i * segment_step
            end_idx = start_idx + (segment_size * num_segments)
            segment = data[start_idx:end_idx]
            segments.append(segment)    # append each segment (row) into the matrix of all segments

        # get rows of segments into array format
        segments = np.array(segments)
        #print(f'Segments shape:  {segments.shape}')

        # zero padding to original signal increased frequency resolution before FFT, based on wanting 0-4Hz
        desired_freq = 4
        zeros_to_add = int((256*fs*num_segments)/desired_freq - (fs*segment_size_seconds*num_segments))  # see laptop notes + cancel out terms for derivation
        #print(f'Zeros to add:  {zeros_to_add}')
        # 3584 for PPG, 1792 for accel when 1 segment
        segments_padded = []

        for i in range(segments.shape[0]):
            segment = segments[i]
            num_zeros = np.zeros(zeros_to_add)
            padded_row = np.concatenate((segment, num_zeros))
            segments_padded.append(padded_row)

        segments_padded = np.array(segments_padded)
        segments_fft = []
        segments_normalised = []

        # calculate FFT independently on each segment
        for i in range(segments_padded.shape[0]):
            segment_padded = segments_padded[i]
            segment_fft = calculate_fft(segment_padded)
            segments_fft.append(segment_fft)

        segments_fft = np.array(segments_fft)

        #print(f'FFT shape:  {segments_fft.shape}')

        # final processing
        for i in range(segments_fft.shape[0]):
            segment_fft = segments_fft[i]
            # cut down to 0-4Hz
            segment_fft = trim_fft(segment_fft, num_segments)

            # z-normalization individually on each new trimmed segment (makes sense)
            mean = np.mean(segment_fft, axis=0)
            std_dev = np.std(segment_fft, axis=0)
            segment_normalised = (segment_fft - mean) / std_dev
            segments_normalised.append(segment_normalised)

        result = np.array(segments_normalised)

        return result

    ppg_input = segment_fft(ppg_data, fs_ppg, num_ppg)
    x_input = segment_fft(x_data, fs_acc, num_acc)
    y_input = segment_fft(y_data, fs_acc, num_acc)
    z_input = segment_fft(z_data, fs_acc, num_acc)

    input_all_channels = np.stack([ppg_input, x_input, y_input, z_input], axis=0)

    # reshape data to (Ntr, Nch, NFFT) as per the paper
    # input_all_channels = np.transpose(input_all_channels, (1, 0, 2))   # CONV2D option
    input_all_channels = np.transpose(input_all_channels, (1, 2, 0))     # CONV1D option

    # convert any NaN from the FFT to zero
    input_all_channels = np.nan_to_num(input_all_channels, nan=0)
    print(f'All channels shape:  {input_all_channels.shape}')

    label_ecg_data = ecg_ground_truth

    # Error handling - find exact locations of any 'NaN'
    # nan_indices = np.argwhere(np.isnan(input_all_channels))
    # if len(nan_indices) > 0:
    #     for index in nan_indices:
    #         print(f"NaN found at index {index} - Value: {input_all_channels[tuple(index)]}")

    assert not np.any(np.isnan(input_all_channels)), "NaN."

    return input_all_channels, label_ecg_data


#### start of model training section. ####
'''

def compute_regression_metrics(model, test_data, label_ecg_data):
    # GOAL OF FUNCTION = compare test data to labelling to get mean absolute error
    predictions = CNN_model.predict(test_data)
    mae = mean_absolute_error(label_ecg_data, predictions)
    return predictions, mae

def CNN_model():
    # This is the model architecture itself (simpler / downsized model as designed for an embedded system)

    input_shape = (4, 257, 1)
    # (should set 257 (NFFT) to be dynamic)
    print("Input shape:", input_shape)

    model = models.Sequential()

    # 1. initialise model

    # Convolution Layer
    model.add(layers.Conv2D(filters=8, kernel_size=(1,3), strides=(1,1), activation='relu', input_shape=input_shape))
    # Max-Pooling Layer
    model.add(layers.MaxPooling2D(pool_size=(1,2), strides=(1,2)))

    # 2. repeat the pattern
    NL = 3   # number of repetitions

    for i in range(1, NL+1):
        # Convolutional Layer
        n_filters = 2 ** (i + 3)        #this function doubles the number of filters with each layer (pick out more features)
        model.add(layers.Conv2D(filters=n_filters, kernel_size=(1,3), strides=(1,1), activation='relu'))

        # Max-Pooling Layer
        model.add(layers.MaxPooling2D(pool_size=(1,2), strides=(1,2)))

    # 3. final layers

    #Final Convolutional Layer
    model.add(layers.Conv2D(filters=16, kernel_size=(1,1), strides=(1,1), activation='relu'))

    ## 2. LSTM layers
    lstm_units = 64
    model.add(layers.Reshape((-1, n_filters)))  # Reshape for LSTM layer into (batch, timesteps, features)
    model.add(layers.LSTM(lstm_units, return_sequences=True))
    model.add(layers.LSTM(lstm_units, return_sequences=True))
    model.add(layers.LSTM(lstm_units, return_sequences=True))

    # Flattening Layer
    model.add(layers.Flatten())

    # fully connected layers are essential for 'labelling' the data - i.e. learning to associate features with a particular category
    # Fully Connected Layer 1
    n1f_c = 64  # number of neurons
    model.add(layers.Dense(n1f_c, activation='relu'))

    # Add Dropout Layer after the first fully connected layer (more useful when there's lots of layers)
    # dropout_rate = 0.5  # You can adjust the dropout rate as needed
    # model.add(layers.Dropout(rate=dropout_rate))

    # Fully Connected Layer 2
    n2f_c = 1
    model.add(layers.Dense(n2f_c, activation='linear'))

    # Compile the model
    learning_rate = 0.001
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    # adam is stochastic gradient descent

    model.summary()

    ### NOTE the weights are float32

    return model

#### TRAINING THE MODEL ####

# set up tracker for training + validation accuracy
class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.train_acc = []
        self.val_acc = []

    def on_epoch_end(self, batch, logs={}):
        self.train_acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))

# Model initialisation
num_epochs = 10  # number of times the dataset is passed through the model (forward + backwards passes) - tweak as necessary (higher = better)
batch_size = 32  # number of examples processed per backwards and forwards pass
losses = []
val_losses = []
mae_scores = []
train_accuracies = []
val_accuracies = []

session_data = {}
label_session_data = {}
test_data = []
test_labels = []
session_keys = []

# #### TRAINING DATASET 2 - PULSE TRANSIT TIME PPG ####
# file_names = ['s{}_{}.csv'.format(i, activity) for i in range(1, 22) for activity in ['sit', 'walk', 'run']]
# file_path = '/Users/jamborghini/Documents/PYTHON/pulse transit time ppg dataset/csv/'
# csv_files = [file_path + file_name for file_name in file_names]
#
# for session_file in csv_files:
#     session_key = os.path.basename(session_file).replace('.csv', '')  # little trick to not export the whole file path
#     input_all_channels, label_ecg_data = process_for_cnn(*preprocess_data_pulse_transit(session_file))  # cleaner way instead of typing out the function parameters in full
#     # * is the UNPACKING operand
#     session_data[session_key] = input_all_channels
#     label_session_data[session_key] = label_ecg_data  # Assign label data - KEY MISSED ERROR FROM BEFORE
#     session_keys.append(session_key)


#### TRAINING DATASET 1 - PPG DALIA ####

# process each file and save for model input
file_names = ['S{}'.format(i) for i in range(1, 16) if i != 6]
pkl_files = [file_name for file_name in file_names]
## NOTE - for S6 the data is incomplete / corrupted (approx last hour)


# Assign session keys as numbers from 1 to 15 (skipping 6)
for i, session_file in enumerate(pkl_files, start=1):
    session_key = i
    session_keys.append(session_key)
    print(f'Session number:  {session_key}')
    input_all_channels, label_ecg_data = process_for_cnn(*preprocess_data_dalia(session_file))  # cleaner way instead of typing out the function parameters in full
    session_data[session_key] = input_all_channels    # Assign session data
    label_session_data[session_key] = label_ecg_data  # Assign label data
    #print(session_data[session_key].shape)
    #print(label_session_data[session_key].shape)


# Create and compile the CNN model
def train_and_evaluate_model(train_data, train_labels, test_data, test_labels):

    # shuffle the order of the SEGMENTS to combat overfitting
    indices = np.random.permutation(len(train_data))
    train_data = train_data[indices]
    train_labels = train_labels[indices]

    model = CNN_model()

    accuracy_history = AccuracyHistory()       # for tracking accuracy

    # Train the model on the training data
    history = model.fit(
        train_data,
        train_labels,
        epochs=num_epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(test_data, test_labels),
        verbose=1,
        callbacks=[accuracy_history]
    )

    # Calculate Mean Absolute Error (MAE) on the test data
    estimated_bpm = model.predict(test_data)
    mae = mean_absolute_error(test_labels, estimated_bpm)
    return mae, history, model, estimated_bpm

# Iterate through sessions and train the model
for session_key in session_keys:
    # Load data for the current session + assign it as the testing set
    print(f'Session number: {session_key}/{len(session_keys)}')
    test_data = session_data[session_key]
    test_labels = label_session_data[session_key]
    print(f'Test data shape: {test_data.shape}')
    print(f'Test labels shape: {test_labels.shape}')

    # Assign the remaining sessions as the training data (LOSO method)
    train_sessions = [key for key in session_keys if key != session_key]

    train_data = []
    train_labels = []

    for train_session_key in train_sessions:
        train_data.append(session_data[train_session_key])
        train_labels.append(label_session_data[train_session_key])

    # combine all the train data into one long array to iterate through (makes logical sense)
    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    print(f'Train data shape:  {train_data.shape}')
    print(f'Train labels shape:  {train_labels.shape}')

    # finally - pass training data through the model
    mae, history, model, estimated_bpm = train_and_evaluate_model(train_data, train_labels, test_data, test_labels)
    print(f'MAE for session:  {mae}')
    mae_scores.append(mae)
    # plot to check the first iteration as a sense-check
    #plot_results(session_key, test_labels, estimated_bpm, model)


model_name = 'HR_model_21_a'  # Set your desired model name
model.save('/Users/jamborghini/Documents/PYTHON/Trained Models/'+model_name+'.h5')

print("Mean Absolute Errors for Each Session:")
for i, mae in enumerate(mae_scores):
    print(f"Session {session_keys[i]}: {mae:.2f}")      # display MAE for each LOSO rotation to 2 decimal points

    #### end of model training section. ####


    '''

    #### PASS DATA THROUGH THE MODEL - aka 'running' the model #### (comment out the training section when using) ####

# plotting results (check MAE is accurate)
def plot_results(session_name, actual_bpm, estimated_bpm, model):
    # smooth_window = 10
    # smoothed_estimated_bpm = np.convolve(estimated_bpm[:, 0], np.ones(smooth_window) / smooth_window, mode='same')
    plt.plot(estimated_bpm, color='red', label='Estimated')
    plt.plot(actual_bpm, color='black', label='Actual')
    plt.ylabel("Heart Rate (BPM)")
    plt.legend()
    plt.title(f"{session_name} - Estimated vs. Actual BPM")
    plt.show()

# change session file directory & preprocessing function & model name

## ppg dalia
# session_name = 'S5'
# input_all_channels, actual_bpm = process_for_cnn(*preprocess_data_dalia(session_name))

# ## wrist ppg
session_name = 's6_low_resistance_bike'
input_all_channels, actual_bpm = process_for_cnn(*preprocess_data_wrist_ppg(session_name))

# ## WESAD
# session_name = 'S5'
# input_all_channels, actual_bpm = process_for_cnn(*preprocess_data_wesad(session_name))

## pulse transit
# session_file = '/Users/jamborghini/Documents/PYTHON/pulse transit time ppg dataset/csv/'+session_name+'.csv'
# input_all_channels, actual_bpm = process_for_cnn(*preprocess_data_pulse_transit(session_file))

print(input_all_channels.shape)
model_name = 'HR_model_Aa'
model = keras.models.load_model('/Users/jamborghini/Documents/PYTHON/Trained Models/' + model_name + '.h5')

# Print layer details
# for i, layer in enumerate(model.layers):
#     print(f"Layer {i} - {layer.name}:")
#     for weight in layer.weights:
#         print(f"  Weight Name: {weight.name}, Shape: {weight.shape}, Type: {weight.dtype}")
#     print()

estimated_bpm = model.predict(input_all_channels)
# equate length of test + label data
# estimated_bpm = estimated_bpm[:len(actual_bpm)]
print(f'Estimated BPM shape: {estimated_bpm.shape}')

# print(len(actual_bpm))
# print(len(estimated_bpm))
mae = mean_absolute_error(estimated_bpm, actual_bpm)
print(f'MAE:  {mae}')

plot_results(session_name, actual_bpm, estimated_bpm, model)


Host *.github.com
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_ed25519
