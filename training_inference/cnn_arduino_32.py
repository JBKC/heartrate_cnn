
'''
Streamlined model that takes up < 40KB
Two steps: (1) train model; (2) compress model
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
import tensorflow_model_optimization as tfmot
import time

def preprocess_data_dalia(session_name):
    '''
    Extract session data from PPG Dalia .pkl files for training
    '''

    session_file = session_name + '.pkl'   # adjust pathname as necessary

    with open(session_file, 'rb') as file:
        data = pickle.load(file, encoding='latin1')

    # sampling rates given by research paper
    fs_ppg = 64  # Hz
    fs_acc = 32  # Hz
    fs_ecg = 0.5 # Hz (effective sampling rate)

    time_analyse_seconds = 7900                        # total time period of session
    # calculate no. of datapoints
    num_ppg = int(time_analyse_seconds * fs_ppg)      
    num_acc = int(time_analyse_seconds * fs_acc)
    num_ecg = int(time_analyse_seconds * fs_ecg)

    # extract PPG sensor data
    ppg_data = data['signal']['wrist']['BVP']
    ppg_data = ppg_data[:num_ppg]
    ppg_data = np.squeeze(ppg_data)

    # reshape data to remove unwanted dimensions
    def unpack(signal):
        data = signal.reshape(-1, 1)
        data = np.squeeze(data)
        return data

    # extract accelerometer sensor data
    acc_data = data['signal']['wrist']['ACC']
    x_data = acc_data[:, 0][:num_acc]
    x_data = unpack(x_data)
    y_data = acc_data[:, 1][:num_acc]
    y_data = unpack(y_data)
    z_data = acc_data[:, 2][:num_acc]
    z_data = unpack(z_data)

    # assign ground truth labels
    ecg_ground_truth = data['label']
    ecg_ground_truth = ecg_ground_truth[:num_ecg-3]

    return ppg_data, x_data, y_data, z_data, ecg_ground_truth, fs_ppg, fs_acc, num_ppg, num_acc

def preprocess_data_wrist_ppg(session_name):
    '''
    Extract session data from Wrist PPG .pkl files for testing
    '''
    
    file_name = session_name                # adjust file path as necessary
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

    with open(session_name +'.pkl', 'rb') as file:                    # adjust file path as necessary
        ecg_ground_truth = pickle.load(file, encoding='latin1')

    # sampling rates given by research paper
    fs_ppg = 256 
    fs_acc = 256
    num_ppg = len(ppg_data)
    num_acc = len(x_data)
    time_analyse_seconds = len(ppg_data) / fs_ppg

    return ppg_data, x_data, y_data, z_data, ecg_ground_truth, fs_ppg, fs_acc, num_ppg, num_acc

def process_for_cnn(ppg_data, x_data, y_data, z_data, ecg_ground_truth, fs_ppg, fs_acc, num_ppg, num_acc):
    '''
    Generic processing function to get data ready for model input
    '''

    def calculate_fft(data):
        '''
        Return the FFT (frequency components) of input signal
        '''
        
        result = np.fft.fft(data)  
        return result

    def trim_fft(data):
        '''
        Trim frequency data to the range of interest to spot heartbeats (0-4Hz)
        '''
        
        trim_index = 256 +1         
        result = data[:trim_index]
        return result

    def segment_fft(data, fs, num_data):
        '''
        Cut time-domain signal into segments and run processing
        '''
        
        segments = []
        segment_size_seconds = 8
        segment_step_seconds = 2
        segment_size = segment_size_seconds * fs       
        segment_step = segment_step_seconds * fs
        num_windows = ((num_data - segment_size) // segment_step) +1             

        # Create segments
        for i in range(num_windows):
            start_idx = i * segment_step
            end_idx = start_idx + segment_size
            segment = data[start_idx:end_idx]
            segments.append(segment)  
        segments = np.array(segments)

        # Zero-pad signal to increase frequency resolution
        segments_padded = []
        desired_freq = 4
        zeros_to_add = int((256*fs)/desired_freq - (fs*segment_size_seconds))

        for i in range(segments.shape[0]):
            segment = segments[i]
            num_zeros = np.zeros(zeros_to_add)
            padded_row = np.concatenate((segment, num_zeros))
            segments_padded.append(padded_row)

        segments_padded = np.array(segments_padded)

        segments_fft = []
        segments_normalised = []

        # calculate FFT independently on each segment
        forCi in range(segments_padded.shape[0]):
            segment_padded = segments_padded[i]
            segment_fft = calculate_fft(segment_padded)
            segments_fft.append(segment_fft)

        segments_fft = np.array(segments_fft)

        # Trim & z-normalise signal
        for i in range(segments_fft.shape[0]):
            segment_fft = segments_fft[i]
            segment_fft = trim_fft(segment_fft)
            mean = np.mean(segment_fft, axis=0)
            std_dev = np.std(segment_fft, axis=0)
            segment_normalised = (segment_fft - mean) / std_dev
            segments_normalised.append(segment_normalised)

        result = np.array(segments_normalised)

        return result

    # Retrieve inputs for model
    ppg_input = segment_fft(ppg_data, fs_ppg, num_ppg)
    x_input = segment_fft(x_data, fs_acc, num_acc)
    y_input = segment_fft(y_data, fs_acc, num_acc)
    z_input = segment_fft(z_data, fs_acc, num_acc)

    # Stack results into tensor
    input_all_channels = np.stack([ppg_input, x_input, y_input, z_input], axis=0)

    # Error handling
    input_all_channels = np.nan_to_num(input_all_channels, nan=0)
    assert not np.any(np.isnan(input_all_channels)), "NaN."

    # Reshape data for Conv1D
    input_all_channels = np.transpose(input_all_channels, (1, 2, 0))

    # Label each segment in the input data
    label_ecg_data = ecg_ground_truth

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

    input_shape = (257,4)
    # (should set 257 (NFFT) to be dynamic)
    print("Input shape:", input_shape)

    model = models.Sequential()

    # 1. initialise model

    # Convolution Layer
    model.add(layers.Conv1D(filters=8, kernel_size=(1), strides=(1), activation='relu', input_shape=input_shape))
    # Max-Pooling Layer
    model.add(layers.MaxPooling1D(pool_size=(2), strides=(2)))

    # 2. repeat the pattern
    NL = 3   # number of repetitions

    for i in range(1, NL+1):
        # Convolutional Layer
        n_filters = 2 ** (i + 3)        #this function doubles the number of filters with each layer (pick out more features)
        model.add(layers.Conv1D(filters=n_filters, kernel_size=(3), strides=(1), activation='relu'))

        # Max-Pooling Layer
        model.add(layers.MaxPooling1D(pool_size=(2), strides=(2)))

    # 3. final layers

    #Final Convolutional Layer
    model.add(layers.Conv1D(filters=16, kernel_size=(1), strides=(1), activation='relu'))

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


model_name = 'HR_model_Aa'  # Set your desired model name
model.save('/Users/jamborghini/Documents/PYTHON/Trained Models/'+model_name+'.h5')

print("Mean Absolute Errors for Each Session:")
for i, mae in enumerate(mae_scores):
    print(f"Session {session_keys[i]}: {mae:.2f}")      # display MAE for each LOSO rotation to 2 decimal points

        #### end of model training section. ####

# Iterate through sessions and train the model


'''
    #### PASS DATA T00HROUGH THE MODEL - aka 'running' the model #### (comment out the training section when using) ####

# plotting results (check MAE is accurate)
def plot_results(session_name, actual_bpm, estimated_bpm):
    # smooth_window = 10
    # smoothed_estimated_bpm = np.convolve(estimated_bpm[:, 0], np.ones(smooth_window) / smooth_window, mode='same')

    plt.plot(estimated_bpm, color='red', label='Estimated')
    plt.plot(actual_bpm, color='black', label='Actual')
    plt.ylabel("Heart Rate (BPM)")
    plt.legend()
    plt.title(f"{session_name} - Estimated vs. Actual BPM")
    plt.show()

# change session file directory & preprocessing function & model name

## ppg dalia (original data)
# session_name = 'S5'
# input_all_channels, actual_bpm = process_for_cnn(*preprocess_data_dalia(session_name))

# wrist ppg (unseen data)
session_name = 's1_low_resistance_bike'
input_all_channels, actual_bpm = process_for_cnn(*preprocess_data_wrist_ppg(session_name))

model_name = 'HR_model_Aa'
# uncomment this line for quantizing
# model = keras.models.load_model('/Users/jamborghini/Documents/PYTHON/Trained Models/'+model_name+'.h5')
# uncomment this line for inference
tflite_model_path = '/Users/jamborghini/Documents/PYTHON/Trained Models/'+model_name+'_quantized.tflite'


def generate_sample_input():
    #input_data = np.expand_dims(input_all_channels, axis=-1)   # used for CONV2D
    input_data = np.float32(input_all_channels)
    print(input_data.shape)
    yield [input_data]

representative_dataset = tf.lite.RepresentativeDataset(generate_sample_input)

# quantization function
def post_training_quantize(model, representative_dataset):

    '''
    Function for quantizing an already-trained model
    '''

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Include both built-in TensorFlow Lite ops and selected TensorFlow ops
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.experimental_lower_tensor_list_ops = False  # Disable experimental lowering of tensor list ops

    converter.representative_dataset = representative_dataset

    # Convert the model to TFLite format
    tflite_quant_model = converter.convert()

    # save it down
    tflite_model_path = '/Users/jamborghini/Documents/PYTHON/Trained Models/' + model_name + '_quantized.tflite'
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_quant_model)

    return tflite_model_path

# only uncomment when quantizing model:
# tflite_model_path = post_training_quantize(model, representative_dataset)

def quantizer_inference(tflite_quant_model_path, input_all_channels):

    '''
    Function for running inference a quantized / compressed model
    '''

    start_time = time.time()   # set timer to measure inference speed

    # create the interpreter for inference (running the model on existing data)
    #interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)    # used when quantizing model for first time
    interpreter = tf.lite.Interpreter(model_path=tflite_quant_model_path)
    interpreter.allocate_tensors()

    # Get input and output details for tf.lite model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print(input_details)        # can see it takes in 3D
    # print(output_details)       # can see it spits out 2D

    # adjust input data to be compatible with new model format (for Conv1D)
    input_all_channels = input_all_channels.astype(np.float32)
    num_inference_segments, height, width = input_all_channels.shape        # 3 dimensions here, not 4 as for Conv2D
    estimated_bpm = []

    # run inference on each segment independently
    for i in range(num_inference_segments):
        input_data = input_all_channels[i:i+1]  # Select one segment
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        estimated_bpm.append(output_data[0][0])  # Assuming output is a scalar

    # finalise output into proper format
    estimated_bpm = np.array(estimated_bpm)
    estimated_bpm = estimated_bpm.reshape(-1, 1)

    elapsed_time = time.time() - start_time
    print(f"Inference time: {elapsed_time} seconds")
    return estimated_bpm

# only uncomment when running inference:
estimated_bpm = quantizer_inference(tflite_model_path, input_all_channels)

# equate length of test + label data
estimated_bpm = estimated_bpm[:len(actual_bpm)]
mae = mean_absolute_error(estimated_bpm, actual_bpm)
print(f'MAE:  {mae}')

plot_results(session_name, actual_bpm, estimated_bpm)




