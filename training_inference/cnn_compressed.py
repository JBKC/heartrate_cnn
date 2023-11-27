
'''
HeartRate estimator taking in pre-recorded PPG and accelerometer sensor readings
Outputs CNN model that takes up < 40KB
Two steps: (1) train model; (2) compress model
'''

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import wfdb
import time

'''
Start of preprocessing section
'''

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

    with open(f'{session_name}.pkl', 'rb') as file:                    # adjust file path as necessary
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

def CNN_model():
    '''
    Architecture of CNN
    '''

    input_shape = (257,4)
    model = models.Sequential()

    # Initial Convolutional Layer
    model.add(layers.Conv1D(filters=8, kernel_size=(1), strides=(1), activation='relu', input_shape=input_shape))
    # Max-Pooling Layer
    model.add(layers.MaxPooling1D(pool_size=(2), strides=(2)))

    # Repeating Pattern
    NL = 3 
    for i in range(1, NL+1):
        
        n_filters = 2 ** (i + 3)
        model.add(layers.Conv1D(filters=n_filters, kernel_size=(3), strides=(1), activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=(2), strides=(2)))

    #Final Convolutional Layer
    model.add(layers.Conv1D(filters=16, kernel_size=(1), strides=(1), activation='relu'))

    # Flattening layer
    model.add(layers.Flatten())

    # Fully Connected Layer 1
    n1f_c = 64  # number of neurons
    model.add(layers.Dense(n1f_c, activation='relu'))

    # Fully Connected Layer 2
    n2f_c = 1
    model.add(layers.Dense(n2f_c, activation='linear'))

    # Compile the model
    learning_rate = 0.001
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    model.summary()

    return model

'''
Start of model training section
'''

# Model initialisation
num_epochs = 10 
batch_size = 32  
losses = []
mae_scores = []

session_data = {}
label_session_data = {}
train_data = []
train_labels = []
test_data = []
test_labels = []
session_keys = []

# Process each file and save for model input
file_names = ['S{}'.format(i) for i in range(1, 16) if i != 6]        # skip Session 6 as data cut short
pkl_files = [file_name for file_name in file_names]

# Assign session keys
for i, session_file in enumerate(pkl_files, start=1):
    session_key = i
    session_keys.append(session_key)
    print(f'Session number:  {session_key}')
    input_all_channels, label_ecg_data = process_for_cnn(*preprocess_data_dalia(session_file))
    session_data[session_key] = input_all_channels    # Assign session data
    label_session_data[session_key] = label_ecg_data  # Assign label data

def train_and_evaluate_model(train_data, train_labels, test_data, test_labels):
    '''
    Compile the CNN model and evaluate against test data
    '''

    # shuffle the order of the input segments
    indices = np.random.permutation(len(train_data))
    train_data = train_data[indices]
    train_labels = train_labels[indices]

    model = CNN_model()

    # Train the model with training data
    history = model.fit(
        train_data,
        train_labels,
        epochs=num_epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(test_data, test_labels),
        verbose=1,
    )

    # Calculate MAE on the test set
    estimated_bpm = model.predict(test_data)
    mae = mean_absolute_error(test_labels, estimated_bpm)
    return mae, history, model, estimated_bpm

# Iterate through sessions and train the model
for session_key in session_keys:
    # Load data for the current session + assign it as the testing set
    print(f'Session number: {session_key}/{len(session_keys)}')
    test_data = session_data[session_key]
    test_labels = label_session_data[session_key]

    # Assign the remaining sessions as the training data
    train_sessions = [key for key in session_keys if key != session_key]
    for train_session_key in train_sessions:
        train_data.append(session_data[train_session_key])
        train_labels.append(label_session_data[train_session_key])

    # Concatenate all the training data
    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    # Pass training data through the model
    mae, history, model, estimated_bpm = train_and_evaluate_model(train_data, train_labels, test_data, test_labels)
    print(f'MAE for session:  {mae}')
    mae_scores.append(mae)

model_name = 'HR_model'  # Set desired model name
model.save()

print("Mean Absolute Errors for Each Session:")
for i, mae in enumerate(mae_scores):
    print(f"Session {session_keys[i]}: {mae:.2f}")     

'''
Start of model compression section
'''

def generate_sample_input():
    '''
    Representative dataset for quantisation
    '''
    
    input_data = np.float32(input_all_channels)
    print(input_data.shape)
    yield [input_data]

# quantization function
def post_training_quantize(model, representative_dataset):
    '''
    Function for quantizing the trained model
    '''

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.experimental_lower_tensor_list_ops = False

    converter.representative_dataset = representative_dataset

    # Convert the model to TFLite format
    result = converter.convert()

    return result

representative_dataset = tf.lite.RepresentativeDataset(generate_sample_input)
tflite_model = post_training_quantize(model, representative_dataset)

'''
Start of model inference section (comment out when training)
'''

def plot_results(session_name, actual_bpm, estimated_bpm):
    '''
    For visualising results against labelled data
    '''

    plt.plot(estimated_bpm, color='red', label='Estimated')
    plt.plot(actual_bpm, color='black', label='Actual')
    plt.ylabel("Heart Rate (BPM)")
    plt.legend()
    plt.title(f"{session_name} - Estimated vs. Actual BPM")
    plt.show()

# choose dataset to run inference on (comment out as necessary)

## ppg dalia (training data)
# session_name = 'S1'
# input_all_channels, actual_bpm = process_for_cnn(*preprocess_data_dalia(session_name))

## wrist ppg (unseen data)
session_name = 's1_high_resistance_bike'
input_all_channels, actual_bpm = process_for_cnn(*preprocess_data_wrist_ppg(session_name))

def quantizer_inference(tflite_quant_model_path, input_all_channels):
    '''
    Run inference on compressed model
    '''

    # timer to measure speed of inference
    start_time = time.perf_counter()

    # create the interpreter for inference
    interpreter = tf.lite.Interpreter(model_content=tflite_model)  
    interpreter.allocate_tensors()

    # get input and output details for tf.lite model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # adjust input data to be compatible with new model format (for Conv1D)
    input_all_channels = input_all_channels.astype(np.float32)
    num_inference_segments, height, width = input_all_channels.shape     
    estimated_bpm = []

    # run inference on each segment independently
    for i in range(num_inference_segments):
        input_data = input_all_channels[i:i+1]
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        estimated_bpm.append(output_data[0][0])

    # finalise output into proper format
    estimated_bpm = np.array(estimated_bpm)
    estimated_bpm = estimated_bpm.reshape(-1, 1)        # convert to 2D array

    end_time = time.perf_counter()
    print(f"Inference time: {end_time - start_time} seconds")
    return estimated_bpm

# Run inference function
estimated_bpm = quantizer_inference(tflite_model_path, input_all_channels)

mae = mean_absolute_error(estimated_bpm, actual_bpm)
print(f'MAE:  {mae}')

plot_results(session_name, actual_bpm, estimated_bpm)




