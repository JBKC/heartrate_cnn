
'''
Run real-time HeartRate inference on a pre-trained model
Retrieve data via Arduino sketch file "Heartrate_accelerometer_realtime.ino"
'''

import serial
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib, matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy import signal as sg

# initialisation
window_time = 8 #seconds                  # total segment for calculating heartrate
segment_data = {}                         # dictionary to store data for each segment
ppg_data = []
x_data =[]
y_data = []
z_data = []
model = load_model('HR_model_21_a.h5')    # adjust pathname as necessary

# collect realtime data from Arduino
ser = serial.Serial('/dev/tty.usbmodem14101', baudrate=115200)        # change input port as necessary

def process_segment(segment_data, indices_ppg, indices_acc):
    '''
    Main function for processing data once extracted
    '''

    # Calculate sampling frequency for butterworth filter
    fs_ppg_buffered = len(segment_data['PPG']['ppg_data']) / (window_time*2)
    fs_acc_buffered = len(segment_data['ACCEL']['x_data']) / (window_time*2)

    def band_pass_filter(data, fs):
        '''
        Filter the input signal to reduce noise and more closely emulate data that model was trained on
        '''

        def butter_bandpass(lowcut, highcut, fs, order=5):
            '''
            Calculate butterworth coefficients
            '''
            
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype='band')
            return b, a

        def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
            '''
            Apply butterworth to data
            '''
            
            b, a = butter_bandpass(lowcut, highcut, fs, order=order)
            y = lfilter(b, a, data)
            return y

        lowcut = 0.5  # Lower cutoff frequency (Hz)
        highcut = 20  # Upper cutoff frequency (Hz)
        result = butter_bandpass_filter(data, lowcut, highcut, fs)

        return result

    # filter all signals in real time
    ppg_data = band_pass_filter(segment_data['PPG']['ppg_data'], fs_ppg_buffered)
    x_data = band_pass_filter(segment_data['ACCEL']['x_data'], fs_acc_buffered)
    y_data = band_pass_filter(segment_data['ACCEL']['y_data'], fs_acc_buffered)
    z_data = band_pass_filter(segment_data['ACCEL']['z_data'], fs_acc_buffered)

    # trim the first half (first 4 segments) to remove butterworth edge effects
    ppg_data = ppg_data[indices_ppg[3]:]
    x_data = x_data[indices_acc[3]:]
    y_data = y_data[indices_acc[3]:]
    z_data = z_data[indices_acc[3]:]

    def process_for_cnn(ppg_data, x_data, y_data, z_data):
        '''
        Processing of data to be passed into CNN
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

        def segment_fft(data):
            '''
            Cut time-domain signal into segments and run processing
            '''
            # Zero-pad signal to increase frequency resolution
            desired_freq = 4
            zeros_to_add = int(256*len(data) /(8*desired_freq) - len(data))   # 8 is scaling factor to get 256 points
            num_zeros = np.zeros(zeros_to_add)
            padded = np.concatenate((data, num_zeros))
            segment_fft = calculate_fft(padded)

            segment_fft = trim_fft(segment_fft)

            # Z-normalization
            mean = np.mean(segment_fft)
            std_dev = np.std(segment_fft)
            segment_normalised = (segment_fft - mean) / std_dev

            result = np.array(segment_normalised)

            return result
            
        # Retrieve inputs for model
        ppg_input = segment_fft(ppg_data)
        x_input = segment_fft(x_data)
        y_input = segment_fft(y_data)
        z_input = segment_fft(z_data)

        input_all_channels = np.stack([ppg_input, x_input, y_input, z_input], axis=0)
        input_all_channels = np.expand_dims(input_all_channels, axis=0)

        # no reshaping needed for Conv2D
        # input_all_channels = np.transpose(input_all_channels, (0, 2, 1))   # CONV1D option

        return input_all_channels

    result = process_for_cnn(ppg_data, x_data, y_data, z_data)

    return result

'''
Running real-time inference
'''

while True:
    # read data sent from Arduino
    data = ser.readline().decode().strip() 
    parts = data.split(',')

    # split readings based on text stamps
    if parts[0] == 'TIME':
        timestamp = int(parts[1])
        print(timestamp)

    elif parts[0] == 'ACCEL':
        # Extract accelerometer data
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
        segment_num = int(parts[4])

        # initialise dictionary at the start of each segment to store new data
        if segment_num not in segment_data:
            segment_data[segment_num] = {'ACCEL': {'x_data': [], 'y_data': [], 'z_data': []},
                                         'PPG': {'ppg_data': []}}
        # append data to this segment's dictionary
        segment_data[segment_num]['ACCEL']['x_data'].append(x)
        segment_data[segment_num]['ACCEL']['y_data'].append(y)
        segment_data[segment_num]['ACCEL']['z_data'].append(z)

    elif parts[0] == 'PPG':
        # Extract PPG data
        ppg = float(parts[1])
        segment_num = int(parts[2])

        # repeat segment setup steps
        if segment_num not in segment_data:
            segment_data[segment_num] = {'ACCEL': {'x_data': [], 'y_data': [], 'z_data': []},
                                         'PPG': {'ppg_data': []}}
        segment_data[segment_num]['PPG']['ppg_data'].append(ppg)

    elif parts[0] == 'SEGMENT':
        # find the end of the segment label
        segment_num = int(parts[1])

        # combine multiple 2 second sub-segements for running inference
        super_segment = {
            'ACCEL': {'x_data': [],'y_data': [],'z_data': []},
            'PPG': {'ppg_data': []}}

        indices_acc = []
        indices_ppg = []

        if segment_num < 8:  # append segments until first 8 segments (16 seconds) reached
            pass

        else:
            for i in range(1,9): # take previous 8 segments whenever a new segment is complete (rolling 8 second window, shifted by 2 seconds each time)
                segment = segment_data[segment_num - i]

                super_segment['ACCEL']['x_data'] += segment['ACCEL']['x_data']
                super_segment['ACCEL']['y_data'] += segment['ACCEL']['y_data']
                super_segment['ACCEL']['z_data'] += segment['ACCEL']['z_data']
                super_segment['PPG']['ppg_data'] += segment['PPG']['ppg_data']
                indices_acc.append(len(super_segment['ACCEL']['x_data']))
                indices_ppg.append(len(super_segment['PPG']['ppg_data']))

            input_all_channels = process_segment(super_segment, indices_ppg, indices_acc)

            # Model inference
            estimated_bpm = model.predict(input_all_channels)
            print(f'Estimated BPM:  {estimated_bpm}')
