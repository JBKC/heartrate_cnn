
'''
File for running real-time inference of continuous sensor readout, on a pre-trained model
'''

import serial
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib, matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy import signal as sg

# initialise
window_time = 8000 # ms
segment_data = {}  # Dictionary to store data for each segment
model_name = 'HR_model_21_a'
model = load_model('/Users/jamborghini/Documents/PYTHON/Trained Models/' + model_name + '.h5')

# collect realtime data
ser = serial.Serial('/dev/tty.usbmodem14101', baudrate=115200)
ppg_data = []
x_data =[]
y_data = []
z_data = []

def process_segment(segment_data, indices_ppg, indices_acc):

    '''
    Main function for processing data once extracted
    '''

    fs_ppg_buffered = len(segment_data['PPG']['ppg_data']) / (window_time*2 /1000)
    fs_acc_buffered = len(segment_data['ACCEL']['x_data']) / (window_time*2 /1000)

    def band_pass_filter(data, fs):

        '''
        Filter the signal as this is similar to what the CNN has been trained on
        '''

        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype='band')
            return b, a

        def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
            b, a = butter_bandpass(lowcut, highcut, fs, order=order)
            y = lfilter(b, a, data)
            return y

        # PLAY AROUND WITH THIS to find optimum (essentially to match PPG Dalia)
        lowcut = 0.5  # Lower cutoff frequency (Hz)
        highcut = 20  # Upper cutoff frequency (Hz)
        # Apply the bandpass filter
        result = butter_bandpass_filter(data, lowcut, highcut, fs)

        return result

    ppg_data = band_pass_filter(segment_data['PPG']['ppg_data'], fs_ppg_buffered)
    x_data = band_pass_filter(segment_data['ACCEL']['x_data'], fs_acc_buffered)
    y_data = band_pass_filter(segment_data['ACCEL']['y_data'], fs_acc_buffered)
    z_data = band_pass_filter(segment_data['ACCEL']['z_data'], fs_acc_buffered)

    # remove the first half (first 8 seconds) to get the cleaned signal with no edge effects
    buffer_ppg = len([x for x in segment_data['PPG']['ppg_data'] if x <= window_time])
    buffer_acc = len([x for x in segment_data['ACCEL']['x_data'] if x <= window_time])
    ppg_data = ppg_data[indices_ppg[3]:]
    x_data = x_data[indices_acc[3]:]
    y_data = y_data[indices_acc[3]:]
    z_data = z_data[indices_acc[3]:]

    # plt.plot(ppg_data, 'black')
    # plt.show()

    def process_for_cnn(ppg_data, x_data, y_data, z_data):

        '''
        Standard process for CNN - finding FFT etc.
        '''

        def calculate_fft(data):
            result = np.fft.fft(data)  # take full FFT not just the positive part
            return result

        def trim_fft(data):
            # only keep 0-4Hz
            trim_index = 256 + 1
            result = data[:trim_index]
            return result

        # get the original signal into segments + take FFT
        def segment_fft(data):

            # zero padding to original signal increased frequency resolution before FFT, based on wanting 0-4Hz
            desired_freq = 4
            zeros_to_add = int(256*len(data) /(8*desired_freq) - len(data))   # 8 is scaling factor if we want 256 points
            # print(f'Zeros to add:  {zeros_to_add}')     # check it matches Excel calculations

            num_zeros = np.zeros(zeros_to_add)
            padded = np.concatenate((data, num_zeros))
            segment_fft = calculate_fft(padded)

            # cut down to 0-4Hz
            segment_fft = trim_fft(segment_fft)
            #print(f'FFT shape:  {segment_fft.shape}')

            # z-normalization individually on each new trimmed segment (makes sense)
            mean = np.mean(segment_fft)
            std_dev = np.std(segment_fft)
            segment_normalised = (segment_fft - mean) / std_dev

            result = np.array(segment_normalised)
            # plt.plot(result)
            # plt.show()

            #print(f'Normalised shape:  {result.shape}')

            return result

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
    print(f'All channels shape:  {result.shape}')

    return result

########### Running code section

while True:   # to make code run indefinitely
    data = ser.readline().decode().strip()  # Modify based on your data reading format
    # print(data)
    parts = data.split(',')

    if parts[0] == 'TIME':
        timestamp = int(parts[1])
        print(timestamp)

    elif parts[0] == 'ACCEL':
        # Extract accelerometer data
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
        segment_num = int(parts[4])

        # initialise dictionary if this is the first in the segment
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

        if segment_num not in segment_data:
            segment_data[segment_num] = {'ACCEL': {'x_data': [], 'y_data': [], 'z_data': []},
                                         'PPG': {'ppg_data': []}}

        segment_data[segment_num]['PPG']['ppg_data'].append(ppg)

    elif parts[0] == 'SEGMENT':
        # find the end of the segment label as coded in Arduino
        segment_num = int(parts[1])

        # super_segment for combining 2 second sub-segments
        super_segment = {
            'ACCEL': {'x_data': [],'y_data': [],'z_data': []},
            'PPG': {'ppg_data': []}}

        indices_acc = []
        indices_ppg = []

        if segment_num < 8:  # first 8 segments (0-7) totalling 16 seconds
            # keep appending onto the end
            pass

        else:
            for i in range(1,9): # taking previous 8 segments whenever a new segment is complete (rolling 8 second window, shifted by 2 seconds)
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
