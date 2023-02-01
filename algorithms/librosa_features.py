import librosa
import numpy as np
def audio_feature_extraction(df):
        data_array= []
        rate_array = []
        for audio in df['path']:
                data, rate = librosa.load(audio)
                data_array.append(data)
                rate_array.append(rate)

        df['data'] = data_array
        df['rate'] = rate_array


def chroma_feature(df):
    chroma_array = []
    for data, rate in zip(df['data'], df['rate']):
        chroma = librosa.feature.chroma_stft(y=data, sr=rate, n_chroma=7)
        chroma_array.append(chroma)
    df['chroma'] = chroma_array


def tonnetz_feature(df):
    tonnetz_array = []
    for data, rate in zip(df['data'], df['rate']):
        tonnetz = librosa.feature.tonnetz(y=data, sr=rate)
        tonnetz_array.append(tonnetz)
    df['tonnetz'] = tonnetz_array


def spectral_contrast(df):
    spectral_contrast_array = []
    for data, rate in zip(df['data'], df['rate']):
        spectral_contrast = librosa.feature.spectral_contrast(y=data, sr=rate)
        spectral_contrast_array.append(spectral_contrast)
    df['spectral_contrast'] = spectral_contrast_array  


def rmse_feature(df):
    # create an empty list to store the RMSE values
    rmse_list = []

    # iterate through the audio files in the dataset
    for data, rate in zip(df['data'], df['rate']):
        # calculate the root mean square energy
        rmse = librosa.feature.rms(y=data)
        # append the rmse mean to the rmse_list
        rmse_list.append(rmse[0])

    # add the rmse_list as a new column to the dataframe
    df['rmse'] = rmse_list


def spectral_flatness(df):
    # create an empty list to store the SF values
    sf_list = []

    # iterate through the audio files in the dataset
    for data, rate in zip(df['data'], df['rate']):
        # calculate the spectral flatness
        sf = librosa.feature.spectral_flatness(y=data)
        # append the SF mean to the sf_list
        sf_list.append(sf[0])

    # add the sf_list as a new column to the dataframe
    df['sf'] = sf_list


def sro_feature(df):
    # Create an empty list to store the spectral roll-off values
    spectral_rolloff_array = []

    for data, rate in zip(df['data'], df['rate']):
        spectral_rolloff = librosa.feature.spectral_rolloff(y=data, sr=rate)
        spectral_rolloff_array.append(spectral_rolloff[0])

    # Add the spectral roll-off values to the dataframe as a new column
    df['spectral_rolloff'] = spectral_rolloff_array


def zcr_feature(df):
    # Create arrays to store the zero-crossing rate values
    zero_crossing_rate_array = []

    for data in df['data']:
        
        # Compute the zero-crossing rate for the current audio file
        zero_crossing_rate = sum(librosa.zero_crossings(data))
        # Append the zero-crossing rate to the zero_crossing_rate_array
        zero_crossing_rate_array.append(zero_crossing_rate)

    # Add the zero-crossing rate arrays as new columns in the dataframe
    df['zero_crossing_rate'] = zero_crossing_rate_array



def mfcc_feature(df):
    # Create arrays to store the mfcc rate values
    mfcc_array = []

    for data, rate in zip(df['data'], df['rate']):
        
        # Compute the mfccs for the current audio file
        mfcc = librosa.feature.mfcc(y=data, sr=rate, n_mfcc=30)
        mfcc_array.append(mfcc)

    # Add the mfcc as a new column in the dataframe
    return mfcc_array