from os import listdir
import multiprocessing
import warnings
import numpy as np
from scipy import stats
import pandas as pd
import librosa
import time
import sys

#import utils


def columns():
    feature_sizes = dict(chroma_stft=12, chroma_cqt=12, chroma_cens=12,
                         tonnetz=6, mfcc=20, rmse=1, zcr=1,
                         spectral_centroid=1, spectral_bandwidth=1,
                         spectral_contrast=7, spectral_rolloff=1)
    moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            it = ((name, moment, '{:02d}'.format(i+1)) for i in range(size))
            columns.extend(it)

    names = ('feature', 'statistics', 'number')
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    # More efficient to slice if indexes are sorted.
    return columns.sort_values()


def compute_features(file_name):
    print('going to compute feature for "{}" track'.format(file_name))

    features = pd.Series(index=columns(), dtype=np.float32, name=file_name)

    # Catch warnings as exceptions (audioread leaks file descriptors).
    warnings.filterwarnings('error', module='librosa')

    def feature_stats(name, values):
        features[name, 'mean'] = np.mean(values, axis=1)
        features[name, 'std'] = np.std(values, axis=1)
        features[name, 'skew'] = stats.skew(values, axis=1)
        features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
        features[name, 'median'] = np.median(values, axis=1)
        features[name, 'min'] = np.min(values, axis=1)
        features[name, 'max'] = np.max(values, axis=1)


    try:
        #filepath = '../audio_samples/group_1/' + file_name
        filepath = file_name
        #filepath = utils.get_audio_path(os.environ.get('AUDIO_DIR'), tid)
        x, sr = librosa.load(filepath, sr=None, mono=True)  # kaiser_fast

        f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
        feature_stats('zcr', f)

        cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                 n_bins=7*12, tuning=None))
        assert cqt.shape[0] == 7 * 12
        assert np.ceil(len(x)/512) <= cqt.shape[1] <= np.ceil(len(x)/512)+1

        f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cqt', f)
        f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cens', f)
        f = librosa.feature.tonnetz(chroma=f)
        feature_stats('tonnetz', f)

        del cqt
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        assert stft.shape[0] == 1 + 2048 // 2
        assert np.ceil(len(x)/512) <= stft.shape[1] <= np.ceil(len(x)/512)+1
        del x

        f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
        feature_stats('chroma_stft', f)

        f = librosa.feature.rmse(S=stft)
        feature_stats('rmse', f)

        f = librosa.feature.spectral_centroid(S=stft)
        feature_stats('spectral_centroid', f)
        f = librosa.feature.spectral_bandwidth(S=stft)
        feature_stats('spectral_bandwidth', f)
        f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
        feature_stats('spectral_contrast', f)
        f = librosa.feature.spectral_rolloff(S=stft)
        feature_stats('spectral_rolloff', f)

        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        del stft
        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        feature_stats('mfcc', f)
    except:
        print('Error occurred while extract features from {}'.format(file_name))
        print("Unexpected error:", sys.exc_info()[0])

    print('feature for "{}" track is completed'.format(file_name))
    
    return features


def save(df, name, n_digits):
    df.to_csv('../csv/{}_features.csv'.format(name), float_format='%.{}e'.format(n_digits))


def extract_features(folder_path, label):
    file_names = listdir(folder_path)
    full_columns = columns()

    df = pd.DataFrame(index=file_names, columns=full_columns)
    # single thread approach
    # for file in file_names:
    #     path_to_file = folder_path + '/' + file
    #     print('goint to compute feature for "{}" track'.format(path_to_file))
    #     features = compute_features(path_to_file)
    #     df.loc[file] = features

    # multithread approach
    file_pathes = list(map(lambda x: folder_path + '/' + x, file_names))

    nb_workers = 4
    pool = multiprocessing.Pool(nb_workers)
    it = pool.map(compute_features, file_pathes)

    for i, row in enumerate(it):
        file_name = file_names[i]
        df.loc[file_name] = row

    df.columns = [' '.join(col).strip() for col in df.columns.values]
    df['label'] = label

    return df

def main():
    start = time.time()

    group_1 = extract_features('../audio_samples/group_1', 1)
    #group_2 = extract_features('../audio_samples/group_2', 2)

    save(group_1, 'group_1', 10)
    #save(group_2, 'group_2', 10)

    end = time.time()
    print(end - start)


if __name__ == "__main__":
    main()