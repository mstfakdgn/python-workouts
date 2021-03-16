import librosa
import os
import json

DATASET_PATH = 'data'
JSON_PATH = 'data.json'
SAMPLES_TO_CONSIDER = 22050 # 1 second worth of sound

def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):
    """Extracts MFCCs from music dataset and saves them into a json file.
    :param dataset_path (str): Path to dataset
    :param json_path (str): Path to json file used to save MFCCs
    :param num_mfcc (int): Number of coefficients to extract
    :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
    :param hop_length (int): Sliding window for FFT. Measured in # of samples
    :return:
    """

    # create dictionary
    data = {
        "mappings": [],
        "labels": [],
        "MFCCs": [],
        "files" : []
    }

    # loop throught all the sub-dirs
    for i, (dirpath,dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # we need to ensure that we'are not at the root level
        if dirpath is not dataset_path:

            # update mappings
            category = dirpath.split("/")[-1] # dataset/down -> [dataset, down]
            data["mappings"].append(category)
            print(f"Processing {category}")

            # loop through all the filenames and extract MFCCs
            for f in filenames:

                # get file path
                file_path = os.path.join(dirpath, f)

                # load audio file 
                signal, sr = librosa.load(file_path)

                # ensure the audio file is at least 1 second
                if len(signal) >=  SAMPLES_TO_CONSIDER:

                    # enforce 1 second long signal
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    # extract the MFCCs 
                    mfccs = librosa.feature.mfcc(signal, sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                                 hop_length=hop_length)

                    # store data
                    data["labels"].append(i-1)
                    data["MFCCs"].append(mfccs.T.tolist())
                    data["files"].append(file_path)
                    print(f"{file_path}: {i-1}")

    # store in json file

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH)