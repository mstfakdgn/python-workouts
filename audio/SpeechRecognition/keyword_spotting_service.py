import tensorflow.keras as keras
import numpy as np
import librosa

MODEL_PATH = 'model.h5'
NUM_SAMPLES_TO_CONSIDER = 22050 # 1 second

class _Keyword_Spotting_Service:

    model = None
    _mappings_ = [
        "stop",
        "yes",
        "left",
        "off",
        "go",
        "right",
        "up",
        "on",
        "down",
        "no"
    ]
    _instance = None

    def predict(self, file_path):

        # extract MFCCs
        MFCCs = self.preprocess(file_path) # (number of segments, number of coefficients)

        # convert 2d MFCCs array into 4d array -> (# samples, # segments, # coefficients, # channels)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make prediction
        predictions = self.model.predict(MFCCs) # [ [0.1,0.1, ...,0.2] ]
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mappings_[predicted_index]

        return predicted_keyword

    def preprocess(self, file_path, n_fft=2048, n_mfcc=13, hop_length=512):
        
        # laod audio file
        signal, sr = librosa.load(file_path)

        # ensure consistncy in audio file length
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]

        # extract MFCC
        MFCCs = librosa.feature.mfcc(signal, sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)

        return MFCCs.T

def Keyword_Spotting_Service():

    # ensure that we only have one instance of KSS
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)

    return _Keyword_Spotting_Service._instance

if __name__ == "__main__":
    kss = Keyword_Spotting_Service()

    keyword1 = kss.predict("test/down.wav")
    keyword2 = kss.predict("test/left.wav")

    print(f"Predicted: {keyword1}, {keyword2}")