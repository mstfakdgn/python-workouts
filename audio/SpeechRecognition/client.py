import requests

# URL = "http://127.0.0.1:5050/predict"
URL = "http://127.0.0.1/predict"

TEST_AUDIO_FILE_TEST = "test/left.wav"

if __name__ == "__main__":

    audio_file = open(TEST_AUDIO_FILE_TEST, "rb")
    values = {
        "file" : (TEST_AUDIO_FILE_TEST, audio_file, "audio/wav")
    }

    response = requests.post(URL, files=values, stream=True)
    print(response)
    # data = response.json()

    # print(f"Predicted Keyword is: {data['key']}")