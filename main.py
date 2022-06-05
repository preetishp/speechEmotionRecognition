import numpy as np
from flask import Flask, render_template, request
#import cv2
#from gtts import gTTS
import os
import librosa
import tensorflow as tf
#import pytesseract
#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
app = Flask(__name__,template_folder='templates')


@app.route("/")
def upload():
    return render_template('upload.html')

data, sample_rate = librosa.load("data/a05.wav")

@app.route("/", methods=["GET", "POST"])
def success():
    if request.method == 'POST':
        #img = request.files['img'].read()
        #Image = numpy.fromstring(img, numpy.uint8)
        #images= cv2.imdecode(Image, cv2.IMREAD_COLOR)
        #text= pytesseract.image_to_string(images)
        #result = gTTS(text=text, lang='en', slow=False)
        #result.save("result.mp3")
        #os.system("result.mp3")

        speech = request.files['img'].read()

        X= []
        feature = get_features("data/a05.wav")
        for ele in feature:
            X.append(ele)
        #print(X)
        X = np.expand_dims(X,axis=2)
        saved_model = tf.keras.models.load_model('speech_model.h5')
        predictions = saved_model.predict(X)
        predictions = np.argmax(predictions, axis=1)
        print("the predictions are : ")
        print(predictions)
        most_frequent_predictions = np.bincount(predictions).argmax()

    return render_template('success.html',text=str(most_frequent_predictions))




def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    # without augmentation
    res1 = extract_features(data)
    result = np.array(res1)
    
    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2)) # stacking vertically
    
    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3)) # stacking vertically
    
    return result


if __name__ == "__main__":
    app.run(debug=True)