import os
import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import seaborn as sns
from pydub import AudioSegment
import speech_recognition as sr
from io import BytesIO
import tempfile
from moviepy.editor import VideoFileClip

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

# Function to load model and vectorizer
def load_model_and_vectorizer(model_file, vectorizer_file):
    loaded_model = joblib.load(model_file)
    loaded_vectorizer = joblib.load(vectorizer_file)
    return loaded_model, loaded_vectorizer

# Function to classify text and get prediction accuracy
def classify_text(text, model, vectorizer):
    preprocessed_text = preprocess_text(text)
    input_transformed = vectorizer.transform([preprocessed_text])
    prediction = model.predict(input_transformed)
    prediction_prob = model.predict_proba(input_transformed)
    class_labels = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
    predicted_class = class_labels[prediction[0]]
    predicted_probability = prediction_prob[0][prediction[0]]
    return predicted_class, predicted_probability

def convert_to_wav(audio_file, temp_audio_file):
    try:
        audio = AudioSegment.from_file(audio_file)
        audio.export(temp_audio_file, format="wav")
        return temp_audio_file
    except Exception as e:
        print(f"Error converting audio to WAV: {e}")
        return None

def audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    audio = sr.AudioFile(audio_file)

    with audio as source:
        audio_data = recognizer.record(source)

    text = recognizer.recognize_google(audio_data)
    return text

def process_audio_file(audio_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
            temp_audio_file_path = convert_to_wav(audio_file, temp_audio_file.name)
            if temp_audio_file_path is None:
                return "Error converting audio to WAV.", None

        text = audio_to_text(temp_audio_file_path)
        if text is None:
            return "Error converting audio to text.", None

        prediction = predict_hate_speech(text)

        if os.path.exists(temp_audio_file_path):
            os.remove(temp_audio_file_path)

        if prediction == 1:
            return "Hate speech detected.", text
        else:
            return "Non-hate speech.", text

    except Exception as e:
        return f"Error processing file: {e}", None

def convert_video_to_audio(video_file):
    try:
        # Create a temporary file for the audio output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio_file:
            temp_audio_file_path = temp_audio_file.name

        # Use moviepy to extract audio from the video file
        video_clip = VideoFileClip(video_file)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(temp_audio_file_path)
        audio_clip.close()
        video_clip.close()

        return temp_audio_file_path  # Return the path to the temporary audio file

    except Exception as e:
        print(f"Error converting {video_file} to audio: {e}")
        return None

def process_video_file(video_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
            temp_video_file.write(video_file.read())
            temp_video_file_path = temp_video_file.name

        temp_audio_file_path = convert_video_to_audio(temp_video_file_path)
        if temp_audio_file_path is None:
            return "Error converting video to audio.", None

        text = audio_to_text(temp_audio_file_path)
        if text is None:
            return "Error converting audio to text.", None

        prediction = predict_hate_speech(text)

        os.remove(temp_audio_file_path)
        os.remove(temp_video_file_path)

        if prediction == 1:
            return "Hate speech detected.", text
        else:
            return "Non-hate speech.", text

    except Exception as e:
        return f"Error processing file: {e}", None

tfidf_vectorizer = joblib.load('tfidf1_vectorizer.pkl')
svm_model = joblib.load('svm1_classifier.joblib')

def predict_hate_speech(text):
    text_vector = tfidf_vectorizer.transform([text])
    prediction = svm_model.predict(text_vector)
    return prediction

import streamlit as st
from PIL import Image

# Streamlit app function
def main(model_accuracy):
    st.set_page_config(page_title="Hate Speech Detector", page_icon="🗣️", layout="centered", initial_sidebar_state="expanded")

    # CSS to inject contained in a string
    page_bg_img = '''
    <style>
    body {
        background-color: #DFDCD3;
    }
    .stApp {
        background: rgba(223, 220, 211, 0.8);
        border-radius: 10px;
        padding: 10px;
    }
    .sidebar .sidebar-content {
        color: #000000;
    }
    .sidebar .sidebar-content .element-container {
        font-size: 18px;
    }
    .sidebar .sidebar-content .element-container .element {
        border-radius: 8px;
        margin: 8px 0;
        background-color: #f0f0f0;
    }
    .sidebar .sidebar-content .element-container .element:hover {
        background-color: #e0e0e0;
    }
    .sidebar .sidebar-content .element-container .element input[type="radio"]:checked + div {
        background-color: #c0c0c0;
    }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.sidebar.title("🌐 **ToxiMonitor**")
    st.sidebar.markdown("<hr style='border: 1px solid #000;'>", unsafe_allow_html=True)

    page = st.sidebar.radio("Choose a page:", ["About", "Text Detection", "Audio Detection", "Video Detection"])

    st.sidebar.markdown("<hr style='border: 1px solid #000;'>", unsafe_allow_html=True)


    if page == "About":
        image_path = r"D:\jn\df.jpg"
        if os.path.exists(image_path):
            st.image(image_path, use_column_width=True)
        else:
            st.warning(f"Image file not found: {image_path}")
        
        st.markdown("<h1 style='text-align: center; font-size: 24px;'>About Us</h1>", unsafe_allow_html=True)

        st.markdown("<div style='font-size: 20px;'>Welcome to the Hate Speech Detector application! This application recognizes hate speech and offensive language in text, audio, and video files. The purpose of this application is to foster a positive online community.</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 20px;'>Our mission is to provide a safe and respectful online environment for everyone.</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 20px;'>This application uses machine learning algorithms to detect hate speech and offensive language in user input.</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 20px;'>We hope you find this application useful and informative.</div>", unsafe_allow_html=True)
        
    elif page == "Text Detection":    
        st.markdown("<h1 style='text-align: center; font-size: 24px;'>Text Detection</h1>", unsafe_allow_html=True)
        
        input_text = st.text_input("Enter text to classify:", placeholder="Type or paste your text here...", key="text_input", max_chars=1000)
        
        if st.button("Classify Text"):
            # Clear previous results
            st.text("")
            
            # Classify text
            if input_text:
                prediction = predict_hate_speech(input_text)
                if prediction == 1:
                    prediction = "Hate Speech Detected 🚨"
                    box_color = "background-color: #ffcccc;"
                else:
                    prediction = "Non-Hate Speech 👍"
                    box_color = "background-color: #ccffcc;"
                
                st.markdown(f"<div style='font-size: 20px;'><b>Prediction for Text Input:</b> {prediction}</div>", unsafe_allow_html=True)
                st.markdown(f"<textarea style='width: 100%; height: 200px; font-size: 20px; {box_color}' readonly>{input_text}</textarea>", unsafe_allow_html=True)

    elif page == "Audio Detection":
        st.markdown("<h1 style='text-align: center; font-size: 24px;'>Audio Detection</h1>", unsafe_allow_html=True)
        
        audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])
        if audio_file is not None:
            st.audio(audio_file, format=audio_file.type)
        
        if st.button("Classify Audio"):
            # Clear previous results
            st.text("")
            
            # Classify audio
            if audio_file:
                result, extracted_text = process_audio_file(audio_file)
                if "Hate speech" in result:
                    box_color = "background-color: #ffcccc;"
                else:
                    box_color = "background-color: #ccffcc;"

                st.markdown(f"<div style='font-size: 20px;'><b>Prediction for Audio File:</b> {result}</div>", unsafe_allow_html=True)
                if extracted_text:
                    st.markdown(f"<textarea style='width: 100%; height: 200px; font-size: 20px; {box_color}' readonly>{extracted_text}</textarea>", unsafe_allow_html=True)

    elif page == "Video Detection":   
        st.markdown("<h1 style='text-align: center; font-size: 24px;'>Video Detection</h1>", unsafe_allow_html=True)
        
        video_file = st.file_uploader("Upload a video file", type=["mp4", "wav", "mov", "avi", "mkv"])
        
        if st.button("Classify Video"):
            # Clear previous results
            st.text("")
            
            # Classify video
            if video_file:
                result, extracted_text = process_video_file(video_file)
                if "Hate speech" in result:
                    box_color = "background-color: #ffcccc;"
                else:
                    box_color = "background-color: #ccffcc;"

                st.markdown(f"<div style='font-size: 20px;'><b>Prediction for Video File:</b> {result}</div>", unsafe_allow_html=True)
                if extracted_text:
                    st.markdown(f"<textarea style='width: 100%; height: 200px; font-size: 20px; {box_color}' readonly>{extracted_text}</textarea>", unsafe_allow_html=True)

if __name__ == "__main__":
    model_accuracy = 85.5  # Example accuracy value, replace with your actual accuracy
    main(model_accuracy)
