# Import all important libraries
import streamlit as st
import os
import imageio
from PIL import Image
import tempfile
import cv2

import tensorflow as tf
from utils import load_data, num_to_char,load_video
from modelutil import load_model

# Set the layout to the strealit app as wide
st.set_page_config(page_title = 'Lip Reading App', layout = 'wide')

# Setup the sidebar
with st.sidebar:
    st.image('https://builtin.com/sites/www.builtin.com/files/styles/og/public/2022-07/future-artificial-intelligence.png')
    st.title('Lip Reading Model')
    st.info('This application is developed from the LipNet deep learning model')

    st.title('Lip reading model information')
    st.info('The model only looks at the lips of the person, it doesnt use any audio!')

    st.title('Vocabulair')
    st.info('The model is for now only trained on one person and his vocabulair')

    st.title('Which model has been used for the predictions?')
    st.info('I have used a neural network model to make the predictions of what the person is saying')

st.title('Lip reading model application')


# Upload video
uploaded_video = st.file_uploader('Upload a video', type=['mp4'])

if uploaded_video is not None:
    # Read the video file
    video_bytes = uploaded_video.read()

    # Save the video to a temporary file
    temp_video_file = tempfile.NamedTemporaryFile(delete=False)
    temp_video_file.write(video_bytes)
    temp_video_file.close()

    # Load the video using OpenCV
    video_path = temp_video_file.name
    video_cap = cv2.VideoCapture(video_path)

    # Generate two columns
    col1, col2 = st.columns(2)

    # Display the original video
    with col1:
        st.info('The video below displays the uploaded video')
        st.video(video_bytes)

    with col2:
        st.info('This is what the model sees when making a prediction')
        try:
            video = load_video(video_path)
            imageio.mimsave('animation.gif', video, fps=10)
            st.image('animation.gif', width=400)

            st.info('This is the output of the machine learning model as tokens')
            model = load_model()
            yhat = model.predict(tf.expand_dims(video, axis=0))
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            st.text(decoder)

            # Convert prediction to text
            st.info('Decode the raw tokens into words')
            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            st.text(converted_prediction)

        except FileNotFoundError:
            st.error('Unable to load the video. Please make sure to upload a valid video file.')



