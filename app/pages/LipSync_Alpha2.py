import streamlit as st
import os
import imageio
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

st.set_page_config(page_title="LipSync alpha2",layout='wide')


st.title('LipSync Alpha2')
options = os.listdir(os.path.join('../..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

col1, col2 = st.columns(2)

if options:

    with col1:
        file_path = os.path.join('../..', 'data', 's1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 ./pages/alpha2/test_video.mp4 -y')

        video = open('./pages/alpha2/test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)


    with col2:

        st.info('lip area')
        video = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('./pages/alpha2/animation.gif', video, fps=10)
        st.image('./pages/alpha2/animation.gif', width=400)


        st.info('Tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        st.info('Decoded into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)

