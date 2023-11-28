import streamlit as st

def main():
    st.title("LipSync - Lip Movement Detection and Speech Prediction")
    st.image('animation.gif', width=400)
    st.info("Research paper :- https://arxiv.org/pdf/1611.01599.pdf")

    st.header("Abstract")
    st.write("LipSync is a machine learning project that focuses on detecting lip movements from video recordings and predicting the corresponding speech. By leveraging computer vision techniques, LipSync aims to create a system that can understand spoken content by analyzing the visual cues provided by lip movements.")

    st.header("Introduction")
    st.write("LipSync delves into the realm of speech perception using visual information from lip movements. Lips play a vital role in speech production, and capturing and interpreting their movements can significantly contribute to understanding spoken language. This project aims to leverage machine learning to detect lip movements from video data and predict what the person is saying, bridging the connection between visual cues and speech.")

    st.header("Methodology")
    st.subheader("Dataset Collection")
    st.write("Compile a dataset of video recordings featuring individuals speaking, accompanied by transcriptions or labels indicating the spoken words. This dataset will serve as the training and evaluation foundation for the project.")

    st.subheader("Preprocessing")
    st.write("Extract individual frames or clips from video recordings. Preprocess the frames by resizing, normalizing, or converting them to a suitable format for subsequent analysis.")

    st.subheader("Lip Movement Detection")
    st.write("Develop a lip movement detection model using computer vision techniques. This model will identify and track the movements of lips in the video frames, potentially utilizing approaches such as facial landmark detection, optical flow, or deep learning-based methods like convolutional neural networks (CNNs) for effective feature extraction.")

    st.subheader("Speech Prediction")
    st.write("Train a machine learning model to predict speech from the detected lip movements. This model will leverage the visual information from lip movements to generate predictions about the spoken content.")

    st.subheader("Evaluation and Deployment")
    st.write("Assess the performance of the LipSync system using appropriate metrics and dedicated validation datasets. Fine-tune the model architecture and training strategies to enhance accuracy and robustness. Finally, deploy the LipSync system to a suitable environment for lip movement detection and speech prediction in real-time or near real-time scenarios.")

    st.header("Conclusion")
    st.write("LipSync presents an innovative approach to speech perception by analyzing lip movements. By leveraging computer vision techniques, LipSync aims to bridge the gap between visual cues from lip movements and speech understanding. Successful implementation of LipSync could have applications in various fields, including human-computer interaction, transcription services, and speech therapy.")

if __name__ == '__main__':
    main()
