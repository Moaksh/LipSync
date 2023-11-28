import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.processing = False

    def transform(self, frame):
        if not self.processing:
            self.processing = True
            # Perform any desired image/video processing here
            # For example, you can apply filters, object detection, etc.
            processed_frame = frame
            self.processing = False
        else:
            processed_frame = frame

        return processed_frame


def main():
    st.title("Live Video Feed")

    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer, rtc_configuration=rtc_configuration)

if __name__ == '__main__':
    main()
