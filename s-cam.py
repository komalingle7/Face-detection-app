import streamlit as st
import streamlit.components.v1 as components
import cv2
import logging as log
import datetime as dt
from time import sleep

# Initialize Haar cascade
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename="webcam.log", level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

# Streamlit App

# Custom CSS
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
        font-family: Arial, sans-serif;
    }
    .title {
        text-align: center;
        color: #4CAF50;
        font-size: 32px;
        margin-bottom: 20px;
    }
    .header {
        color: #333;
        font-size: 24px;
    }
    .subheader {
        color: #555;
        font-size: 18px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #eaf7f7;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-size: 16px;
    }
    .sidebar-content {
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 8px;
        font-size: 16px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Title
st.markdown(
    "<div class='title'>Face Detection App using OpenCV and Streamlit</div>",
    unsafe_allow_html=True,
)

# Description
st.markdown(
    """
    <div class='info-box'>
        This app uses Haar Cascade Classifiers from OpenCV to detect faces in real-time. Streamlit enhances interactivity by providing user-friendly UI elements.
    </div>
""",
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.markdown(
    "<div class='sidebar-content'><b>User Details</b></div>", unsafe_allow_html=True
)
t1 = st.sidebar.text_input("Name of the Person 1", placeholder="Enter your name")
s1 = st.sidebar.slider("Age of the Person 1", min_value=0, max_value=100, value=25)

st.sidebar.markdown("---")
t2 = st.sidebar.text_input("Name of the Person 2", placeholder="Enter your name")
s2 = st.sidebar.slider("Age of the Person 2", min_value=0, max_value=100, value=30)

st.write(f"Name: {t1}, Age: {s1}")
st.write(f"Name: {t2}, Age: {s2}")

# Face Detection Explanation
st.markdown("<div class='header'>Face Detection Steps</div>", unsafe_allow_html=True)
steps = [
    "Load the Haar Cascade XML file.",
    "Convert video frames to grayscale for better performance.",
    "Use the detectMultiScale method to find faces in the frame.",
    "Draw rectangles around detected faces.",
    "Quit the app by pressing 'q'.",
]
for i, step in enumerate(steps, 1):
    st.markdown(f"**Step {i}:** {step}")

# Face Detection in Action
st.markdown("<div class='subheader'>Start Face Detection</div>", unsafe_allow_html=True)

if st.button("Detect Faces"):
    st.text("Initializing camera...")
    while True:
        if not video_capture.isOpened():
            st.error("Unable to load camera. Please check your webcam.")
            sleep(5)
            break

        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # Draw rectangles around faces
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if anterior != len(faces):
            anterior = len(faces)
            log.info("faces: " + str(len(faces)) + " at " + str(dt.datetime.now()))

        # Display the video frame
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
