from email.mime import image
import tensorflow as tf
from tensorflow import keras
from keras.utils import img_to_array, load_img
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2  
import cvlib as cv
import streamlit as st
from streamlit_option_menu import option_menu
import os
from os import listdir
import numpy as np
from numpy import asarray,save
import time

classes = ['male','female']
age_classes = ['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-120']
age_classes_reversed = list(reversed(age_classes))

def main():
    st.set_page_config(
        page_title="AGE-GENDER DETECTOR",
        page_icon="calendar",
        layout="wide",
    )

    with st.sidebar:
        selected = option_menu("Main Menu", ["Home",'Upload Image','Camera Detection'], 
            icons=['house', 'person-circle','camera'], menu_icon="cast", default_index = 0)
        selected

    if selected == 'Home' :
        st.title(":blue[REAL TIME AGE & GENDER DETECTION APPLICATION] :camera:")
        content_html = """"""
        st.write(content_html)
        html_temp_home1 = """<div style="padding:10px">
                                <h2 style="color:green">With Using Upload Image & Using Web came also</h2>
                                <h3 style="color:violet;text-align:left;">
                                Age and Gender Detection application.</h3>
                                <h4>For use upload image : click on left side on "Upload Image"</h4>
                                <h4>For use webcam : click on left side on "Camera Detection"</h4>                        
                                </div>
                                </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)

    if selected == 'Upload Image' :
        # load model
        gender_model = load_model('models\Gender_Detection.h5')
        age_model = load_model('models\Age_Detection.h5')

        st.title(':blue[UPLOAD YOUR PICTURE TO PREDICT] :frame_with_picture:')
        upload_file = st.file_uploader("Choose an image to predict", type=["jpg","jpeg","png"], label_visibility="collapsed")
        if not (upload_file is None) :
            st.write('**Size of file is :**',upload_file.size)
            img = image.load_img(upload_file, target_size = (350,350))
            st.image(img, channels="RGB")
            img_process = image.load_img(upload_file, target_size = (96,96))
            img_process=img_to_array(img_process)
            img_process=img_process.astype('float32')
            img_process=img_process/255
            img_process=np.expand_dims(img_process,axis=0)

            button_click = st.button("Age and Gender Predict")
            if button_click :
                progress_text = "Operation in progress. Please wait."
                my_bar = st.progress(0, text=progress_text)
                for percent_complete in range(100):
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                gender = (gender_model.predict(img_process).argmax())
                age = (age_model.predict(img_process).argmax())

                gender_result = gender_model.predict(img_process).max()
                age_result = age_model.predict(img_process).max()

                st.subheader("Your Gender is : {} ".format(classes[gender], gender_result*100))
                st.subheader("Your Age Range is : {}".format(age_classes_reversed[age], age_result*100))

    if selected == "Camera Detection" :
        # load model
        gender_model = load_model('models\Gender_Detection.h5')
        age_model = load_model('models\Age_Detection.h5')
        st.title(':green[REAL TIME AGE AND GENDER DETECTION] :')
        st.subheader("Click on Start to use webcam and detect your age and gender real time :heart_decoration:")
        Camera_button = st.button("Start Camera")
        if Camera_button :
            # Camera frame on streamlit 
            frame_window = st.image([])
            webcam = cv2.VideoCapture(0)
            Stop_camera = st.button("Stop Camera")
            while webcam.isOpened():
                # read frame from webcam 
                status, frame = webcam.read()

                # apply face detection
                face, confidence = cv.detect_face(frame)

                # loop through detected faces
                for idx, f in enumerate(face):

                    # get corner points of face rectangle        
                    (startX, startY) = f[0], f[1]
                    (endX, endY) = f[2], f[3]

                    # draw rectangle over face
                    cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

                    # crop the detected face region
                    face_crop = np.copy(frame[startY:endY,startX:endX])

                    if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                        continue

                    # preprocessing for gender detection model
                    face_crop = cv2.resize(face_crop, (96,96))
                    face_crop = face_crop.astype("float") / 255.0
                    face_crop = img_to_array(face_crop)
                    face_crop = np.expand_dims(face_crop, axis=0)

                    # apply gender detection on face
                    conf = gender_model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
                    age_conf = age_model.predict(face_crop)[0]

                    # get label with max accuracy
                    idx = np.argmax(conf)
                    age_idx = np.argmax(age_conf)

                    label = classes[idx]
                    age_label = list(reversed(age_classes))[age_idx]

                    label = "{}: {:.2f}%".format(label, conf[idx] * 100)
                    age_label = "{}: {:.2f}%".format(age_label, age_conf[age_idx] * 100)

                    Y = startY - 10 if startY - 10 > 10 else startY + 10

                    # write label and confidence above face rectangle
                    cv2.putText(frame, label, (startX, Y-16),  cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)

                    cv2.putText(frame, age_label, (startX, Y+5),  cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)

                    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    frame_window.image(imgRGB)

                if Stop_camera & 0xFF == ord('q'):
                    break
                
if __name__ == "__main__":
    main()
    
