#region Libraries
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from keras.models import load_model
from keras.utils import img_to_array
import cv2
import numpy as np
import math
#endregion

#region Properties
show_cam = True     #open webcam for debug
text_max_size = 200
text_max_alph = 1
#endregion

#region Functions
def lerp(a: float, b: float, t: float) -> float:
    return (1 - t) * a + t * b
#endregion

#region Main
def main():

    # region Variables
    #--------------------V A R I A B L E S--------------------
    text_size = 0
    text_alph = 0

    emotion_value = 0
    angry_value_red = 0
    face_distance = 0

    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    classifier = load_model('EmotionDetectionModel.h5')

    class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

    cap = cv2.VideoCapture(0)

    WIDTH = 1920
    HEIGHT = 1080

    CAM_WIDTH = 0
    CAM_HEIGHT = 0

    _value0 = 0
    _value1 = 0
    _value2 = 0
    _value3 = 0
    _value4 = 0

    angry_value = 0
    happy_value = 0
    neutral_value = 0
    sad_value = 0
    surprise_value = 0

    weight_value = 0

    # --------------------------------------------------------
    # endregion

    #set up chrome webdriver
    options = Options()
    options.add_argument("start-maximized")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_window_size(WIDTH, HEIGHT)
    driver.get(r"file:\\\Users\kaantanhan\Desktop\variable_fonts_2\main.html");

    #get webcam resolution
    CAM_WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    CAM_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    #main loop
    while True:
        #set css style of html(body element)
        element = driver.find_element(By.TAG_NAME, "Body")
        driver.execute_script("var it = document.getElementsByTagName('Body'); it[0].style.fontVariationSettings = '\"wght\" " + str(weight_value)+ "'; it[0].style.color = 'rgb(" + "255," + str(255-(255*angry_value_red)) + "," + str(255-(255*angry_value_red)) + ")'; it[0].style.fontSize = '" + str(text_size) + "px';  it[0].style.opacity = " + str(text_alph) + ";")

        #get image from webcam and find faces
        ret, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = classifier.predict(roi)[0]
                label = class_labels[preds.argmax()]
                label_position = (x, y)
                cv2.putText(frame, str(label), label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.putText(frame, str(weight_value), (label_position[0],label_position[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                if(label == "Neutral"):
                    weight_value = lerp(weight_value,0,.2)
                if (label == "Happy"):
                    weight_value = lerp(weight_value, 1000, .2)
                if (label == "Sad"):
                    weight_value = lerp(weight_value, -1000, .2)

                #set emotion values
                _value0 = lerp(_value0, preds[0], .1)
                angry_value = _value0

                _value1 = lerp(_value1,preds[1], .1)
                happy_value = _value1

                _value2 = lerp(_value2,preds[2], .1)
                neutral_value = _value2

                _value3 = lerp(_value3,preds[3], .1)
                sad_value = _value3

                _value4 = lerp(_value4,preds[4], .1)
                surprised_value = _value4
            else:
                cv2.putText(frame, 'No Face Found', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        #region Text Behaviours
        #--------------------T E X T   B E H A V I O U R S--------------------
        # detect face and set text size and text alpha

        text_size = lerp(text_size, text_max_size, .25)
        text_alph = lerp(text_alph, text_max_alph, .1)


        # calculate face distance depending on webcam resolution
        if len(faces) > 0:
            face_distance = CAM_HEIGHT - (faces[0][2] * 1.5)

        # check if face is angry then make text go red
        if (angry_value > .3):
            angry_value_red = lerp(angry_value_red, 1, .15)
        else:
            if (angry_value_red > .05):
                angry_value_red = lerp(angry_value_red, 0, .15)
            else:
                angry_value_red = 0
        # --------------------------------------------------------------------
        #endregion

        #show webcam debug
        if(show_cam == True):
            cv2.imshow('Emotion Detector', frame)

        #exit application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #clean memory
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
#endregion