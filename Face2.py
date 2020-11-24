import cv2 as cv
import face_recognition
import ctypes
from ctypes.wintypes import HWND, LPWSTR, UINT
import time
import webbrowser


def MessageBoxW(hwnd, text, caption, utype):
    result = _MessageBoxW(hwnd, text, caption, utype)
    if not result:
        raise ctypes.WinError(ctypes.get_last_error())
    return result


def dia(message, title):
    try:
        result = MessageBoxW(None, message, title, MB_OK)
        if result == IDYES:
            print("user pressed ok")
        elif result == IDNO:
            print("user pressed no")
        elif result == IDCANCEL:
            print("user pressed cancel")
        else:
            print("unknown return code")
    except WindowsError as win_err:
        print("Error: An error occurred:\n{}".format(win_err))


def load_caffe_models():
    age_net = cv.dnn.readNetFromCaffe('model/deploy_age.prototxt', 'model/age_net.caffemodel')
    gender_net = cv.dnn.readNetFromCaffe('model/deploy_gender.prototxt', 'model/gender_net.caffemodel')
    return [age_net, gender_net]


def detectAndDisplay(frame, ageNet, genderNet, first_read):
    blinkFlag = False
    eyeOpenFlag = False
    eyeCloseFlag = True

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    # Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    print("Found {0} Faces!".format(len(faces)))
    # Detect Gender and Age
    blob = cv.dnn.blobFromImage(frame, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=True)
    genderNet.setInput(blob)
    gender_preds = genderNet.forward()
    gender = gender_list[gender_preds[0].argmax()]
    print("Gender : " + gender)
    ageNet.setInput(blob)
    age_preds = ageNet.forward()
    age = age_list[age_preds[0].argmax()]
    print("Age Range: " + age)
    text = "{}, {}".format(gender, age)

    # Loop through each face
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        frame = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2, cv.LINE_8)
        cv.putText(frame, text, (x, y + h), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        faceROI = frame_gray[y:y + h, x:x + w]
        # In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0), 4)

        if (len(eyes) == 2):
            no_of_eyes = 2
            print("Two eyes present.")
            # Check if program is running for detection
            eyeOpenFlag = True
            if (first_read):
                cv.putText(frame, "Eye detected",  (70, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
            else:
                cv.putText(frame, "Eyes open!", (70, 70), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        else:
            eyeCloseFlag = True
            if (first_read):
                # To ensure if the eyes are present before starting
                cv.putText(frame, "No eyes detected", (70, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
            else:
                # This will print on console and restart the algorithm
                print("Blink detected--------------")
                blinkFlag = True
                cv.putText(frame, "Blink!!!", (70, 70), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

                cv.waitKey(3000)
                first_read = True
    # Print the box around face
    cv.imshow("Capture Image(space bar)", frame)
    return blinkFlag

if __name__ == "__main__":

    start = time.time()
    CONST_WEBCAM_IMAGE_FILENAME = "opencv_frame.jpg"
    CONST_IMAGE_DIR = "image/"
    CONST_IMAGE_FILES_FACE_DETECTION = ["sk2.jpg", "SUMAN1.JPG", "SUMAN2.jpg"]
    CONST_URL = "https://retail.sc.com/in/nfs/login.htm"
    KNOWN_FACE_LIST = []
    # Loading saved images and Reading saved image face
    try:
        for img_file in CONST_IMAGE_FILES_FACE_DETECTION:
            img = face_recognition.load_image_file(CONST_IMAGE_DIR + img_file)
            img_face = face_recognition.face_encodings(img)[0]
            KNOWN_FACE_LIST.append(img_face)
    except IndexError as e:
        print(e)
        print("Error: Cannot read face from file. Please try again!")

    # Declare Variables
    _user32 = ctypes.WinDLL('user32', use_last_error=True)
    _MessageBoxW = _user32.MessageBoxW
    _MessageBoxW.restype = UINT  # default return type is c_int, this is not required
    _MessageBoxW.argtypes = (HWND, LPWSTR, LPWSTR, UINT)
    MB_OK = 0
    MB_OKCANCEL = 1
    MB_YESNOCANCEL = 3
    MB_YESNO = 4
    IDOK = 0
    IDCANCEL = 2
    IDABORT = 3
    IDYES = 6
    IDNO = 7
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
    gender_list = ['Male', 'Female']
    age_net, gender_net = load_caffe_models()
    img_counter = 0
    face_cascade_name = cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
    eyes_cascade_name = cv.data.haarcascades + 'haarcascade_eye.xml'
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eyes_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
    # 1. Load the cascades
    if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
        print('Error loading face cascade')
        exit(1)
    if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
        print('Error loading eyes cascade')
        exit(1)
    interval = 2
    loop_counter = 0
    end = time.time()
    delta = end - start
    print("It took %.2f seconds for loading data." % delta)

    # Open WebCam and change settings
    start = time.time()
    first_read = True
    cap = cv.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    cap.set(10, 100)
    cv.namedWindow("Capture Image(space bar)")
    if not cap.isOpened:
        print('Error opening video capture')
        exit(1)
    while True:
        loop_counter = loop_counter + 1
        ret, frame = cap.read()
        if frame is None:
            print('No captured frame -- Break!')
            break
        # Running Face Detection Program
        blinkFlag = detectAndDisplay(frame, age_net, gender_net, first_read)
        first_read = False
        if blinkFlag:
            print("User is blinking")
        # If space bar pressed then save the picture
        if cv.waitKey(1) == 32 or loop_counter % 23 == 0:
            # SPACE pressed

            cv.imwrite(CONST_IMAGE_DIR + CONST_WEBCAM_IMAGE_FILENAME, frame)
            print("Webcam frame is written in {}!".format(CONST_WEBCAM_IMAGE_FILENAME))
            img_counter += 1

            # Load webcam frame
            unknown_image = frame
            unknown_face = 0
            captureFlag = True
            try:
                unknown_face = face_recognition.face_encodings(unknown_image)[0]
            except IndexError as e:
                print(e)
                captureFlag = False
                print("Error: Face has not been captured property from webcam. Please try again!")
                dia("Face has not been captured property from webcam. Please try again!", "Failure!")
            if captureFlag:
                flag = False
                results = face_recognition.compare_faces(KNOWN_FACE_LIST, unknown_face, tolerance=.60)
                for x in results:
                    if x:
                        end = time.time()
                        delta = end - start
                        dia("Face has been detected!!!", "Success!")

                        # Load URL for Login
                        new = 2
                        url = CONST_URL
                        # webbrowser.open(url, new=new)
                        cap.release()
                        cv.destroyAllWindows()
                        print("It took %.2f seconds to detect face successfully." % delta)
                        #exit(0)

        # To Exit Press Esc for sometime
        if cv.waitKey(1) == 27:
            break
    cap.release()
    cv.destroyAllWindows()
    end = time.time()
    delta = end - start
    print("It ran for %.2f seconds." % delta)
    exit(0)
