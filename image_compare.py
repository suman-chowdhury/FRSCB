#import face_recognition
import cv2 as cv
import os
import stat



def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)

def load_caffe_models():
    age_net = cv.dnn.readNetFromCaffe('model/deploy_age.prototxt', 'model/age_net.caffemodel')
    gender_net = cv.dnn.readNetFromCaffe('model/deploy_gender.prototxt', 'model/gender_net.caffemodel')
    return (age_net, gender_net)


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']
age_net, gender_net = load_caffe_models()

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
#nose_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'Nariz.xml')
directory_in_str = 'image/'
directory = os.fsencode(directory_in_str)
for filename in os.listdir(directory):
    file = os.fsdecode(filename)
    if os.path.isfile(directory_in_str+file):
        #print(file+" is a file")
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".JPG"):
            print("Processing File: " + file)
            print(directory_in_str + file)
            img = cv.imread(directory_in_str + file)
            imgOrg = cv.imread(directory_in_str + file)

            blob = cv.dnn.blobFromImage(img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=True)
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            print("Gender : " + gender)

            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            print("Age Range: " + age)

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            gray = cv.equalizeHist(gray)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            text = "{}, {}".format(gender, age)

            for (x, y, w, h) in faces:
                cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3, cv.FILLED)
                cv.putText(img, text, (x, y + h), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)
                # nose = nose_cascade.detectMultiScale(roi_gray)
                # for (ex, ey, ew, eh) in nose:
                #     cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)

                #imgS = ResizeWithAspectRatio(img, 320, 240, inter=cv.INTER_AREA)
                #im = cv.imshow(file, imgS)


                im_v = cv.hconcat([img, imgOrg])

                imgC = ResizeWithAspectRatio(im_v, 640, 480, inter=cv.INTER_AREA)
                cv.imshow(file, imgC)

                if cv.waitKey(1) == 32:
                    continue
                if cv.waitKey(1) == 27:
                    break
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            print("Skipping file (not an image)...  " + file)
    else:
        print("Skipping file (not a file)...  "+file)
        continue
print("Program end")




