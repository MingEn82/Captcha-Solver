import os
import xml.etree.cElementTree as ET
import csv
import numpy as np
import cv2
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping

DATA_PATH = r"D:\Python projects\AutoTempTaking\CAPTCHAsolver\Data\Images"
XML_PATH = r"D:\Python projects\AutoTempTaking\CAPTCHAsolver\Data\xml"
OUTPUT_FOLDER = r"D:\Python projects\AutoTempTaking\CAPTCHAsolver\Data\extracted letters"


def xml_data_to_csv(xml_path, csv_path, savename):
    """
    Convert xml files to csv - deprecated
    """
    with open(os.path.join(csv_path, savename+'.csv'), 'w', newline='') as csvfile:
        fieldnames = ['filename', 'width', 'height', 'name', 'xmin', 'ymin', 'xmax', 'ymax']
        xmlwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        xmlwriter.writeheader()
        for file in os.listdir(xml_path):
            if file.endswith('.xml'):
                tree = ET.parse(os.path.join(xml_path, file)).getroot()
                filename = tree.find('filename').text
                width = tree.find('size').find('width').text
                height = tree.find('size').find('height').text
                for char in tree.findall('object'):
                    name = char.find('name').text
                    xmin = char.find('bndbox').find('xmin').text
                    ymin = char.find('bndbox').find('ymin').text
                    xmax = char.find('bndbox').find('xmax').text
                    ymax = char.find('bndbox').find('ymax').text
                    xmlwriter.writerow({'filename': filename, 'width': width, 'height': height,
                                        'name': name, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})


def char_separator(DATA_PATH, XML_PATH, OUTPUT_FOLDER):
    """
    To separate the captcha images into individual letters for training and validating
    """
    counts = {}

    for img in os.listdir(DATA_PATH):
        image = cv2.imread(os.path.join(DATA_PATH, img))
        image = cv2.medianBlur(image, 5)

        # edge detection
        edges = cv2.Canny(image, 100, 200)
        edges = cv2.bitwise_not(edges)

        base_name = os.path.splitext(img)[0]
        tree = ET.parse(os.path.join(XML_PATH, base_name + '.xml')).getroot()
        for char in tree.findall('object'):
            name = char.find('name').text
            xmin = int(char.find('bndbox').find('xmin').text)
            ymin = int(char.find('bndbox').find('ymin').text)
            xmax = int(char.find('bndbox').find('xmax').text)
            ymax = int(char.find('bndbox').find('ymax').text)

            letter_image = edges[ymin:ymax, xmin:xmax]
            save_path = os.path.join(OUTPUT_FOLDER, name)

            # creating different output folder for storing different letters
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # write the letter image to a file
            count = counts.get(name, 1)
            p = os.path.join(save_path, "{}.png".format(str(count)))
            cv2.imwrite(p, letter_image)

            # increment the count
            counts[name] = count + 1


data = []
labels = []
for image in paths.list_images(OUTPUT_FOLDER):
    # pre-processing images - may not be necessary as I already did some
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # standardise images to 30x30
    img = cv2.resize(img, (30,30))
    # expand dimensions of image from 30x30 to 30x30x1 - Tensorflow requirement
    img = np.expand_dims(img, axis=2)

    label = image.split(os.path.sep)[-2]
    data.append(img)
    labels.append(label)

data = np.array(data, dtype='float')
labels = np.array(labels)

# normalise the points
data = data/255.0
(train_imgs, test_imgs, train_labels, test_labels) = train_test_split(data, labels, test_size=0.2, random_state=0)

# labels index are based on label file position
lb = LabelBinarizer().fit(train_labels)
train_labels = lb.transform(train_labels)
test_labels = lb.transform(test_labels)

model = Sequential()
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(30, 30, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(28, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# using early stopping for avoiding overfitting
estop = EarlyStopping(patience=10, mode='min', min_delta=0.001, monitor='val_loss')

model.fit(train_imgs, train_labels, validation_data=(test_imgs, test_labels), batch_size=32, epochs=50, verbose=1, callbacks = [estop])

model.save("model.h5")

# loads model for future use
# model = keras.models.load_model("model.h5")

