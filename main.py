import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import os
import cv2
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score





train_folder = "Assignment dataset/train"
test_folder= "Assignment dataset/test"


x_train = []
y_train = []


classes_of_train = os.listdir(train_folder)
for class_name in classes_of_train:
    class_path = os.path.join(train_folder, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(img, (128, 64))
        hog_features = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        x_train.append(hog_features)
        y_train.append(class_name)


x_train = np.array(x_train)
y_train = np.array(y_train)


x_test = []
y_test = []

classes_of_test = os.listdir(test_folder)
for class_name in classes_of_test:
    class_path = os.path.join(test_folder, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(img, (128, 64))
        hog_features = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        x_test.append(hog_features)
        y_test.append(class_name)


x_test = np.array(x_test)
y_test= np.array(y_test)

#modelling
C = 1
svc=svm.SVC(kernel='linear', C=C).fit(x_train, y_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.8, C=C).fit(x_train, y_train)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(x_train, y_train)




for i, clf in enumerate((svc,rbf_svc, poly_svc)):
    predictions = clf.predict(x_test)
    accuracy = np.mean(predictions == y_test)
    print(accuracy)

