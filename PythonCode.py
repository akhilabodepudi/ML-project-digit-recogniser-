# ML-project-digit-recogniser
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
data=pd.read_csv("/content/drive/MyDrive/MachineLearningProject/dataset.csv")

data=data[:3437]
print(data)

x = data.iloc[:,1:].values
print(x)
print(x.shape)

y = data.iloc[:,:1]["label"]
print(y)
print(y.shape)

x=x/255
print(x)
print(x.shape)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)

from PIL import Image,ImageChops
import numpy as np
def preprocess_img(img_path):
    image = Image.open(img_path,'r')
    image = image.convert('L')
    image = ImageChops.invert(image)
    image = image.resize((28,28))
    px_data = list(image.getdata())
    for i in range(len(px_data)):
        if px_data[i]/255 <= 0.43:
            px_data[i] = 0
    px_data = np.array(px_data)/255
    return px_data
    
image=input("Enter path of the Image:")
pix_img =preprocess_img(image)

import matplotlib.pyplot as plt
plt.imshow(pix_img.reshape(28,28))

import cv2 
import matplotlib.pyplot as plt

# Load sample image
test_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

# Preview sample image
plt.imshow(test_image, cmap='gray')

# Format Image
img_resized = cv2.resize(test_image, (28, 28), interpolation=cv2.INTER_LINEAR)
img_resized = cv2.bitwise_not(img_resized)

# Preview reformatted image
plt.imshow(img_resized, cmap='gray')

from sklearn.naive_bayes import GaussianNB

def Naive_Bayes_classifier(image):
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred = gnb.predict(image)
    accuracy = gnb.score(x_test,y_test) 
    print("MutiNomial Naive_Bayes_Classification")
    print("Accuracy :",round(accuracy,2))
    print("label for given image :",y_pred,end="\n")
Naive_Bayes_classifier([pix_img])

from sklearn.tree import DecisionTreeClassifier
def DecisionTree_Classifier(image,type):
    ans = None
    if type=="gini":
        clf_gini = DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=3, min_samples_leaf=5)
        clf_gini.fit(x_train, y_train)
        accuracy = clf_gini.score(x_test,y_test)
        ans = clf_gini.predict(image)
    elif type=="entropy":
        clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth = 3, min_samples_leaf = 5)
        clf_entropy.fit(x_train, y_train)
        accuracy = clf_entropy.score(x_test,y_test)
        ans = clf_entropy.predict(image)
    else:
        print("valid types are: 1)gini 2)entropy")
    print("DecisionTree_Classifier: ")
    print("Accuracy :",round(accuracy,2))
    print("label for given image :",ans,end="\n")

DecisionTree_Classifier([pix_img],"gini")

from sklearn.ensemble import RandomForestClassifier
def RandomForest_Classifier(image):
    rd = RandomForestClassifier()
    rd.fit(x_train,y_train)
    accuracy = rd.score(x_test,y_test) 
    rd_pred = rd.predict(image)
    print("RandomForest_Classifier: ")
    print("Accuracy :",round(accuracy,2))
    print("label for given image :",rd_pred,end="\n")

RandomForest_Classifier([pix_img])

from sklearn.neighbors import KNeighborsClassifier 
def KNeighbors_Classifier(image):
    classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
    classifier.fit(x_train, y_train)
    accuracy = classifier.score(x_test,y_test)
    ans = classifier.predict(image)
    print("KNeighbors_Classifier: ")
    print("Accuracy :",round(accuracy,2))
    print("label for given image :",ans,end="\n")

KNeighbors_Classifier([pix_img])

from sklearn.svm import SVC
def SVC_classifier(image):
    clf = SVC(kernel='linear') 
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test,y_test)
    ans = clf.predict(image)
    print("SVC_classifier: ")
    print("Accuracy :",round(accuracy,2))
    print("label for given image :",ans,end="\n")

SVC_classifier([pix_img])

from sklearn.ensemble import BaggingClassifier
def Bagging_classifier(image):
    bagging = BaggingClassifier(base_estimator=KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
    bagging.fit(x_train,y_train)
    accuracy = bagging.score(x_test,y_test)
    ans = bagging.predict(image)
    print("Bagging_classifier: ")
    print("Accuracy :",round(accuracy,2))
    print("label for given image :",ans,end="\n")

Bagging_classifier([pix_img])

from sklearn.ensemble import AdaBoostClassifier
def Boosting_classifier(image):
    boosting = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
    boosting.fit(x_train,y_train)
    accuracy = boosting.score(x_test,y_test)
    ans = boosting.predict(image)
    print("Boosting classifier: ")
    print("Accuracy :",round(accuracy,2))
    print("label for given image :",ans,end="\n")

Boosting_classifier([pix_img])

from sklearn.neural_network import MLPClassifier
def ANN(image):
  clf= MLPClassifier(hidden_layer_sizes=(500,250,125,75,35),activation="relu",solver='sgd',learning_rate_init= 0.01, max_iter=500)
  clf.fit(x_train, y_train)
  accuracy = clf.score(x_test,y_test)
  ans = clf.predict(image)
  print("Artificial Neural Network classifier: ")
  print("Accuracy :",round(accuracy,2))
  print("label for given image :",ans,end="\n")
  shape=[coef.shape for coef in clf.coefs_]
  print(shape)

ANN([pix_img])

