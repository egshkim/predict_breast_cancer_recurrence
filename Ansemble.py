from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical

# Define base directory path
base_dir = "/data/Projects/Deep_learning/eg/data/NIFTI/2.normalized_nparrays&labels/96_96_32/"

FCNN_x = np.load(base_dir + "ClinicaldataNumpyArray.npy")
CNN_x = np.load(base_dir + "Available_images.npy")
y = np.load(base_dir + "Available_labels.npy")
y = to_categorical(y, num_classes=3, dtype='float32')

FCNN_x_train, FCNN_x_test, FCNN_y_train, FCNN_y_test = train_test_split(
    FCNN_x, y, test_size=0.2, random_state=2022, shuffle=True, stratify=y)
CNN_x_train, CNN_x_test, CNN_y_train, CNN_y_test = train_test_split(
    CNN_x, y, test_size=0.2, random_state=2022, shuffle=True, stratify=y)

FCNNModelDir = "/data/Projects/Deep_learning/eg/FCNNforClinicaldata/model/"
FCNNModelName = "FCNN_5dense_lr-0.01epochs-100batch_size-128he_normal-1Dropout-0"

CNNModelDir = "/data/Projects/Deep_learning/eg/Custom3DResNet/model/"
CNNModelName = "Custom3DResNet152V2_Dropout_WithAug_epochs-400_StartLr-0.001_lrdecayrate0.8_lrdecaytimes10"

AnsembleResultDir = "/data/Projects/Deep_learning/eg/Ansemble/result/"

FCNN = tf.keras.models.load_model(f"{FCNNModelDir}{FCNNModelName}")
CNN = tf.keras.models.load_model(f"{CNNModelDir}{CNNModelName}")

FCNN_Prediction = FCNN.predict(FCNN_x_test)
CNN_Prediction = CNN.predict(CNN_x_test)

prediction = (FCNN_Prediction + CNN_Prediction)/2

y_pred = np.argmax(prediction, axis=1)
y_true = np.argmax(FCNN_y_test, axis=1)

with open(f"{AnsembleResultDir}Ansemble.txt", "w") as text_file:
    print(classification_report(y_true, y_pred, target_names=[
          'NoRECUR', 'RECURwithin3yrs', 'RECURafter3yrs']), file=text_file)
