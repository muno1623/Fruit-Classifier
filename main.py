import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.datasets import mnist, cifar10
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow.keras.utils as np_utils
import tensorflow.keras.utils as np_utils
from tensorflow.keras.models import load_model
import pydot
import os
from os import listdir
from os.path import isfile, join
import sys
import shutil
import pickle

opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
num_classes = 81
img_rows, img_cols = 32, 32
batch_size = 16

train_data_dir = './fruits-360/Training'
validation_data_dir = './fruits-360/Test'

# Let's use some data augmentaiton
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

class_labels = validation_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())

# #Training
# model = Sequential()
#
# # Padding = 'same'  results in padding the input such that
# # the output has the same length as the original input
# model.add(Conv2D(32, (3, 3), padding='same', input_shape= (img_rows, img_cols, 3)))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))
#
# print(model.summary())
#
# ## Checkpoint, Early Stopping, and Learning rates
# checkpoint = ModelCheckpoint("E:/Computer Vision and Machine Learning Projects/BuildingCNN/Trained Models/Fruit_Classifier_Checkpoint.h5",
#                              monitor="val_loss",
#                              mode="min",
#                              save_best_only=True,
#                              verbose=1)
#
# earlystop = EarlyStopping(monitor = 'val_loss', # value being monitored for improvement
#                           min_delta = 0, #Abs value and is the min change required before we stop
#                           patience = 3, #Number of epochs we wait before stopping
#                           verbose = 1,
#                           restore_best_weights = True) #keeps the best weigths once stopped
#
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience = 3, verbose = 1, min_delta = 0.001)
#
# callbacks = [checkpoint, earlystop, reduce_lr]
#
# # We use a very small learning rate
# model.compile(loss='categorical_crossentropy',
#               optimizer=RMSprop(lr=0.001),
#               metrics=['accuracy'])
#
# nb_train_samples = 41322
# nb_validation_samples = 13877
# epochs = 10
#
# history = model.fit(
#     train_generator,
#     steps_per_epoch = nb_train_samples // batch_size,
#     epochs = epochs,
#     callbacks = callbacks,
#     validation_data = validation_generator,
#     validation_steps = nb_validation_samples // batch_size)
#
# #Visualizing Model
# np_utils.plot_model(model, "E:/Computer Vision and Machine Learning Projects/BuildingCNN/model_plot.png",show_shapes=True, show_layer_names=True )
#
# #saving model
# model.save("E:/Computer Vision and Machine Learning Projects/BuildingCNN/fruit_classifier_10ep.h5")
#
# #Saving History of Model
# pickle_out = open("Fruit_history.pickle", "wb")
# pickle.dump(history.history, pickle_out)
# pickle_out.close()
#
# #Loading history of model
# pickle_in = open("Fruit_history.pickle", "rb")
# saved_history = pickle.load(pickle_in)
# print(saved_history)
#
# #plotting model
# #Loss Plot
# history_dict = history.history
# loss_values = history_dict["loss"]
# val_loss_values = history_dict["val_loss"]
# ep = range(1, len(loss_values)+1)
# line1 = plt.plot(ep, val_loss_values, label = "Validation Loss")
# line2 = plt.plot(ep, loss_values, label = "Training Loss")
# plt.setp(line1, linewidth=2.0, marker="+", markersize=10.0)
# plt.setp(line2, linewidth=2.0, marker="4", markersize=10.0)
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.grid(True)
# plt.legend()
# plt.show()
#
# #Accuracy Plot
# loss_values = history_dict["accuracy"]
# val_loss_values = history_dict["val_accuracy"]
# ep = range(1, len(loss_values)+1)
# line1 = plt.plot(ep, val_loss_values, label = "Validation Accuracy")
# line2 = plt.plot(ep, loss_values, label = "Training Accuracy")
# plt.setp(line1, linewidth=2.0, marker="+", markersize=10.0)
# plt.setp(line2, linewidth=2.0, marker="4", markersize=10.0)
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.grid(True)
# plt.legend()
# plt.show()
#
# #Confution Matrix and Classification Report
# Y_pred = model.predict_generator(validation_generator, nb_validation_samples // batch_size+1)
# y_pred = np.argmax(Y_pred, axis=1)
# print('Confusion Matrix')
# print(confusion_matrix(validation_generator.classes, y_pred))
# print('Classification Report')
# class_labels = validation_generator.labels
# target_names = list(class_labels.values())
# print(classification_report(validation_generator.classes, y_pred, target_names=target_names))


#Predction
classifier = load_model("E:/Computer Vision and Machine Learning Projects/BuildingCNN/fruit_classifier_10ep.h5")


def draw_test(name, pred, im, true_label):
    BLACK = [0, 0, 0]
    expanded_image = cv2.copyMakeBorder(im, 160, 0, 0, 500, cv2.BORDER_CONSTANT, value=BLACK)
    cv2.putText(expanded_image, "predited - " + pred, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(expanded_image, "true - " + true_label, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow(name, expanded_image)


def getRandomImage(path, img_width, img_height):
    """function loads a random images from a random folder in our test path """
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0, len(folders))
    path_class = folders[random_directory]
    file_path = path + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0, len(file_names))
    image_name = file_names[random_file_index]
    final_path = file_path + "/" + image_name
    return image.load_img(final_path, target_size=(img_width, img_height)), final_path, path_class


# dimensions of our images
img_width, img_height = 32, 32

files = []
predictions = []
true_labels = []
# predicting images
for i in range(0, 10):
    path = './fruits-360/Test/'
    img, final_path, true_label = getRandomImage(path, img_width, img_height)
    files.append(final_path)
    true_labels.append(true_label)
    x = image.img_to_array(img)
    x = x * 1. / 255
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = classifier.predict_classes(images, batch_size=10)
    predictions.append(classes)

for i in range(0, len(files)):
    image = cv2.imread((files[i]))
    draw_test("Prediction", class_labels[predictions[i][0]], image, true_labels[i])
    cv2.waitKey(0)

cv2.destroyAllWindows()
