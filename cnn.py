import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
import tensorflow.keras as keras

folders = ['1', '4', '7', '8', '10', '13', '16', '17', '21', '22',
           '23', '28', '31', '34', '36', '43', '45', '48', '53', '55',
           '57', '61', '64', '67', '68']

#mixed dataset (PIE + selfie)
mixed_train_labels = []
mixed_train_imgs = []
mixed_test_labels = []
mixed_test_imgs = []

#random to ensure reproducibility
random.seed(6)

for folder in tqdm(folders):
    img_folder = os.path.join('PIE', folder)
    test_index_list = random.sample(range(1, 171), 51)
    for img_file in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_file)
        img = plt.imread(img_path)
        #30% testing and 70% training (0.3*170 = 51)
        img_name = os.path.splitext(img_file)[0]
        if int(img_name) in test_index_list:
            #add to mixed dataset
            mixed_test_imgs.append(img)
            mixed_test_labels.append(int(folder))
        else:
            #add to mixed dataset
            mixed_train_imgs.append(img)
            mixed_train_labels.append(int(folder))

print('Image size: {}'.format(mixed_train_imgs[0].shape))
print('Training set: ' + str(len(mixed_train_labels)) + "     Testing set: " + str(len(mixed_test_labels)))

#selfie only dataset
own_test_imgs = []
own_test_labels = []
own_train_imgs = []
own_train_labels = []
own_test_index_list = random.sample(range(1, 11), 3)

for img_file in os.listdir('mine'):
    img_path = os.path.join('mine', img_file)
    img = plt.imread(img_path)
    #30% testing and 70% training (0.3*170 = 51)
    img_name = os.path.splitext(img_file)[0]
    if int(img_name) in own_test_index_list:
        own_test_imgs.append(img)
        own_test_labels.append(26)
    else:
        own_train_imgs.append(img)
        own_train_labels.append(26)

#added selfie to mixed dataset
mixed_train_imgs.extend(own_train_imgs)
mixed_train_labels.extend(own_train_labels)
mixed_test_imgs.extend(own_test_imgs)
mixed_test_labels.extend(own_test_labels)
print('New Training set: ' + str(len(mixed_train_labels)) + "     New Testing set: " + str(len(mixed_test_labels)))

train_x = np.array(mixed_train_imgs)
train_y = np.array(mixed_train_labels)
test_x = np.array(mixed_test_imgs)
test_y = np.array(mixed_test_labels)

train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], train_x.shape[2], 1))
test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], test_x.shape[1], 1))
#normalize data
train_x = train_x/255
test_x = test_x/255

#new y label
new_labels = np.arange(0, 26, 1)
label_dic = {}
for i in range(len(new_labels)):
    if i==25: #my own photos
        label_dic[26] = new_labels[i]
    else:
        label_dic[int(folders[i])] = new_labels[i]
reverse_labels = {v: k for k, v in label_dic.items()}
vectorize_label = np.vectorize(lambda t: label_dic[t])
train_y = vectorize_label(train_y)
test_y = vectorize_label(test_y)

#construct model
model = Sequential()
#1st layer
model.add(Conv2D(filters=20, kernel_size=5, input_shape=(32, 32, 1)))
model.add(MaxPooling2D(pool_size=2, strides = 2))
#Network architecture change 2
#model.add(MaxPooling2D(pool_size=10, strides=2))
#-------------------------------
#2nd layer
model.add(Conv2D(filters = 50, kernel_size = 5))
model.add(MaxPooling2D(pool_size=2, strides = 2))
#fully connected layer
model.add(Flatten())
model.add(Dense(500))
#Network architecture change 1
#model.add(Dense(100))
#-------------------------------
model.add(Activation('relu'))
#output
model.add(Dense(26))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
print(model.summary())

history = model.fit(train_x, train_y, batch_size = 32, epochs = 20, validation_data=(test_x, test_y), verbose= 2)

test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)
print('Test accuracy: {} Test loss: {}'.format(test_acc, test_loss))
plt.figure(figsize=(12, 4.8))
plt.subplot(121)
plt.plot(history.history['accuracy'], 'g', label = 'accuracy')
plt.plot(history.history['val_accuracy'], 'r', label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.legend()

plt.subplot(122)
plt.plot(history.history['loss'], 'b', label = 'loss')
plt.plot(history.history['val_loss'], 'r', label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()