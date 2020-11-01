import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from sklearn import svm

folders = ['1', '4', '7', '8', '10', '13', '16', '17', '21', '22',
           '23', '28', '31', '34', '36', '43', '45', '48', '53', '55',
           '57', '61', '64', '67', '68']

#mixed dataset (PIE + selfie)
mixed_train_labels = []
mixed_train_imgs = []
mixed_test_labels = []
mixed_test_imgs = []
#pie only dataset
pie_test_labels = []
pie_test_imgs = []

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
            #add to pie dataset
            pie_test_imgs.append(img)
            pie_test_labels.append(int(folder))
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
#vectorize images
train_x = train_x.reshape(len(mixed_train_imgs), -1)
test_x = test_x.reshape(len(mixed_test_imgs), -1)
print('Vectorized Training set: {}     Vectorized Test set: {}'.format(train_x.shape, test_x.shape))
