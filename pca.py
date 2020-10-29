import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d
import plotly
import plotly.graph_objs as go
from tqdm import tqdm
from sklearn.decomposition import PCA

folders = ['1', '4', '7', '8', '9', '10', '16', '17', '21', '22',
           '23', '28', '31', '34', '36', '43', '46', '48', '53', '57',
           '60', '61', '64', '67', '68']

train_labels = []
train_imgs = []
test_labels = []
test_imgs = []

for folder in tqdm(folders):
    img_folder = os.path.join('PIE', folder)
    test_index_list = random.sample(range(1, 171), 51)
    for img_file in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_file)
        img = plt.imread(img_path)
        #30% testing and 70% training (0.3*170 = 51)
        img_name = os.path.splitext(img_file)[0]
        if int(img_name) in test_index_list:
            test_imgs.append(img)
            test_labels.append(int(folder))
        else:
            train_imgs.append(img)
            train_labels.append(int(folder))

print('Image size: {}'.format(train_imgs[0].shape))
print('Training set: ' + str(len(train_labels)) + "     Testing set: " + str(len(test_labels)))

#add own selfie photos here
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

train_imgs.extend(own_train_imgs)
train_labels.extend(own_train_labels)
test_imgs.extend(own_test_imgs)
test_labels.extend(own_test_labels)
print('New Training set: ' + str(len(train_labels)) + "     New Testing set: " + str(len(test_labels)))

train_x = np.array(train_imgs)
train_y = np.array(train_labels)
test_x = np.array(test_imgs)
test_y = np.array(test_labels)
#vectorize images
train_x = train_x.reshape(len(train_imgs), -1)
test_x = test_x.reshape(len(test_imgs), -1)
print('Vectorized Training set: {}     Vectorized Test set: {}'.format(train_x.shape, test_x.shape))

#sampling 500
samples = random.sample(range(train_x.shape[0]), 500)
sampled_x = train_x[samples]
sampled_y = train_y[samples]

pca = PCA(n_components=2)
pca.fit(sampled_x)
#reduce dimensionality
x_reduced = pca.transform(sampled_x)

#2D plot
plt.figure(figsize = (7, 5))
plt.scatter(x_reduced[0, samples], x_reduced[1, samples], marker = '.', alpha=1)

