import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d
import scipy.spatial.distance as dist
import plotly
import plotly.graph_objs as go
from tqdm import tqdm

dirpath = os.path.join(os.getcwd(), 'PIE')
folders = ['1', '4', '7', '8', '9', '10', '16', '17', '21', '22',
           '23', '28', '31', '34', '36', '43', '46', '48', '53', '57',
           '60', '61', '64', '67', '68']

train_folder_name = []
train_imgs = []
test_folder_name = []
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
            test_folder_name.append(int(folder))
        else:
            train_imgs.append(img)
            train_folder_name.append(int(folder))

print('Image size: {}'.format(train_imgs[0].shape))
print('Training set: ' + str(len(train_folder_name)) + " Testing set: " + str(len(test_folder_name)))

fig, axs = plt.subplots(1, 5)
for i in range(5):
    img = random.choice(train_imgs)
    axs[i].imshow(img, cmap='gray')
    axs[i].axis('off')
plt.show()