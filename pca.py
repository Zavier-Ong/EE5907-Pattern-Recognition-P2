import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score

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

#sampling 500
samples = random.sample(range(train_x.shape[0]), 500)
sampled_x = train_x[samples]
sampled_y = train_y[samples]
#2D PCA
pca = PCA(n_components=2)
pca.fit(sampled_x)
#reduce dimensionality
x_reduced2D = pca.transform(sampled_x)

#setting up colors for different folders
folder_color = np.linspace(0,1,25)
np.random.shuffle(folder_color)
colors = cm.rainbow(folder_color)

fig = plt.figure()
#2D plot
ax = fig.add_subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.75, box.height])
#plotting photos
for i, label in enumerate(folders):
    plt.scatter(x_reduced2D[sampled_y == int(label)][:, 0], x_reduced2D[sampled_y == int(label)][:, 1], label=label)

#plotting own photos
plt.plot(x_reduced2D[sampled_y == 26][:, 0], x_reduced2D[sampled_y == 26][:, 1], 'k*', markersize=20)
plt.legend(ncol=2, bbox_to_anchor=(1.03,1), loc='upper left')

#3D PCA
pca = PCA(n_components=3)
pca.fit(sampled_x)
#reduce dimensionality
x_reduced3D = pca.transform(sampled_x)

#3D plot
fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
#plotting photos
for i, label in enumerate(folders):
    ax.scatter(x_reduced3D[sampled_y == int(label)][:,0],
                x_reduced3D[sampled_y == int(label)][:,1],
                x_reduced3D[sampled_y == int(label)][:,2],
                label=label)
#plotting own photos
plt.plot(x_reduced3D[sampled_y==26][:,0], x_reduced3D[sampled_y==26][:,1], x_reduced3D[sampled_y==26][:,2], 'k*', markersize=20)
plt.legend(ncol=2, bbox_to_anchor=(1.03,-0.3), loc='lower left')

#3 eigenfaces
fig3 = plt.figure()
eigenfaces = pca.components_.reshape(3, 32, 32)

for i in range(3):
    ax = fig3.add_subplot(1,3,i+1)
    ax.imshow(eigenfaces[i,:,:])
    ax.set


#vectorize CMU PIE test and selfie sets
test_pie_x = np.array(pie_test_imgs)
test_pie_y = np.array(pie_test_labels)
test_own_x = np.array(own_test_imgs)
test_own_y = np.array(own_test_labels)
#vectorize CMU PIE and selfie test images
test_pie_x = test_pie_x.reshape(len(pie_test_imgs), -1)
test_own_x = test_own_x.reshape(len(own_test_imgs), -1)

def doPCA(dimensions, isOwn):
    knn = KNN()
    n_pca = PCA(n_components=dimensions)
    if isOwn:
        n_pca.fit(train_x)
        train_x_reduced = n_pca.transform(train_x)
        knn.fit(train_x_reduced, train_y)
        test_x_reduced = n_pca.transform(test_own_x)
        result = knn.predict(test_x_reduced)
        print('Accuracy for PCA {} on own selfie test: {}'.format(dimensions, accuracy_score(test_own_y, result)))
    else:
        n_pca.fit(train_x)
        train_x_reduced = n_pca.transform(train_x)
        knn.fit(train_x_reduced, train_y)
        test_x_reduced = n_pca.transform(test_pie_x)
        result = knn.predict(test_x_reduced)
        print('Accuracy for PCA {} on CMU PIE test: {}'.format(dimensions, accuracy_score(test_pie_y, result)))

#PCA 40
doPCA(40, True)
doPCA(40, False)
#PCA 80
doPCA(80, True)
doPCA(80, False)
#PCA 200
doPCA(200, True)
doPCA(200, False)

#show 2D, 3D and 3 eigen faces figures
plt.show()

