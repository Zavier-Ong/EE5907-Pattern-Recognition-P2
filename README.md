# EE5907_CA2

This code is wrriten in python 3.8 64 bit

## Dependencies
1. Ensure that Python 3.8.6 64 bit is downloaded (cnn.py will not be able to work on 32 bit version)
2. Ensure that numpy, tqdm, sklearn, tensorflow and matplotlib packages are installed
3. Ensure that PIE folder and mine (selfie folder) is in the same directory as the source codes

##Running pca.py.
1. Opening Command Prompt in source code folder
2. Type "py pca.py"
3. A load bar should appearing representing the reading of images of the folders.
4. Things to observed:
	* output in console
	* figure containing projected data vector in 2D
	* figure containing projected data vector in 3D
	* figure containing 3 eigenfaces

##Running lda.py.
1. Opening Command Prompt in source code folder
2. Type "py lda.py"
3. A load bar should appearing representing the reading of images of the folders.
4. Things to observed:
	* output in console
	* figure containing distribution of sampled data in 2D
	* figure containing distrubution of sampled data in 3D

##Running gmm.py.
1. Opening Command Prompt in source code folder
2. Type "py gmm.py"
3. A load bar should appearing representing the reading of images of the folders.
4. Things to observed:
	* output in console
	* figure containing 2D plot of raw image clustering
	* figure containing 2D plot of PCA 80 clustering
	* figure containing 2D plot of PCA 200 clustering

##Running svm.py.
1. Opening Command Prompt in source code folder
2. Type "py svm.py"
3. A load bar should appearing representing the reading of images of the folders.
4. Things to observed:
	* output in console
	* bar chart containing the classfication acccuracy with different parameters and dimensions

##Running cnn.py
1. Opening Command Prompt in source code folder
2. Type "py cnn.py"
3. A load bar should appearing representing the reading of images of the folders.
4. Things to observed:
	* output in console
	* graph containing the test accuracy and test loss against epochs
5. In order to run cnn with the different network architectures, changes to source code is needed.
	* open cnn.py on any editor
	* look for the comment network architecture change #num
	* uncomment the line below and comment the line above that comment
	* save the file and start from step 2.
	* you should be able to observe a change in the model summary.

Please contact me if you have any difficulties
Author: Zavier (A0138993L)
Email : e0002878@u.nus.edu