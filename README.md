# sign_language_recognition
Before running the code, we need to install some base libraries like torch, torchvision, pandas, matplotlib, keras and seaborn.

In this thesis, we aim to propose a generalized model for American Sign Language Recognition. Firstly, we utilize the MNIST ASL dataset to propose a CNN-based model. The code file for the proposed model is present in the Chapter5 directory.
Chapter_5.py file has the proposed model architecture, training and results. The training and testing MNIST dataset for the CNN-based model is also present in the same directory under CSV files with the names “sign_mnist_train.csv”  and “sign_mnist_train.csv” respectively. 


Next, we aim to propose a generalized model that performs on real-world images. We utilize an ASL dataset which has colored images collected from real-world scenarios having backgrounds and taken from different perspectives. The dataset can be found at 'https://www.kaggle.com~/datasets/grassknoted/asl-alphabet' and should be dowloaded at Chapter_6/data folder location.
We first re-use the proposed CNN model architecture by re-training it on the new dataset and evaluating its performance on the ASL test data. Chapter6_new/CNN_no_noise.py contains the code for the same. 

Next, we add Gaussian noise to the ASL data with a variance of 0.1 and train the CNN model on the noisy data. We then evaluate the proposed CNN-based model on test data having the same noise at a 0.1 variance level. The code for this can be found in Chapter_6/CNN_gaussian_noise.py.

As the CNN-based model desnt give fair accuracy, we propose a transfer learning-based model. We train the data with Guassian noise with variance of 0.1 and test the model's performance. The code for the same is in Chapter_6/transfer_learning_model.py. We test the generalizability of the model by creating different test data with varying noise levels. This can be done by varying the noise variance level for test data in transformations in transfer_learning_model.py.

For clarity, we also include the graphs for performance evaluation in the Figures folder in .eps format. Chapter_5 and Chpater_6 folders contain the figures for the respective chapters.
