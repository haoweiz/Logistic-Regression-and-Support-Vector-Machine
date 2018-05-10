import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC

f = open('result.txt','w')

def printAndwrite(strs):
    f.write(strs + '\n')
    print(strs)
    f.flush()

def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    # initialWeights:(716,)  train_data:(50000,715)  labeli:(50000,1)
    train_data, labeli = args

    n_data = train_data.shape[0]         #50000
    n_features = train_data.shape[1]     #715

    weight = initialWeights[:,np.newaxis];   # (716,1)
    train_data_bias = np.concatenate((np.ones(shape = (n_data,1)), train_data), axis = 1)  # (50000,716)
    C = sigmoid(np.dot(train_data_bias, weight))  # sigmoid((50000,716) * (716,1)) = (50000,1)
    error = 0 - (1/n_data) * np.sum(np.multiply(labeli,np.log(C)) + np.multiply(np.subtract(1,labeli), np.log(np.subtract(1,C))))
    error_grad = np.reshape(np.multiply(1/n_data, np.dot(np.transpose(np.subtract(C,labeli)), train_data_bias)),n_features+1)
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    #printAndwrite('error = ' + str(error));
    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    # W:(716,10) data:(50000,715)
    #label = np.zeros((data.shape[0], 1))
    data_bias = np.concatenate((np.ones(shape = (data.shape[0],1)), data), axis = 1) #(50000,716)
    label = np.argmax(np.dot(np.transpose(W),np.transpose(data_bias)),0)   #(10,716) * (716,50000) = (10,50000)
    label = label.reshape(label.shape[0],1)
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    # train_data:(50000,715), labeli:(50000,10), params:(7160,)
    train_data, labeli = args
    n_data = train_data.shape[0]       #50000
    n_feature = train_data.shape[1]    #715
    n_class = labeli.shape[1]          #10
    error_grad = np.zeros((n_feature + 1, n_class))   #(716,10)
    error = 0
    train_data_bias = np.concatenate((np.ones(shape = (n_data,1)), train_data), axis = 1)  # (50000,716)
    weight = params[:,np.newaxis].reshape(n_feature+1,n_class)   # (716,10)

    for i in range(0, n_data):
        theta_top = np.exp(np.dot(train_data_bias[i], weight)).reshape(1,n_class)              #(1,10)
        theta_bottom = np.sum(theta_top, axis=1)           #(1,)
        theta = theta_top / theta_bottom.reshape(theta_bottom.shape[0],1) #(1,1)
        a = np.dot(train_data_bias[i].reshape(train_data_bias[i].shape[0], 1),theta - labeli[i].reshape(1, labeli[i].shape[0]))   #(716,10)
        error_grad = np.add(error_grad, a)    #(716,10)
        error += np.sum(labeli[i].reshape(1, labeli[i].shape[0]) * np.log(theta))  #1

    error_grad = (error_grad/n_data).flatten()    #(7160,)
    error = -n_data * error
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    #printAndwrite('error = ' + str(error));
    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    # W:(716,10)  data:(50000,715)
    #label = np.zeros((data.shape[0], 1))

    data_bias = np.concatenate((np.ones(shape = (data.shape[0],1)), data), axis = 1) #(50000,716)
    label = np.argmax(np.dot(np.transpose(W),np.transpose(data_bias)),0)   #(10,716) * (716,50000) = (10,50000)
    label = label.reshape(label.shape[0],1)
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return label

"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()


printAndwrite('--------------Logistic Regression-------------------')

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
printAndwrite('Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
printAndwrite('Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
printAndwrite('Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""

printAndwrite('--------------SVM-------------------')
##################
# YOUR CODE HERE #
##################
# kernel = 'linear'
clf1 = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf1.fit(train_data,train_label.flatten())
printAndwrite('kernel = linear')
printAndwrite('Training set Accuracy:' + str(100 * clf1.score(train_data, train_label.flatten())) + '%')
printAndwrite('Validation set Accuracy: '+ str(100 * clf1.score(validation_data, validation_label.flatten()))+'%') 
printAndwrite('Testing set Accuracy:' + str(100 * clf1.score(test_data, test_label.flatten())) + '%\n')

# gamma = 1
clf2 = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=1.0, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf2.fit(train_data,train_label.flatten())
printAndwrite('gamma = 1')
printAndwrite('Training set Accuracy:' + str(100 * clf2.score(train_data, train_label.flatten())) + '%')
printAndwrite('Validation set Accuracy: '+ str(100 * clf2.score(validation_data, validation_label.flatten()))+'%') 
printAndwrite('Testing set Accuracy:' + str(100 * clf2.score(test_data, test_label.flatten())) + '%\n')

# Default
clf3 = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf3.fit(train_data,train_label.flatten())
printAndwrite('Default')
printAndwrite('Training set Accuracy:' + str(100 * clf3.score(train_data, train_label.flatten())) + '%')
printAndwrite('Validation set Accuracy: '+ str(100 * clf3.score(validation_data, validation_label.flatten()))+'%') 
printAndwrite('Testing set Accuracy:' + str(100 * clf3.score(test_data, test_label.flatten())) + '%\n')

# C:{1,10,20,30,40,50,60,70,80,90,100}
c = 1
clf4 = SVC(C=c, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf4.fit(train_data,train_label.flatten())
printAndwrite('C = ' + str(c))
printAndwrite('Training set Accuracy:' + str(100 * clf4.score(train_data, train_label.flatten())) + '%')
printAndwrite('Validation set Accuracy: '+ str(100 * clf4.score(validation_data, validation_label.flatten()))+'%') 
printAndwrite('Testing set Accuracy:' + str(100 * clf4.score(test_data, test_label.flatten())) + '%\n')

for c in range(10,101,10):
    clf5 = SVC(C=c, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
    clf5.fit(train_data,train_label.flatten())
    printAndwrite('C = ' + str(c))
    printAndwrite('Training set Accuracy:' + str(100 * clf5.score(train_data, train_label.flatten())) + '%')
    printAndwrite('Validation set Accuracy: '+ str(100 * clf5.score(validation_data, validation_label.flatten()))+'%') 
    printAndwrite('Testing set Accuracy:' + str(100 * clf5.score(test_data, test_label.flatten())) + '%\n')


"""
Script for Extra Credit Part
"""

# FOR EXTRA CREDIT ONLY
printAndwrite('--------------Multi-Logistic Regression-------------------')

W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

args_multi_test = (test_data, Ytest)
error_multi_test = mlrObjFunction(W_b,*args_multi_test)

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
printAndwrite('Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
printAndwrite('Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
printAndwrite('Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
f.close()

