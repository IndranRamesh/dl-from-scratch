import typing_extensions
from numpy import dtypes
import numpy as np 

# 1 . Create a simple binary classification problem
# np.random.default_rng(42) is used to initialize the random number generator
rng = np.random.default_rng(42)  

# X is the input data , it is a matrix of size 100x2 , it is used to calculate the dot product between X and w
# y is the output data , it is a vector of size 100 , it is used to calculate the loss

X = rng.normal(size=(100,2)) 
print("X = ", X)

y_true = (X[:,0] + X[:,1] > 0).astype(int)
print("y_true = ", y_true)

# 2 . Initialize weights and bias
w , b = np.array([0.5 , -0.5]) , 0.0  

# w is the weight vector , it is a vector of size 2 , it is used to calculate the dot product between X and w
# b is the bias , it is a scalar value , it is used to shift the values of the dot product

# 3 . Weighted sum (linear transformation)
z = X @ w + b  # This is the dot product between X and w (x1w1 + x2w2) + bias b

# 4 . Sigmoid activation function 
def sigmoid(z):
    # This is the sigmoid activation function
    # it squashes the values between 0 and 1
    return 1 / (1 + np.exp(-z))  
    
# y_pred is the predicted value, it is a probability between 0 and 1
y_pred = sigmoid(z) 

# 5 . Binary cross-entropy loss function
def binary_cross_entropy(y_train,y_pred):
    eps = 1e-8 # This is a small value to prevent division by zero
    # it's used to avoid the log of 0
    y_pred = np.clip(
        y_pred,
        eps,
        1-eps
        ) 
    # This is to clip the values of y_pred to be between 
    # eps and 1-eps
    # It's used to avoid the log of 0
    
    # loss function
    # loss = -np.mean(y_train * np.log(y_pred) + (1-y_train) * np.log(1-y_pred))
    loss = -np.mean(y_train * np.log(y_pred) + (1-y_train) * np.log(1-y_pred))
    # this is the binary cross entropy loss function , 
    # it is used to calculate the loss 
    # between the predicted values and the actual values

    return loss

# calling the function
loss = binary_cross_entropy(y_true, y_pred)

# printing the result
print(f"Loss = {loss:.4f}")

# if y = 1 - y_true , then loss = -1 * loss  
# it is used to calculate the loss for the other class
print("Loss = ",binary_cross_entropy(1 - y_true, y_pred))

