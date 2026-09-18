
import numpy as np 
# 1 . Create a simple binary classification problem 
rng = np.random.default_rng(42)
X_train = rng.normal(size=(100, 2))
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int) 

# 2 . Initialize weights and bias 
w , b = np.array([0.5, -0.5]) , 0.0 

# 3 . Weighted sum (linear transformation)
z = X_train @ w + b 

# 4 . Sigmoid activation 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

y_pred = sigmoid(z)


# 5 . Binary Cross-Entropy loss
def binary_cross_entropy(_train,y_pred):
    eps = 1e-8
    y_pred = np.clip(y_pred,eps,1 - eps)

    return -np.mean(
        y_train * np.log(y_pred)
        + (1 - y) * np.log( 1 - y_pred)
    )

loss = binary_cross_entropy(y_train,y_pred)

# 6. Print the result
print(f"Loss: {loss:.4f}")
