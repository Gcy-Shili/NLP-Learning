import numpy as np
from sklearn.model_selection import train_test_split

class LinearRegression:
    def __init__(self):
        self.w = None
        self.b = None
    
    def gen_data(self, w_true, b_true, n=300, noise=0.15):
        X = np.random.rand(n, 1)  # nums and fea_dims, 1 here
        y = X.squeeze() * w_true + b_true + np.random.normal(0, noise, n)
        return X, y

    def mse(self, y_pred, y):
        return np.mean((y_pred - y) ** 2) / 2

    def compute_grad(self, nums, loss, X):
        gw = (1 / nums) * np.dot(loss, X)
        gb = (1 / nums) * np.sum(loss)
        return gw, gb

    def train(self, X, y, lr=0.01, epochs=1000):
        nums, feas = X.shape
        self.w = np.zeros(feas)  # a commom initialization, 0 here
        self.b = 0
        for epoch in range(epochs):
            y_pred = X.dot(self.w) + self.b  # compute output
            loss = y_pred - y
            loss_f = self.mse(y_pred, y)  # compute loss
            gw, gb = self.compute_grad(nums, loss, X)  # compute gradients
            self.w -= lr * gw
            self.b -= lr * gb
            if epoch % 500 == 0:
                print(f"Epoch {epoch}: loss={loss_f: .4f}")

    def predict(self, X):
        return X.dot(self.w) + self.b
            
    def validate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mse = self.mse(y_pred, y_test)
        print(f"Validation MSE: {mse:.4f}")
        


if __name__ == "__main__":
    lr = LinearRegression()
    X, y = lr.gen_data(w_true=2.5, b_true=1.6)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr.train(X_train, y_train, lr=0.01, epochs=8000)
    lr.validate(X_test, y_test)
    print()
    print(f"trained w: {lr.w[0]:.4f}")
    print(f"trained b: {lr.b:.4f}")


"""
Epoch 0: loss= 4.3644
Epoch 500: loss= 0.0426
Epoch 1000: loss= 0.0265
Epoch 1500: loss= 0.0183
Epoch 2000: loss= 0.0141
Epoch 2500: loss= 0.0120
......
Epoch 5500: loss= 0.0099
Epoch 6000: loss= 0.0098
Epoch 6500: loss= 0.0098
Epoch 7000: loss= 0.0098
Epoch 7500: loss= 0.0098
Validation MSE: 0.0111

trained w: 2.4480
trained b: 1.6177

true w: 2.5
true b: 1.6
"""