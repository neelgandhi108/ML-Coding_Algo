## Logistic Regression

def train_log_regression(X, y, learning_rate, max_epochs):
        num_datapoints, num_features = X.shape

​

        # set weights and bias to 0
        weights = np.zeros(shape=(num_features, 1))
        bias = 0

        for i in range(max_epochs):
            # Calculate logistic regression based on the above equation

                       # We can create a simple sigmoid fn if needed: 1 / (1 + np.exp(-x))
            y_predict = sigmoid(np.dot(X, weights) + bias) # O(r⋅c)

​

            # Use log loss to calculate the loss and then
            # get the average over all datapoints to get the cost
            cost = (- 1 / num_datapoints) * np.sum(y * np.log(y_predict) + (1 - y) * (np.log(1 - y_predict))) # O(r)

​

            # Calculate gradients
            # 1st - gradient with respect to weights
            grad_weight = (1 / num_datapoints) * np.dot(X.T, (y_predict - y)) # O(c⋅r)
            # 2st - gradient with respect to bias
            grad_bias = (1 / num_datapoints) * np.sum((y_predict - y)) # O(r)
            
            # update weights and bias
            weights -= learning_rate * grad_weight
            bias -= learning_rate * grad_bias

​

        return weights, bias