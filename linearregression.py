## Linear Regression

def train_linear_reg_gd(X, y, learning_rate, max_epochs):
        num_datapoints, num_features = X.shape

​

        # set weights and bias to 0
        weights = np.zeros(shape=(num_features, 1))
        bias = 0

        for i in range(max_epochs):
            # Calculate simple linear combination y = mx + c or y = X * w + b
            y_predict = np.dot(X, weights) + bias # O(r*c)

​

            # Use mean squared error to calculate the loss and then
            # get the average over all datapoints to get the cost
            cost = (1 / num_datapoints) * np.sum((y_predict - y)**2) # O(r)

            print(cost)

​

            # Calculate gradients
            # 1st - gradient with respect to weights
            grad_weights = (1 / num_datapoints) * np.dot(X.T, (y_predict - y)) # O(c⋅r)
            # 2st - gradient with respect to bias
            grad_bias = (1 / num_datapoints) * np.sum((y_predict - y)) # O(r)
            
            # update weights and bias
            weights -= learning_rate * grad_weights
            bias -= learning_rate * grad_bias

​

        return weights, bias