## Neural Network

def step_function(x):
   return np.where(x >= 0, 1, 0)

​

def train(X, y, learning_rate, max_epochs):
    num_datapoints, num_features = X.shape

    weights = np.zeros(shape=(num_features, 1))
    bias = 0

    for i in range(max_epochs):
        # Calculate linear combination then pass it through the

        # step function
        y_predict = step_function(np.dot(X, weights) + bias)

        # Compute the update values
        update_weight =  learning_rate * np.dot(X.T, (y - y_predict))
        update_bias = learning_rate *  np.sum(y - y_predict)

        # Update the weights and bias
        weights +=  update_weight
        bias += update_bias

    return weights, bias