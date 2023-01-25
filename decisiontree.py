##Decision Tree Classification
def calculate_cost(left_subset, right_subset, classes):
    total_data_points = len(left_subset) + len(right_subset)

    gini_left = gini_right = 0


    if len(left_subset) > 0:
        sum_of_prob = 0
        for c in classes:
            # find the num of data points which are in class c
            num_in_c = len([l for l in left_subset if l[1] == c])
            # now we can get the probability of class c in this subset
            prob = num_in_c / len(left_subset)
            # sum this value squared as per the equation
            sum_of_prob += prob ** 2
        gini_left = 1 - sum_of_prob

​

    # we do the same for the right subset
    if len(right_subset) > 0:
        sum_of_prob = 0
        for c in classes:
            num_in_c = len([l for l in right_subset if l[1] == c])
            prob = num_in_c / len(right_subset)
            sum_of_prob += prob ** 2
        gini_right = 1 - sum_of_prob

​

    # we calculate the cost 
    cost = (len(left_subset) / total_data_points) * gini_left + \
           (len(right_subset) / total_data_points) * gini_right

    return cost

​

def decision_tree_best_split(X, y):
    best_feature_val = best_feature_idx = best_cost_val = float('inf')
    best_data_point = None
    classes = range(len(set(y)))  # get the number of classes

​

    # Iterate over each possible feature value
    for feature in range(len(X[0])):
        for data_point in range(len(X)):
            feature_threshold_val = X[data_point][feature]
            # based on our threshold value, we determine
            # which data points go into the left or right subset
            left_subset = []
            right_subset = []
            for instance, label in zip(X, y):
                if instance[feature] < feature_threshold_val:
                    left_subset.append((instance, label))
                else:
                    right_subset.append((instance, label))
            # calculate the cost
            curr_cost = calculate_cost(left_subset, right_subset, classes)

​

            # update our best feature if the current cost 
            # is less than our best so far
            if curr_cost < best_cost_val:
                best_feature_idx = feature
                best_feature_val = feature_threshold_val
                best_cost_val = curr_cost
                best_data_point = data_point

​

    return (best_feature_val, best_feature_idx, best_data_point)
    
    
 ## Regression
 
 
 def calculate_cost(left_subset, right_subset):
    total_data_points = len(left_subset) + len(right_subset)

    mse_left = mse_right = 0


    if len(left_subset) > 0:
        # find the mean of the values in the left subset, this will

        # be our predicted y value
        y_pred = sum([v[1] for v in left_subset]) / len(left_subset)

        # calculate the difference between the predicted and actual value

        difference = [(y[1] - y_pred) ** 2 for y in left_subset]

        # calculate MSE

        mse_left = (1 / len(left_subset)) * sum(difference)

 

    # we do the same for the right subset

    if len(right_subset) > 0:
        # find the mean of the values in the left subset, this will

        # be our predicted y value
        y_pred = sum([v[1] for v in right_subset]) / len(right_subset)

        # calculate the difference between the predicted and actual value

        difference = [(y[1] - y_pred) ** 2 for y in right_subset]

        # calculate MSE

        mse_right = (1 / len(left_subset)) * sum(difference)

​

    # we calculate the cost 
    cost = (len(left_subset) / total_data_points) * mse_left + \
           (len(right_subset) / total_data_points) * mse_right

    return cost