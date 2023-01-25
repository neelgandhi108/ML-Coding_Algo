import numpy as np

def predict_knn(X_train, y_train, x_test, k):
    distances = np.array([np.linalg.norm(x_test - x, axis=1) for x in X_train])
    nearest_indices = np.argsort(distances, axis=0)[:k]
    nearest_labels = y_train[nearest_indices]
    return np.argmax(np.bincount(nearest_labels))
