# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from libsvm.svmutil import svm_read_problem
from tqdm import tqdm  # For progress bar


def filter_and_map_classes(y, x, class1=2, class2=6):
    """
    Filter out specified classes and map labels to -1 and +1.
    """
    filtered_y = []
    filtered_x = []
    for label, features in zip(y, x):
        if label == class1 or label == class2:
            mapped_label = -1 if label == class1 else 1
            filtered_y.append(mapped_label)
            filtered_x.append(features)
    return np.array(filtered_y), filtered_x

def convert_to_dense(x, num_features):
    """
    Convert sparse feature dictionaries to dense NumPy arrays and add bias term.
    """
    dense_x = []
    for features in x:
        dense = [1.0]  # Bias term x0 = 1
        for i in range(1, num_features + 1):
            dense.append(features.get(i, 0.0))
        dense_x.append(dense)
    return np.array(dense_x)

def zero_one_error(y_true, y_pred):
    """
    Calculate 0/1 error.
    """
    return np.mean(y_true != y_pred)

def count_non_zero_weights(model):
    """
    Count the number of non-zero weights in the model.
    """
    if hasattr(model, 'w'):
        return np.sum(np.array(model.w) != 0)
    else:
        return 0

def decision_stump(X, y, weights):
    """
    Find the best decision stump for the given data and weights.
    
    Parameters:
    - X: numpy array of shape (n_samples, n_features)
    - y: numpy array of shape (n_samples,)
    - weights: numpy array of shape (n_samples,)
    
    Returns:
    - best_stump: dictionary with keys 'feature', 'threshold', 'polarity'
    - error: weighted error of the best stump
    """
    n_samples, n_features = X.shape
    min_error = float('inf')
    best_stump = {}
    
    for feature in range(n_features):
        feature_values = X[:, feature]
        unique_values = np.unique(feature_values)
        if len(unique_values) == 1:
            continue  # Skip if all values are the same
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2  # Midpoints
        
        for threshold in thresholds:
            for polarity in [1, -1]:
                predictions = polarity * np.sign(X[:, feature] - threshold)
                predictions[predictions == 0] = 1  # Handle zero as positive class
                
                misclassified = predictions != y
                weighted_error = np.sum(weights * misclassified)
                
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_stump = {
                        'feature': feature,
                        'threshold': threshold,
                        'polarity': polarity
                    }
                    
    return best_stump, min_error

def adaboost_stump(X_train, y_train, X_test, y_test, T=500):
    """
    Implement AdaBoost with decision stumps.
    
    Parameters:
    - X_train: numpy array of shape (n_samples_train, n_features)
    - y_train: numpy array of shape (n_samples_train,)
    - X_test: numpy array of shape (n_samples_test, n_features)
    - y_test: numpy array of shape (n_samples_test,)
    - T: number of iterations
    
    Returns:
    - Ein_list: list of in-sample errors at each iteration
    - Eout_list: list of out-of-sample errors at each iteration
    - epsilons: list of weighted errors at each iteration
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    
    # Initialize weights
    weights = np.ones(n_train) / n_train
    
    # To store weak learners and their alphas
    weak_learners = []
    alphas = []
    
    # To store errors
    Ein_list = []
    Eout_list = []
    epsilons = []
    
    # Initialize cumulative predictions for Gt
    cumulative_prediction_train = np.zeros(n_train)
    cumulative_prediction_test = np.zeros(n_test)
    
    for t in tqdm(range(1, T + 1), desc="AdaBoost Iterations"):
        # Find the best decision stump
        stump, error = decision_stump(X_train, y_train, weights)
        
        # Avoid division by zero
        error = max(error, 1e-10)
        
        # Compute alpha
        alpha = 0.5 * np.log((1 - error) / error)
        
        # Store the weak learner and alpha
        weak_learners.append(stump)
        alphas.append(alpha)
        
        # Make predictions
        predictions = stump['polarity'] * np.sign(X_train[:, stump['feature']] - stump['threshold'])
        predictions[predictions == 0] = 1  # Handle zero as positive class
        
        # Update weights
        weights *= np.exp(-alpha * y_train * predictions)
        weights /= np.sum(weights)  # Normalize
        
        # Update cumulative predictions for Gt on training data
        cumulative_prediction_train += alpha * predictions
        Gt_train = np.sign(cumulative_prediction_train)
        Gt_train[Gt_train == 0] = 1  # Handle zero as positive class
        Ein_t = zero_one_error(y_train, Gt_train)
        Ein_list.append(Ein_t)
        
        # Compute weighted error epsilon_t
        epsilons.append(error)
        
        # Make predictions on test data
        predictions_test = stump['polarity'] * np.sign(X_test[:, stump['feature']] - stump['threshold'])
        predictions_test[predictions_test == 0] = 1  # Handle zero as positive class
        cumulative_prediction_test += alpha * predictions_test
        Gt_test = np.sign(cumulative_prediction_test)
        Gt_test[Gt_test == 0] = 1  # Handle zero as positive class
        Eout_t = zero_one_error(y_test, Gt_test)
        Eout_list.append(Eout_t)
        
    return Ein_list, Eout_list, epsilons

def preprocess_data(x):
    """
    Convert list of dictionaries to numpy array.
    
    Parameters:
    - x: list of dictionaries
    
    Returns:
    - X: numpy array of shape (n_samples, n_features)
    """
    n_samples = len(x)
    n_features = max([max(sample.keys()) if sample else 0 for sample in x])
    X = np.zeros((n_samples, n_features))
    for i, sample in enumerate(x):
        for key, value in sample.items():
            X[i, key - 1] = value  # libsvm format indices start at 1
    return X

def main():
   # Load data using libsvm format
    print("Loading training data...")
    y_train, x_train = svm_read_problem('madelon_train.txt')
    print("Loading testing data...")
    y_test, x_test = svm_read_problem('madelon_test.txt')
    # Convert labels from {0,1} to {-1,+1} if necessary
    y_train = np.array([1 if label > 0 else -1 for label in y_train])
    y_test = np.array([1 if label > 0 else -1 for label in y_test])
    
    # Preprocess data to numpy arrays
    print("Preprocessing training data...")
    X_train = convert_to_dense(x_train, num_features=max([max(sample.keys()) if sample else 0 for sample in x_train]))
    print("Preprocessing testing data...")
    X_test = convert_to_dense(x_test, num_features=max([max(sample.keys()) if sample else 0 for sample in x_test]))
    
    # Run AdaBoost-Stump
    print("Running AdaBoost-Stump algorithm...")
    Ein, Eout, epsilons = adaboost_stump(X_train, y_train, X_test, y_test, T=500)
    
    # Plotting Ein(Gt) and Eout(Gt)
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, 501), Ein, label='Ein(Gt)', color='blue')
    plt.plot(range(1, 501), Eout, label='Eout(Gt)', color='red')
    plt.xlabel('Iteration t')
    plt.ylabel('Error')
    plt.title('Ein(Gt) and Eout(Gt) over 500 Iterations of AdaBoost-Stump')
    plt.legend()
    plt.grid(True)
    plt.savefig('Ein_Eout_plot.png')
    plt.close()
    
    # Print final average errors
    print(f"Final Average Ein(Gt): {Ein[-1]:.4f}")
    print(f"Final Average Eout(Gt): {Eout[-1]:.4f}")
    
    # Optionally, save Ein and Eout for further analysis
    np.save('Ein.npy', Ein)
    np.save('Eout.npy', Eout)
    np.save('Epsilons.npy', epsilons)

if __name__ == "__main__":
    main()
