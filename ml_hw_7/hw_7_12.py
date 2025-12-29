# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
# Set Matplotlib to non-interactive backend to avoid GUI errors
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from libsvm.svmutil import svm_read_problem
import urllib.request
import os
from tqdm import tqdm  # For progress bar

def download_dataset(url, filename):
    """
    Download dataset from the specified URL if not already present.
    """
    if not os.path.exists(filename):
        print(f"Downloading {filename} from {url}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}.")
    else:
        print(f"{filename} already exists. Skipping download.")

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
    - U_list: list of cumulative weighted errors up to each iteration
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
    U_list = []
    
    # Initialize cumulative sum of epsilon
    cumulative_epsilon = 0.0
    
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
        
        # Make predictions on training data
        predictions = stump['polarity'] * np.sign(X_train[:, stump['feature']] - stump['threshold'])
        predictions[predictions == 0] = 1  # Handle zero as positive class
        
        # Update weights
        weights *= np.exp(-alpha * y_train * predictions)
        weights /= np.sum(weights)  # Normalize
        
        # Compute Ein(Gt)
        # Aggregate all weak learners up to current t
        aggregated_predictions_train = np.zeros(n_train)
        for i in range(t):
            agg_pred = alphas[i] * (weak_learners[i]['polarity'] * np.sign(X_train[:, weak_learners[i]['feature']] - weak_learners[i]['threshold']))
            aggregated_predictions_train += agg_pred
        Gt_train = np.sign(aggregated_predictions_train)
        Gt_train[Gt_train == 0] = 1  # Handle zero as positive class
        Ein_t = np.mean(Gt_train != y_train)
        Ein_list.append(Ein_t)
        
        # Compute weighted error epsilon_t and update cumulative U
        epsilon_t = error
        cumulative_epsilon += epsilon_t
        U_list.append(cumulative_epsilon)
        
        # Compute Eout(Gt)
        aggregated_predictions_test = np.zeros(n_test)
        for i in range(t):
            agg_pred_test = alphas[i] * (weak_learners[i]['polarity'] * np.sign(X_test[:, weak_learners[i]['feature']] - weak_learners[i]['threshold']))
            aggregated_predictions_test += agg_pred_test
        Gt_test = np.sign(aggregated_predictions_test)
        Gt_test[Gt_test == 0] = 1  # Handle zero as positive class
        Eout_t = np.mean(Gt_test != y_test)
        Eout_list.append(Eout_t)
        
    return Ein_list, Eout_list, U_list

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
 
    print("Loading training data...")
    y_train, x_train = svm_read_problem('madelon_train.txt')
    print("Loading testing data...")
    y_test, x_test = svm_read_problem('madelon_test.txt')
    # Convert labels from {0,1} to {-1,+1} if necessary
    y_train = np.array([1 if label > 0 else -1 for label in y_train])
    y_test = np.array([1 if label > 0 else -1 for label in y_test])
    
    # Preprocess data to numpy arrays
    print("Preprocessing training data...")
    X_train = preprocess_data(x_train)
    print("Preprocessing testing data...")
    X_test = preprocess_data(x_test)
    
    # Run AdaBoost-Stump
    print("Running AdaBoost-Stump algorithm...")
    Ein, Eout, U = adaboost_stump(X_train, y_train, X_test, y_test, T=500)
    
    # Plotting Ein(Gt) and U_t
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, 501), Ein, label='Ein(Gt)', color='blue')
    plt.plot(range(1, 501), U, label='Ut (Cumulative Weighted Errors)', color='green')
    plt.xlabel('Iteration t')
    plt.ylabel('Error')
    plt.title('Ein(Gt) and Ut over 500 Iterations of AdaBoost-Stump')
    plt.legend()
    plt.grid(True)
    plt.savefig('Ein_Ut_plot.png')
    plt.close()
    
    # Print final average errors
    print(f"Final Average Ein(Gt): {Ein[-1]:.4f}")
    print(f"Final Average Eout(Gt): {Eout[-1]:.4f}")
    
    # Optionally, save Ein and Eout for further analysis
    np.save('Ein.npy', Ein)
    np.save('Eout.npy', Eout)
    np.save('U.npy', U)

if __name__ == "__main__":
    main()
