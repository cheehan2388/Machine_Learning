import numpy as np
import matplotlib
# Use non-interactive backend to avoid GUI errors
import matplotlib.pyplot as plt
from liblinear.liblinearutil import train, predict, svm_read_problem
import os
from tqdm import tqdm  # For displaying progress bars

# Function to calculate 0/1 error
def zero_one_error(y_true, y_pred):
    return np.mean(np.array(y_true) != np.array(y_pred))

# Function to count the number of non-zero weights in the model
def count_non_zero_weights(model):

    if hasattr(model, 'w'):
        # Convert ctypes array to NumPy array
        num_features = model.nr_feature
        bias_term = int(model.bias >= 0)
        weights = np.ctypeslib.as_array(model.w, shape=(num_features + bias_term,))
        # Count non-zero weights
        non_zero = np.sum(weights != 0)
        return non_zero
    else:
        return 0

# Load training data
print("Loading training data...")
y_train_full, x_train_full = svm_read_problem('mnist.scale')

# Load test data
print("Loading test data...")
y_test_full, x_test_full = svm_read_problem('mnist.scale.t')

# Function to filter and map classes
def filter_and_map_classes(y, x, class1=2, class2=6):
    """
    Filters the dataset to include only the specified classes and maps them to +1 and -1.

    Parameters:
    - y: Original labels.
    - x: Original features.
    - class1: The first class to include (mapped to +1).
    - class2: The second class to include (mapped to -1).

    Returns:
    - filtered_y: Filtered and mapped labels.
    - filtered_x: Filtered features.
    """
    filtered_y = []
    filtered_x = []
    for label, features in zip(y, x):
        if label == class1 or label == class2:
            # Map class1 to +1, class2 to -1
            mapped_label = 1 if label == class1 else -1
            filtered_y.append(mapped_label)
            filtered_x.append(features)
    return filtered_y, filtered_x

# Filter and map training and test data
print("Filtering and mapping training data...")
y_train, x_train = filter_and_map_classes(y_train_full, x_train_full, class1=2, class2=6)
print("Filtering and mapping test data...")
y_test, x_test = filter_and_map_classes(y_test_full, x_test_full, class1=2, class2=6)

# Number of training samples
N = len(y_train)

# Determine the number of features
max_feature_index = 0
for sample in x_train + x_test:
    if sample:
        current_max = max(sample.keys())
        if current_max > max_feature_index:
            max_feature_index = current_max

print(f"Number of features (excluding bias term): {max_feature_index}")

# Define log10(lambda) values and corresponding C values
log_lambda_values = [-2, -1, 0, 1, 2, 3]
lambda_values = [10**l for l in log_lambda_values]
C_values = [1 / (N * lambda_val) for lambda_val in lambda_values]

print("Lambda values:", lambda_values)
print("Corresponding C values:", C_values)

# Initialize lists to store Ein for each lambda
Ein_per_lambda = []
models_per_lambda = []

# Compute Ein for different lambda values
print("Computing Ein for different lambda values...")
for C in C_values:
    # Train the model using -s 6 option (L1-regularized logistic regression) with corresponding -c C value
    model = train(y_train, x_train, f"-s 6 -c {C} -B 1 -q")
    # Predict on training data to compute Ein
    p_labels, p_acc, p_vals = predict(y_train, x_train, model, '-q')
    # Calculate 0/1 error rate
    E_in = zero_one_error(y_train, p_labels)
    Ein_per_lambda.append(E_in)
    models_per_lambda.append(model)

# Select the lambda* that minimizes Ein, choosing the largest lambda in case of a tie
min_Ein = min(Ein_per_lambda)
candidate_indices = [i for i, Ein in enumerate(Ein_per_lambda) if Ein == min_Ein]
best_lambda_idx = max(candidate_indices)  # Choose the largest lambda (smallest C)
lambda_star = lambda_values[best_lambda_idx]
C_star = C_values[best_lambda_idx]
print(f"Selected λ* = {lambda_star}, corresponding C = {C_star}")

# Initialize lists to store E_out and non-zero weight counts for each experiment
E_out_hist = []
non_zero_counts = []

# Start experiments
print("Starting experiments...")
for seed in tqdm(range(1, 1127), desc="Experiments"):
    # Train the model with the best C value
    # Note: The '-r' option is removed because it is not supported by the liblinear Python wrapper
    model = train(y_train, x_train, f"-s 6 -c {C_star} -B 1 -q")
    # Predict on test data
    p_labels_test, p_acc_test, p_vals_test = predict(y_test, x_test, model, '-q')
    # Calculate E_out
    E_out = zero_one_error(y_test, p_labels_test)
    E_out_hist.append(E_out)
    # Count the number of non-zero weights
    non_zero = count_non_zero_weights(model)
    non_zero_counts.append(non_zero)

# Plot histogram of E_out
print("Plotting E_out histogram...")
plt.figure(figsize=(10, 6))
plt.hist(E_out_hist, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Out-of-sample Error (E_out)')
plt.ylabel('Frequency')
plt.title('Histogram of Out-of-sample Errors over 1126 Experiments')
plt.grid(True)
plt.savefig('E_out_histogram.png')  # Save the figure
plt.close()

# Plot histogram of non-zero weight counts
print("Plotting non-zero weight counts histogram...")
plt.figure(figsize=(10, 6))
plt.hist(non_zero_counts, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Number of Non-zero Weights')
plt.ylabel('Frequency')
plt.title('Histogram of Non-zero Weights in Models over 1126 Experiments')
plt.grid(True)
plt.savefig('non_zero_weights_histogram.png')  # Save the figure
plt.close()

print("Experiments completed.")
