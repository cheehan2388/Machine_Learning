# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from liblinear.liblinearutil import train, predict, svm_read_problem
import urllib.request
import os
import bz2
from tqdm import tqdm 

def download_and_extract(url, bz2_filename, extracted_filename):

    if not os.path.isfile(extracted_filename):
        if not os.path.isfile(bz2_filename):
            print(f"正在下載 {bz2_filename} 從 {url}...")
            urllib.request.urlretrieve(url, bz2_filename)
            print(f"已下載 {bz2_filename}.")
        else:
            print(f"{bz2_filename} 已存在。")
        print(f"正在解壓縮 {bz2_filename}...")
        with bz2.BZ2File(bz2_filename) as fr, open(extracted_filename, 'wb') as fw:
            fw.write(fr.read())
        print(f"已解壓縮到 {extracted_filename}.")
    else:
        print(f"{extracted_filename} 已存在。跳過下載和解壓縮。")

def filter_and_map_classes(y, x, class1=2, class2=6):

    filtered_y = []
    filtered_x = []
    for label, features in zip(y, x):
        if label == class1 or label == class2:
            mapped_label = -1 if label == class1 else 1
            filtered_y.append(mapped_label)
            filtered_x.append(features)
    return np.array(filtered_y), filtered_x

def convert_to_dense(x, num_features):

    dense_x = []
    for features in x:
        dense = [1.0]  # 偏置項 x0 = 1
        for i in range(1, num_features + 1):
            dense.append(features.get(i, 0.0))
        dense_x.append(dense)
    return dense_x

def zero_one_error(y_true, y_pred):

    return np.mean(y_true != y_pred)


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
def main():

    y_train_full, x_train_full = svm_read_problem('mnist.scale')

    
    y_test_full, x_test_full = svm_read_problem('mnist.scale.t')

 
    y_train, x_train = filter_and_map_classes(y_train_full, x_train_full, class1=2, class2=6)
    y_test, x_test = filter_and_map_classes(y_test_full, x_test_full, class1=2, class2=6)
    
    # 確定特徵數量
    max_feature_index = 0
    for sample in x_train + x_test:
        if sample:
            current_max = max(sample.keys())
            if current_max > max_feature_index:
                max_feature_index = current_max
    num_features = max_feature_index
    print(f"特徵數量（不含偏置項）: {num_features}")
    
    X_train_dense = convert_to_dense(x_train, num_features)
    X_test_dense = convert_to_dense(x_test, num_features)
    
    # 定義 lambda 值和對應的 C 值
    log_lambda_values = [-2, -1, 0, 1, 2, 3]
    lambda_values = [10**l for l in log_lambda_values]
    C_values = [1/lambda_val for lambda_val in lambda_values]
    print("Lambda 值:", lambda_values)
    print("對應的 C 值:", C_values)
    
    # 實驗參數
    experiment_time = 1126  # 總實驗次數
    
    # 初始化結果儲存列表
    E_out_hist = []
    non_zero_counts = []
    

    for exp in tqdm(range(experiment_time), desc="進行實驗"):
        
        np.random.seed(exp)
        indices = np.random.permutation(len(y_train))
        y_train_shuffled = y_train[indices]
        X_train_shuffled = [X_train_dense[i] for i in indices]
        
       
        Ein_per_lambda = []
        models_per_lambda = []
        
        
        for C in C_values:
            y_train_liblinear = y_train_shuffled.tolist()
            X_train_liblinear = [dict(enumerate(row, start=1)) for row in X_train_shuffled]
 
            model = train(y_train_liblinear, X_train_liblinear, f"-s 6 -c {C} -q")
            
            
            p_labels, p_acc, p_vals = predict(y_train_liblinear, X_train_liblinear, model, '-q')
            
          
            E_in = zero_one_error(y_train_shuffled, np.array(p_labels))
            Ein_per_lambda.append(E_in)
            models_per_lambda.append(model)
        
       
        min_Ein = min(Ein_per_lambda)
        candidate_indices = [i for i, Ein in enumerate(Ein_per_lambda) if Ein == min_Ein]
        best_lambda_idx = max(candidate_indices) 
        best_lambda = lambda_values[best_lambda_idx]
        best_C = C_values[best_lambda_idx]
        best_model = models_per_lambda[best_lambda_idx]
        
        y_test_liblinear = y_test.tolist()
        X_test_liblinear = [dict(enumerate(row, start=1)) for row in X_test_dense]
        p_labels_test, p_acc_test, p_vals_test = predict(y_test_liblinear, X_test_liblinear, best_model, '-q')
    
        E_out = zero_one_error(y_test, np.array(p_labels_test))
        E_out_hist.append(E_out)
        
      
        non_zero = count_non_zero_weights(best_model)
        non_zero_counts.append(non_zero)
    
   
    print("繪製 E_out 直方圖並保存為 'E_out_histogram.png'...")
    plt.figure(figsize=(10, 6))
    plt.hist(E_out_hist, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Out-of-sample Error (E_out)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Out-of-sample Errors over 1126 Experiments')
    plt.grid(True)
    plt.savefig('E_out_histogram.png')
    plt.close()
    
  
    plt.figure(figsize=(10, 6))
    plt.hist(non_zero_counts, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Non-zero Weights')
    plt.ylabel('Frequency')
    plt.title('Histogram of Non-zero Weights in Models over 1126 Experiments')
    plt.grid(True)
    plt.savefig('non_zero_weights_histogram.png')
    plt.close()
  
    average_E_out = np.mean(E_out_hist)
    average_non_zero = np.mean(non_zero_counts)
    
    print(f"Average Out-of-sample Error (E_out): {average_E_out:.4f}")
    print(f"Average Number of Non-zero Weights: {average_non_zero:.2f}")

if __name__ == "__main__":
    main()
