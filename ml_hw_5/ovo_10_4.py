# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from liblinear.liblinearutil import train, predict, svm_read_problem
import urllib.request
import os
import bz2
from tqdm import tqdm  # 進度條顯示

def download_and_extract(url, bz2_filename, extracted_filename):
    """
    下載並解壓縮 .bz2 檔案。
    """
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
    """
    過濾出指定類別並映射標籤為 -1 和 +1。
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
    將稀疏特徵字典轉換為密集的 NumPy 陣列，並添加偏置項。
    """
    dense_x = []
    for features in x:
        dense = [1.0]  # 偏置項 x0 = 1
        for i in range(1, num_features + 1):
            dense.append(features.get(i, 0.0))
        dense_x.append(dense)
    return dense_x

def zero_one_error(y_true, y_pred):
    """
    計算 0/1 誤差。
    """
    return np.mean(y_true != y_pred)


def count_non_zero_weights(model):
    """
    Counts the number of non-zero weights in the model.

    Parameters:
    - model: The model trained by LIBLINEAR.

    Returns:
    - Number of non-zero weights.
    """
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
    # Load training data
    print("Loading training data...")
    y_train_full, x_train_full = svm_read_problem('mnist.scale')

    # Load test data
    print("Loading test data...")
    y_test_full, x_test_full = svm_read_problem('mnist.scale.t')

    # 過濾出類別 2 和 6 並映射標籤
    print("過濾類別並映射標籤...")
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
    
    # 將稀疏特徵轉換為密集格式並添加偏置項
    print("將特徵轉換為密集格式並添加偏置項...")
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
    
    # 進行實驗
    print(f"開始進行 {experiment_time} 次實驗...")
    for exp in tqdm(range(experiment_time), desc="進行實驗"):
        # 設定隨機種子並打亂訓練資料
        np.random.seed(exp)
        indices = np.random.permutation(len(y_train))
        y_train_shuffled = y_train[indices]
        X_train_shuffled = [X_train_dense[i] for i in indices]
        
        # 初始化每個 lambda 的 Ein 和模型
        Ein_per_lambda = []
        models_per_lambda = []
        
        # 對每個 C 值訓練模型
        for C in C_values:
            y_train_liblinear = y_train_shuffled.tolist()
            X_train_liblinear = [dict(enumerate(row, start=1)) for row in X_train_shuffled]
 
            model = train(y_train_liblinear, X_train_liblinear, f"-s 6 -c {C} -q")
            
            
            p_labels, p_acc, p_vals = predict(y_train_liblinear, X_train_liblinear, model, '-q')
            
          
            E_in = zero_one_error(y_train_shuffled, np.array(p_labels))
            Ein_per_lambda.append(E_in)
            models_per_lambda.append(model)
        
        # 選擇最佳 lambda*，即使 Ein 最小的 lambda，如果有多個，選擇最大的 lambda
        min_Ein = min(Ein_per_lambda)
        candidate_indices = [i for i, Ein in enumerate(Ein_per_lambda) if Ein == min_Ein]
        best_lambda_idx = max(candidate_indices)  # 選擇最大的 lambda
        best_lambda = lambda_values[best_lambda_idx]
        best_C = C_values[best_lambda_idx]
        best_model = models_per_lambda[best_lambda_idx]
        
        y_test_liblinear = y_test.tolist()
        X_test_liblinear = [dict(enumerate(row, start=1)) for row in X_test_dense]
        p_labels_test, p_acc_test, p_vals_test = predict(y_test_liblinear, X_test_liblinear, best_model, '-q')
        
        # 計算 0/1 誤差
        E_out = zero_one_error(y_test, np.array(p_labels_test))
        E_out_hist.append(E_out)
        
        # 計算非零權重的數量
        non_zero = count_non_zero_weights(best_model)
        non_zero_counts.append(non_zero)
    
    # 繪製 E_out 直方圖
    print("繪製 E_out 直方圖並保存為 'E_out_histogram.png'...")
    plt.figure(figsize=(10, 6))
    plt.hist(E_out_hist, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Out-of-sample Error (E_out)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Out-of-sample Errors over 1126 Experiments')
    plt.grid(True)
    plt.savefig('E_out_histogram.png')
    plt.close()
    
    # 繪製非零權重數量直方圖
    print("繪製非零權重數量直方圖並保存為 'non_zero_weights_histogram.png'...")
    plt.figure(figsize=(10, 6))
    plt.hist(non_zero_counts, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Non-zero Weights')
    plt.ylabel('Frequency')
    plt.title('Histogram of Non-zero Weights in Models over 1126 Experiments')
    plt.grid(True)
    plt.savefig('non_zero_weights_histogram.png')
    plt.close()
    
    # 計算並顯示平均 E_out 和平均非零權重數量
    average_E_out = np.mean(E_out_hist)
    average_non_zero = np.mean(non_zero_counts)
    
    print(f"Average Out-of-sample Error (E_out): {average_E_out:.4f}")
    print(f"Average Number of Non-zero Weights: {average_non_zero:.2f}")

if __name__ == "__main__":
    main()
