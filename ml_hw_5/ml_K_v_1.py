# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from liblinear.liblinearutil import train, predict, svm_read_problem
import urllib.request
import os
import bz2
from tqdm import tqdm  # 進度條顯示
from sklearn.model_selection import KFold  # 3-fold Cross-Validation

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
    y_train_full, x_train_full = svm_read_problem('mnist.scale')

    # Load test data
    
    y_test_full, x_test_full = svm_read_problem('mnist.scale.t')

    # 過濾出類別 2 和 6 並映射標籤

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
        
        kf = KFold(n_splits=3, shuffle=True, random_state=exp)

        Eval_per_lambda = np.zeros(len(lambda_values))
        

        for fold_idx, (train_index, val_index) in enumerate(kf.split(X_train_shuffled)):
            X_sub_train = [X_train_shuffled[i] for i in train_index]
            y_sub_train = y_train_shuffled[train_index]
            X_val = [X_train_shuffled[i] for i in val_index]
            y_val = y_train_shuffled[val_index]
            
            for i, C in enumerate(C_values):
                try:
                    # 將子訓練集轉換為 LIBLINEAR 格式
                    y_sub_train_liblinear = y_sub_train.tolist()
                    X_sub_train_liblinear = [dict(enumerate(row, start=1)) for row in X_sub_train]
                    
                    # 訓練模型，使用 -s 6 (L1-regularized logistic regression) 和 -c C
                    model = train(y_sub_train_liblinear, X_sub_train_liblinear, f"-s 6 -c {C} -q")
                    
                    # 在驗證集上預測以計算 Eval
                    y_val_liblinear = y_val.tolist()
                    X_val_liblinear = [dict(enumerate(row, start=1)) for row in X_val]
                    p_labels_val, p_acc_val, p_vals_val = predict(y_val_liblinear, X_val_liblinear, model, '-q')
                    
                    # 計算 0/1 誤差並累積
                    Eval = zero_one_error(y_val, np.array(p_labels_val))
                    Eval_per_lambda[i] += Eval
                except Exception as e:
                    print(f"實驗 {exp} 的折疊 {fold_idx} 中 lambda {lambda_values[i]} 出現錯誤: {e}")
                    Eval_per_lambda[i] += np.inf  # 將出現錯誤的 lambda 設為無限大誤差
        
        # 計算平均驗證誤差
        Avg_Eval_per_lambda = Eval_per_lambda / 3  # 3-fold
        
        # 選擇最佳 lambda*
        min_Eval = np.min(Avg_Eval_per_lambda)
        candidate_indices = np.where(Avg_Eval_per_lambda == min_Eval)[0]
        best_lambda_idx = np.max(candidate_indices)  # 選擇最大的 lambda
        best_lambda = lambda_values[best_lambda_idx]
        best_C = C_values[best_lambda_idx]
        
        # 使用最佳 lambda* 在整個訓練集上重新訓練模型
        try:
            y_train_liblinear = y_train_shuffled.tolist()
            X_train_liblinear = [dict(enumerate(row, start=1)) for row in X_train_shuffled]
            final_model = train(y_train_liblinear, X_train_liblinear, f"-s 6 -c {best_C} -q")
        except Exception as e:
            print(f"實驗 {exp} 中重新訓練最佳模型時出現錯誤: {e}")
            E_out_hist.append(np.inf)
            non_zero_counts.append(0)
            continue
        
        # 在測試集上預測以計算 E_out
        try:
            y_test_liblinear = y_test.tolist()
            X_test_liblinear = [dict(enumerate(row, start=1)) for row in X_test_dense]
            p_labels_test, p_acc_test, p_vals_test = predict(y_test_liblinear, X_test_liblinear, final_model, '-q')
            
            # 計算 0/1 誤差
            E_out = zero_one_error(y_test, np.array(p_labels_test))
            E_out_hist.append(E_out)
            
            # 計算非零權重的數量
            non_zero = count_non_zero_weights(final_model)
            non_zero_counts.append(non_zero)
        except Exception as e:
            print(f"實驗 {exp} 中在測試集上預測時出現錯誤: {e}")
            E_out_hist.append(np.inf)
            non_zero_counts.append(0)
            continue
    
    # 繪製 E_out 直方圖
    print("繪製 E_out 直方圖並保存為 'E_out_histogram_cv.png'...")
    plt.figure(figsize=(10, 6))
    plt.hist(E_out_hist, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Out-of-sample Error (E_out)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Out-of-sample Errors over 1126 Experiments (With 3-Fold CV)')
    plt.grid(True)
    plt.savefig('E_out_histogram_cv.png')
    plt.close()
    
    # 繪製非零權重數量直方圖
    print("繪製非零權重數量直方圖並保存為 'non_zero_weights_histogram_cv.png'...")
    plt.figure(figsize=(10, 6))
    plt.hist(non_zero_counts, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Non-zero Weights')
    plt.ylabel('Frequency')
    plt.title('Histogram of Non-zero Weights in Models over 1126 Experiments (With 3-Fold CV)')
    plt.grid(True)
    plt.savefig('non_zero_weights_histogram_cv.png')
    plt.close()
    
    # 計算並顯示平均 E_out 和平均非零權重數量
    average_E_out = np.mean(E_out_hist)
    average_non_zero = np.mean(non_zero_counts)
    
    print(f"Average Out-of-sample Error (E_out): {average_E_out:.4f}")
    print(f"Average Number of Non-zero Weights: {average_non_zero:.2f}")

if __name__ == "__main__":
    main()
