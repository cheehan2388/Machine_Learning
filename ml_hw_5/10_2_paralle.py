import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端以避免 GUI 錯誤
import matplotlib.pyplot as plt
from liblinear.liblinearutil import train, predict, svm_read_problem
import urllib.request
import os
import bz2
from joblib import Parallel, delayed
from tqdm import tqdm

# 定義計算 0/1 錯誤的純 Python 函數
def zero_one_error(y_true, y_pred):
    incorrect = sum(1 for yt, yp in zip(y_true, y_pred) if yt != yp)
    return incorrect / len(y_true)

# 定義單次實驗的函數
def run_single_experiment(exp_idx):
    global global_y_train, global_X_train, global_y_test, global_X_test, global_lambda_values, global_C_values
    
    # 設置隨機種子並隨機打亂訓練數據
    import random
    random.seed(exp_idx)
    shuffled_indices = list(range(len(global_y_train)))
    random.shuffle(shuffled_indices)
    y_train_shuffled = [global_y_train[i] for i in shuffled_indices]
    X_train_shuffled = [global_X_train[i] for i in shuffled_indices]
    
    # 初始化每個 lambda 的 Ein 和最佳模型
    min_Ein = float('inf')
    best_model = None
    
    for C in global_C_values:
        # 將訓練數據轉換為 LIBLINEAR 格式
        y_train_liblinear = y_train_shuffled
        X_train_liblinear = [dict(enumerate(row, start=1)) for row in X_train_shuffled]
        
        # 訓練模型 (-s 6: L1 正則化的邏輯回歸, -c: 正則化參數, -q: 靜默模式)
        model = train(y_train_liblinear, X_train_liblinear, f"-s 6 -c {C} -q")
        
        # 使用訓練數據進行預測以計算 Ein
        p_labels, _, _ = predict(y_train_liblinear, X_train_liblinear, model, '-q')
        
        # 計算 0/1 錯誤率
        E_in = zero_one_error(y_train_shuffled, p_labels)
        
        # 選擇 Ein 最小的模型，若相等則選擇最大的 lambda（最小的 C）
        if E_in < min_Ein or (E_in == min_Ein and C < global_C_values[best_lambda_idx] if best_model else False):
            min_Ein = E_in
            best_model = model
    
    # 使用最佳模型在測試集上進行預測
    y_test_liblinear = global_y_test
    X_test_liblinear = [dict(enumerate(row, start=1)) for row in global_X_test]
    
    p_labels_test, _, _ = predict(y_test_liblinear, X_test_liblinear, best_model, '-q')
    
    # 計算 E_out
    E_out = zero_one_error(y_test_liblinear, p_labels_test)
    
    # 計算非零權重數量
    non_zero = sum(1 for w in best_model.w if w != 0)
    
    return E_out, non_zero

# 定義下載並解壓縮 bz2 文件的函數
def download_and_extract(url, bz2_filename, extracted_filename):
    if not os.path.isfile(extracted_filename):
        if not os.path.isfile(bz2_filename):
            print(f"正在從 {url} 下載 {bz2_filename}...")
            urllib.request.urlretrieve(url, bz2_filename)
            print(f"已下載 {bz2_filename}。")
        print(f"正在解壓縮 {bz2_filename}...")
        with bz2.BZ2File(bz2_filename) as fr, open(extracted_filename, 'wb') as fw:
            fw.write(fr.read())
        print(f"已解壓縮至 {extracted_filename}。")
    else:
        print(f"{extracted_filename} 已存在，跳過下載和解壓縮。")

# 定義數據集的 URL 和文件名
train_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2"
test_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.t.bz2"

train_bz2 = "mnist.scale.bz2"
test_bz2 = "mnist.scale.t.bz2"
train_file = "mnist.scale"
test_file = "mnist.scale.t"

# 下載並解壓縮訓練和測試數據
download_and_extract(train_url, train_bz2, train_file)
download_and_extract(test_url, test_bz2, test_file)

# 加載訓練和測試數據
print("加載訓練數據...")
y_train_full, x_train_full = svm_read_problem(train_file)
print("加載測試數據...")
y_test_full, x_test_full = svm_read_problem(test_file)

# 定義過濾類別並映射標籤的函數
def filter_and_map_classes(y, x, class1=2, class2=6):
    filtered_y = []
    filtered_x = []
    for label, features in zip(y, x):
        if label == class1 or label == class2:
            # 將 class1 映射為 -1，class2 映射為 +1
            mapped_label = -1 if label == class1 else 1
            filtered_y.append(mapped_label)
            filtered_x.append(features)
    return filtered_y, filtered_x

# 過濾並映射訓練和測試數據
print("過濾並映射訓練數據...")
y_train, x_train = filter_and_map_classes(y_train_full, x_train_full, class1=2, class2=6)
print("過濾並映射測試數據...")
y_test, x_test = filter_and_map_classes(y_test_full, x_test_full, class1=2, class2=6)

# 確定特徵數量
max_feature_index = 0
for sample in x_train + x_test:
    if sample:
        current_max = max(sample.keys())
        if current_max > max_feature_index:
            max_feature_index = current_max

# 特徵數量（不包括偏置項）
feat = max_feature_index
print(f"特徵數量（不包括偏置項）：{feat}")

# 定義將稀疏特徵轉換為密集列表並添加偏置項的函數
def convert_to_dense(x, num_features):
    dense_x = []
    for features in x:
        dense = [0.0] * (num_features + 1)  # +1 為偏置項
        dense[0] = 1.0  # 偏置項 x0=1
        for index, value in features.items():
            if 1 <= index <= num_features:
                dense[index] = value
        dense_x.append(dense)
    return dense_x

# 將訓練和測試數據轉換為密集列表
print("轉換訓練數據為密集格式並添加偏置項...")
global_X_train = convert_to_dense(x_train, feat)
print("轉換測試數據為密集格式並添加偏置項...")
global_X_test = convert_to_dense(x_test, feat)

# 定義 log10(lambda) 值及其對應的 C 值
log_lambda_values = [-2, -1, 0, 1, 2, 3]
global_lambda_values = [10**l for l in log_lambda_values]
global_C_values = [1/lambda_val for lambda_val in global_lambda_values]

print("Lambda 值:", global_lambda_values)
print("對應的 C 值:", global_C_values)

# 將 y_train 和 y_test 設置為全局變量
global_y_train = y_train
global_y_test = y_test

# 定義實驗參數
experiment_time = 1126  # 實驗次數

# 並行化執行實驗
print("開始進行並行實驗...")
results = Parallel(n_jobs=-1, backend='loky')(
    delayed(run_single_experiment)(exp_idx) for exp_idx in tqdm(range(experiment_time), desc="進行實驗")
)

# 分離出 E_out 和 non_zero_counts
E_out_hist, non_zero_counts = zip(*results)
E_out_hist = list(E_out_hist)
non_zero_counts = list(non_zero_counts)

# 繪製 E_out 的直方圖
print("繪製 E_out 直方圖...")
plt.figure(figsize=(10, 6))
plt.hist(E_out_hist, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Out-of-sample Error (E_out)')
plt.ylabel('Frequency')
plt.title('Histogram of Out-of-sample Errors over 1126 Experiments')
plt.grid(True)
plt.savefig('E_out_histogram.png')  # 保存圖像
plt.close()

# 繪製非零權重數量的直方圖
print("繪製非零權重數量直方圖...")
plt.figure(figsize=(10, 6))
plt.hist(non_zero_counts, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Number of Non-zero Weights')
plt.ylabel('Frequency')
plt.title('Histogram of Non-zero Weights in Models over 1126 Experiments')
plt.grid(True)
plt.savefig('non_zero_weights_histogram.png')  # 保存圖像
plt.close()

print("實驗完成。")
