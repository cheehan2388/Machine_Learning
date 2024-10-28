import math
import random 
from itertools import islice

def preprocess_data( pre_data : str , N)  :
    y = []
    x = []

    with open(pre_data,'r' ) as text :
        for row in islice(text,N) :  #slice有點像head, 因爲open 后不能用head  
            row  = row.strip()       #preproces數據有時候會有奇怪的符號或空
            if not row :             #經過strip后如果是空字符就
                continue             #跳到下個循環
            row = row.split()        #把數據依據空白來分割
            label = int(row[0])
            y.append(label)
            feature = {}
            feature[0] = 1           #每個字典的頭 x_0 設定為 1
            for item in row[1:]:        
                key ,value = item.split(':')
                feature[int(key)]= float(value)
            x.append(feature)
            
        
    return y , x
def dot_product(w, x_i):
    dot   =  0
    for index, key_value in x_i.items() : 
        init_w = w.get(index, 0.0)       
        dot    += init_w*key_value
    
    return dot


def calculate_norm(w):
    return math.sqrt(sum(value ** 2 for value in w.values()))

def weight_update(w , x_i , y_i):
    for index , item in x_i.items() :
        w[index] = w.get(index, 0.0) + y_i*item

   
    
def predictor_sign(dot) :
    if dot > 0:
        return 1
    if dot <= 0 :
        return -1
    

def main() :
    update_counts = []
    N    = 200
    y , x  = preprocess_data('rcv1_train.txt',N)
    norms_per_experiment = []
    Tmin = None

    for count in range(1000):
        random.seed(count)  # 设置随机种子
        w = {}  # 初始化权重向量
        correct_count = 0
        update_count = 0
        norms = []  # 当前实验的范数列表

        while correct_count < 5 * N:
            idx = random.randint(0, N - 1)
            x_n = x[idx]
            y_n = y[idx]

            dot = dot_product(w, x_n)
            prediction = predictor_sign(dot)

            if prediction != y_n:
                weight_update(w, x_n, y_n)
                update_count += 1
                correct_count = 0
                norm = calculate_norm(w)
                norms.append(norm)
            else:
                correct_count += 1

        norms_per_experiment.append(norms)
        update_counts.append(update_count)

        if Tmin is None or update_count < Tmin:
            Tmin = update_count

    # 截取范数到 Tmin
    truncated_norms = [norms[:Tmin] for norms in norms_per_experiment]
    t_values = list(range(1, Tmin + 1))
    import matplotlib.pyplot as plt

    for norms in truncated_norms:
        plt.plot(t_values, norms, color='blue', alpha=0.1)  # 使用 alpha 设置透明度

    plt.xlabel('update_count')
    plt.ylabel('||w_t||Norm of Weight Vector')
    plt.title('Norm of Weight Vector vs. Update Count in 1000 Experiments of PLA')
    plt.show()

if __name__ == "__main__" :    # 讓最先運作的程式行是 main  .
    main()