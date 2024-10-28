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
    for index, key_value in x_i.items() : #這樣 index 與 key_value 值就是字典裏對應的, 如 {200:3 } index = 200 , key_value = 3
        init_w = w.get(index, 0.0)       #取  x 裏有值 的對應 w
        dot    += init_w*key_value
    
    return dot

def PLA(x, N, y) :
    w = {}
    correct_count = 0
    update_count = 0
    while correct_count < 5*N :
        indx = random.randint(0,N-1)
        y_i  = y[indx]
        x_i  = x[indx]
        dot  = dot_product(w, x_i)
        predict = predictor_sign(dot)
        if predict != y_i : 
            weight_update (w, x_i, y_i) # 不需要return weight 儅我對 w 做修改 ，就會影響全局 因爲 dict
            correct_count = 0
            update_count += 1
        elif predict == y_i : 
            correct_count += 1
            
    return update_count
    
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
    
    for trial in range(1000):
        random.seed(trial)  # 設置隨機種子
        count = PLA( x, N, y)
        update_counts.append(count)
    min_up = min(update_counts)
    print(min_up)
    # 统计结果，例如计算平均更新次数
    # average_updates = sum(update_counts) / len(update_counts)
    # print(f'平均更新次数：{average_updates}')
    # print
    # #绘制直方图（可以使用 matplotlib）
    # import matplotlib.pyplot as plt
    # plt.hist(update_counts, bins=30)
    # plt.xlabel('exp_counts')
    # plt.ylabel('update_counts')
    # plt.title('PLA')
    # plt.show()    

if __name__ == "__main__" :    # 讓最先運作的程式行是 main  .
    main()