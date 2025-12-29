import random 
from itertools import islice
import statistics
def preprocess_data( pre_data : str , N)  :
    y = []
    x = []

    with open(pre_data,'r' ) as text :
        for row in islice(text,N) :  
            row  = row.strip()       
            if not row :             
                continue             
            row = row.split()        
            label = int(row[0])
            y.append(label)
            feature = {}
            feature[0] = 1         
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

def PLA(x, N, y) :
    w = {}
    correct_count = 0
    update_count = 0
    indx = random.randint(1,N-1)
    while correct_count < 5*N :
        
        y_i  = y[indx]
        x_i  = x[indx]
        dot  = dot_product(w, x_i)
        predict = predictor_sign(dot)
        if predict != y_i : 
            weight_update (w, x_i, y_i) 
            correct_count = 0
            update_count += 1
        elif predict == y_i : 
            correct_count += 1
            indx = random.randint(0, N - 1)         
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
        random.seed(trial)  
        count = PLA( x, N, y)
        update_counts.append(count)
    min_up = min(update_counts)
    
    
    median_value = statistics.median(update_counts)
    print(f'meidan : {median_value}')

    
    import matplotlib.pyplot as plt
    plt.hist(update_counts, bins=30)
    plt.xlabel('exp_counts')
    plt.ylabel('update_counts')
    plt.title('PLA')
    plt.show()    

if __name__ == "__main__" :    
    main()