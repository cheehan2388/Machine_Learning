import numpy as np
import random
from itertools import islice
import matplotlib.pyplot as plt
import statistics

def preprocess_data(pre_data: str, N):
    y = []
    x = []

    with open(pre_data, 'r') as text:
        for row in islice(text, N):
            row = row.strip()  
            if not row:
                continue  
            row = row.split() 
            label = int(row[0])
            y.append(label)
            feature = {}
            feature[0] = 1  
            for item in row[1:]:
                key, value = item.split(':')
                feature[int(key)] = float(value)
            x.append(feature)

    return y, x


def calculate_normalize(w):
    
    w_values = np.array(list(w.values()))
  
    return np.linalg.norm(w_values)


def dot_product(w, x_i):
    dot = 0
    for index, key_value in x_i.items():
        init_w = w.get(index, 0.0)  
        dot += init_w * key_value
    return dot


def PLA(x, N, y):
    w = {} 
    correct_count = 0
    update_count = 0
    w_path = [] 
    while correct_count < 5 * N:
        indx = random.randint(0, N - 1)  
        y_i = y[indx]
        x_i = x[indx]
        dot = dot_product(w, x_i)
        predict = predictor_sign(dot)
        if predict != y_i:  
            weight_update(w, x_i, y_i)
            correct_count = 0 
            update_count += 1 
            norm = calculate_normalize(w)
            w_path.append(norm)  
        else:
            correct_count += 1  

    return update_count, w_path  

def weight_update(w, x_i, y_i):
    for index, item in x_i.items():
        w[index] = w.get(index, 0.0) + y_i * item


def predictor_sign(dot):
    if dot > 0:
        return 1
    else:
        return -1  


def main():
    update_counts = []
    path_exp = [] 
    N = 200
    y, x = preprocess_data('rcv1_train.txt', N)

    for trial in range(1000):
        random.seed(trial)  
        count, w_path = PLA(x, N, y)
        update_counts.append(count)
        path_exp.append(w_path)  

   
    T_min = min(update_counts)
    print(f"T_min: {T_min}")

    truncated_norms = [norms[:T_min] for norms in path_exp]

  
    t_values = list(range(1, T_min + 1))  

    plt.figure(figsize=(10, 6))

    for norms in truncated_norms:
        plt.plot(t_values, norms, color='blue', alpha=0.1)

    plt.xlabel('update_count', fontsize=12)
    plt.ylabel('||w_t||Norm of Weight Vector', fontsize=12)
    plt.title('Norm of Weight Vector vs. Update Count in 1000 Experiments of PLA', fontsize=14)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
