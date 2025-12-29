from libsvm.svmutil import *
import matplotlib.pyplot as plt
import numpy as np

label, data = svm_read_problem('ml_hw4_feat')

arr_label   = np.array(label)

N = 64
data_num = 8192
experiment_time = 1126
feat  = 12

alpha = .01
E_in_LR  = []
E_out_LR = []

t_values = list(range(200, 100001, 200))

w_interact = 100000
w_200 = 200


#t = 200
E_all = []

#exp 
E_expt = []


np.random.seed(1)

x = np.zeros((data_num, feat + 1))

for i  in range(data_num) :
   
    x[i,0] = 1 #set x0
    for j in range(1,feat + 1) :
        x[i,j] = data[i][j]   # 也可以 data.get(j,0) 避免有缺失



for e in range(experiment_time) :
    w_init = np.zeros(feat+1)
    
    #mean
    E_in_t  = [] 
    E_out_t = []
    splice_data = np.random.permutation(data_num)
    train_set   = splice_data[:N] #N_exp  traindata
    test_set    = splice_data[N:]
    
    xtrain_set  = x[train_set]   
    xtest_set   = x[test_set]



    ytrain_set  = arr_label[train_set]
    ytest_set   = arr_label[test_set]

    #W_Lin = x^pseudoinv @ y
    x_pseudo_inv = np.linalg.pinv(xtrain_set) 
    w_Lin        = x_pseudo_inv@ytrain_set
    
    # y = wx (outsample)
    y_ht_train = xtrain_set @ w_Lin 
    y_ht_test = xtest_set @ w_Lin

    #E_in = (1/N)(y - y^)^2 , y^ = wLin@X , 
    per_E_in     = np.mean((ytrain_set - y_ht_train)**(2))
    per_E_out    = np.mean((ytest_set - y_ht_test)**(2))  

    t_val = 0

    #SGD time 
    for t in range(1,w_interact+1):  
        #sgD : w_t+1 = w_t + a * Del[x]

        random_n = np.random.randint(0,N)
        x_i = xtrain_set[random_n]
        y_i = ytrain_set[random_n]
        y_hat_i = np.dot(w_init, x_i)
        gradient = -2 * x_i * (y_i - y_hat_i)
        w_init = w_init - alpha * gradient            

        
        if t % 200 == 0 : #這樣可以保留 t = 200 , 400 , 600 。而且t 會增長
          y_hat_train = xtrain_set @  w_init 
          y_hat_test = xtest_set @  w_init
          E_in = np.mean((ytrain_set - y_hat_train)**2)
          E_out = np.mean((ytest_set - y_hat_test)**2)
          E_in_t.append(E_in)
          E_out_t.append(E_out)
          t_val += 1
    E_all.append({'Ein': E_in_t,'Eout' : E_out_t})
    E_out_LR.append(per_E_out) , E_in_LR.append(per_E_in)

E_in_LR_fin = np.mean(E_in_LR)
E_out_LR_fin = np.mean(E_out_LR)

E_in_avg  = []
E_out_avg = []
#mean allvalue exp of 200 , then 400.. 
for i in range(len(E_all[0]['Ein'])) :
    E_in_avg.append(np.mean([E_all[exp]['Ein'][i] for exp in range(experiment_time)])) #npmean 裏面要是可迭代的 比如 list 所以要 []
    E_out_avg.append(np.mean([E_all[exp]['Eout'][i] for exp in range(experiment_time)]))

        
plt.figure(figsize=(11,6))
plt.plot(t_values,E_in_avg , label = "In sample Error SGD (E_in)") 
plt.plot(t_values,E_out_avg, label = "Out of sample Error SGD (E_out)")
plt.axhline(E_in_LR_fin, ls = "--",color = 'r', label = "Avg Ein Linear Regress")
plt.axhline(E_out_LR_fin, ls = "--", color = 'g' , label = 'Avg Eout Linear Regress')

plt.xlabel("interaction (200 per unit)")
plt.ylabel("MSE Error")
plt.title("In sample and Out of sample Error of SGD and SQR")
plt.legend()
plt.grid()
plt.show()



 


    
print(x)