from libsvm.svmutil import *
import matplotlib.pyplot as plt
import numpy as np

label, data = svm_read_problem('ml_hw4_feat')

arr_label   = np.array(label)

N = 64
data_num = 8192
experiment_time = 1126
feat  = 12

E_in_lin  = []
E_out_lin = []

E_out_nonl = []
E_in_nonl  = []
np.random.seed(1)

#set poly 

R = 3

x_l = np.zeros((data_num, feat + 1))

z_nl = np.zeros((data_num, feat*R + 1))
for i  in range(data_num) :
    x_l[i,0] = 1 #set x0
    for j in range(1,feat + 1) :
        x_l[i,j] = data[i][j]  
for i in range(data_num):

    z_nl[i,0] = 1
    i_idx = 1
    for l in range (1,R + 1) : 
        
        for j in range (1, feat + 1) :      
            z_nl[i,i_idx] = data[i][j]**l  
            i_idx += 1


for t in range(experiment_time) :
    splice_data = np.random.permutation(data_num)
    train_set   = splice_data[:N] #N_exp  traindata
    test_set    = splice_data[N:]
    
    xtrain_set  = x_l[train_set]   
    xtest_set   = x_l[test_set]

    ztrain_set  = z_nl[train_set]
    ztest_set   = z_nl[test_set]

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

    #polynomial
    z_p_i = np.linalg.pinv(ztrain_set)
    w_nonlin = z_p_i @ ytrain_set
    
    y_hat_non_train = ztrain_set @ w_nonlin 

    
    #Error 
    E_in_non_each  = np.mean((y_hat_non_train - ytrain_set)**2)

    E_in_nonl.append(E_in_non_each)
    E_out_lin.append(per_E_out) , E_in_lin.append(per_E_in)

E_in_diff = []




E_in_diff = np.array(E_in_lin) - np.array(E_in_nonl)

E_in_avg = np.mean(E_in_diff)
print(f"Average In-sample Error Difference: {E_in_avg:.6f}")
plt.figure(figsize=(11,6))  
plt.plot(range(len(E_in_diff)),E_in_diff, label = "In sample Error different(E_in_diff)") 
# plt.plot(E_out, label = "Out of sample Error (E_out)")
plt.xlabel("exp_time")
plt.ylabel("MSE Error deff")
plt.title("In sample Error different btw Lin and Non Lin")
plt.legend()
plt.grid()
plt.show()