from libsvm.svmutil import *
import matplotlib.pyplot as plt
import numpy as np

label, data = svm_read_problem('ml_hw_2_feat')

arr_label   = np.array(label)

N = 32
data_num = 8192
experiment_time = 1126
feat  = 12

E_in  = []
E_out = []

np.random.seed(1)

x = np.zeros((data_num, feat + 1))

for i  in range(data_num) :
    x[i,0] = 1 #設置x0
    for j in range(1,feat + 1) :
        x[i,j] = data[i][j] #data[第幾個字典][第幾個鍵值]

for t in range(experiment_time) :
    splice_data = np.random.permutation(data_num)
    train_set   = splice_data[:N] #N_exp 個 traindata
    test_set    = splice_data[N:]
    
    xtrain_set  = x[train_set]  # train set 是一組數字 index, x【train_set] 就會對著他要的行來取。
    xtest_set   = x[test_set]

    ytrain_set  = arr_label[train_set]
    ytest_set   = arr_label[test_set]

    #W_Lin = x^pseudoinv @ y
    x_pseudo_inv = np.linalg.pinv(xtrain_set) 
    w_Lin        = x_pseudo_inv@ytrain_set
    
    # y = wx (outsample)
    y_ht_train = xtrain_set @ w_Lin #don't forget mtrx rule
    y_ht_test = xtest_set @ w_Lin

    #E_in = (1/N)(y - y^)^2 , y^ = wLin@X , 
    per_E_in     = np.mean((ytrain_set - y_ht_train)**(2))
    per_E_out    = np.mean((ytest_set - y_ht_test)**(2))  

    E_out.append(per_E_out) , E_in.append(per_E_in)

plt.figure(figsize=(11,6))
plt.plot(E_in, label = "In sample Error (E_in)") 
plt.plot(E_out, label = "Out of sample Error (E_out)")
plt.xlabel("exp_time")
plt.ylabel("MSE Error")
plt.title("In sample and Out of sample Error")
plt.legend()
plt.grid()
plt.show()



# for n in exp_t:


    
print(x)