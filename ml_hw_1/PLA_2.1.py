import random
import math
import matplotlib.pyplot as plt

# 自定义 sign 函数
def sign(value):
    if value > 0:
        return 1
    else:
        return -1  # 包括 value == 0 的情况

# 计算权重向量的范数
def calculate_norm(w):
    return math.sqrt(sum(value ** 2 for value in w.values()))

# 计算权重向量和特征向量的内积
def dot_product(w, x_n):
    return sum(w.get(index, 0.0) * value for index, value in x_n.items())

# 更新权重向量
def update_weights(w, x_n, y_n):
    for index, value in x_n.items():
        w[index] = w.get(index, 0.0) + y_n * value

# 加载数据
def load_data(file_path, N):
    X = []
    Y = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            label = int(parts[0])
            Y.append(label)
            features = {}
            for item in parts[1:]:
                index, value = item.split(':')
                features[int(index)] = float(value)
            features[0] = 1.0  # 添加偏置项 x0 = 1.0
            X.append(features)
            if len(X) == N:
                break
    return X, Y

# 主函数
def main():
    N = 200  # 样本数量
    T = 70   # 只记录前 70 次更新
    X, Y = load_data('rcv1_train.txt', N)
    norms_per_experiment = []

    for trial in range(1000):
        random.seed(trial)  # 设置随机种子
        w = {}  # 初始化权重向量
        consecutive_correct = 0
        update_count = 0
        norms = []  # 当前实验的范数列表

        while consecutive_correct < 5 * N:
            idx = random.randint(0, N - 1)
            x_n = X[idx]
            y_n = Y[idx]

            dp = dot_product(w, x_n)
            prediction = sign(dp)

            if prediction != y_n:
                update_weights(w, x_n, y_n)
                update_count += 1
                consecutive_correct = 0
                norm = calculate_norm(w)
                if update_count <= T:
                    norms.append(norm)
                elif update_count > T:
                    # 超过 T 次更新后，不再记录范数
                    pass
            else:
                consecutive_correct += 1

            # 如果更新次数超过 T，且已记录完毕，可以提前结束循环
            if update_count > T and len(norms) >= T:
                break

        norms_per_experiment.append(norms)

    # 准备绘图数据
    t_values = list(range(1, T + 1))  # t 从 1 到 T

    # 绘制曲线
    plt.figure(figsize=(10, 6))

    for norms in norms_per_experiment:
        if len(norms) == T:
            plt.plot(t_values, norms, color='blue', alpha=0.1)

    plt.xlabel('update_count')
    plt.ylabel('||w_t||Norm of Weight Vector')
    plt.title('Norm of Weight Vector vs. Update Count in 1000 Experiments of PLA')
    plt.show()
if __name__ == "__main__":
    main()
