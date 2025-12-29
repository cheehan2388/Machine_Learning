Machine Learning course offered by profesor Hsuan-Tien Lin .

course url ： "[covered content](https://www.csie.ntu.edu.tw/~htlin/course/ml24fall/)"
cat << 'EOF' > README.md
# 🤖 NTU Machine Learning: Foundations & Systems
### *Mathematical Rigor, System Robustness & Competitive Performance*

This repository contains high-level implementations and theoretical analyses from the **NTU Machine Learning (Fall 2024)** course[cite: 1, 119]. This work represents a mastery of both the mathematical "why" and the engineering "how," culminating in a highly robust final project.

---

## 🏆 Featured Achievement: Generalization & Robustness
In the **Hyper Thrill Machine Learning Baseball (HTMLB)** final project, I engineered a model that prioritized stability over leaderboard noise[cite: 430, 431, 455]:

* **Public Leaderboard Ranking:** 113 / 130 [cite: 438, 473]
* **Private Leaderboard Ranking:** **23 / 130** [cite: 439, 474]
* **Robustness Delta:** **+90 Positions** * **Insight:** This significant jump demonstrates an advanced understanding of the **Bias-Variance Tradeoff**[cite: 107, 655, 656]. While other competitors over-fit to the public test set, my approach utilized rigorous validation to ensure high performance on unseen 2024 game data[cite: 437, 474].

---

## 🛠 Technical Skill Level

### 1. Mathematical Foundations & Proofs
Moving beyond "black-box" library usage, this portfolio includes rigorous derivations for:
* **Newton's Method for Logistic Regression:** Derived the Hessian matrix $A_E(w_t) = X^T DX$ to optimize cross-entropy error[cite: 591, 595, 596].
* **Support Vector Machines (SVM):** Proved the equivalence between Primal and Dual soft-margin problems and analyzed the impact of $C$ and $\gamma$ on margin width[cite: 948, 949, 986, 990].
* **VC Dimension & Complexity:** Calculated $d_{vc}$ and analyzed Rademacher complexity for various hypothesis sets to predict generalization error[cite: 19, 57, 180].

### 2. Algorithmic Implementation (from Scratch)
Implementation of foundational and advanced algorithms without high-level wrappers:
* **Ensemble Learning:** Implementation of **AdaBoost-Stump** and analysis of $E_{in}$ and $E_{out}$ over 500 iterations[cite: 389, 392, 395].
* **Optimization:** Stochastic Gradient Descent (SGD) for Multinomial Logistic Regression and Coordinate Descent for Elastic Net regularization[cite: 300, 301, 612, 652].
* **State Estimation:** Used Extended Kalman Filters (EKF) and PID control for robotics/AMR integration.

### 3. Large-Scale Systems
* **L1-Regularized Logistic Regression:** Applied to the MNIST dataset using **LIBLINEAR** to achieve sparsity and high efficiency[cite: 241, 243, 251, 267].
* **Kernel Engineering:** Implemented and proved the validity of non-linear kernels (Trigonometric, RBF, and Threshold-based)[cite: 884, 887, 893, 905].

---

## 💻 Tech Stack
* **Languages:** Python (NumPy, SciPy), C++
* **Specialized Tools:** LIBLINEAR, LIBSVM, Gazebo, ROS/ROS2
* **Environments:** Kaggle (Competitive ML), Gradescope

---

## 📑 Course Context (NTU Fall 2024)
* **Instructor:** Prof. Hsuan-Tien Lin [cite: 2, 120]
* **Focus:** Bridging the gap between computational learning theory and practical, robust machine learning systems[cite: 11, 202, 214].
EOF