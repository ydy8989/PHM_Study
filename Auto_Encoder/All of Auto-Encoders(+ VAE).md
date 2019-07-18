# All of Auto-Encoders(+ VAE)

**A way for Unsupervised Learning of Nonlinear Manifold**

---

**오토인코더 :** 입력과 출력의 형태가 동일하게 만드는 네트워크 구조를 일컫는다고 생각할 수 있음.

**4 Main Keywords**

1. Unsupervised learning
2. Manifold learning (dimensionality reduction)
3. Generative model learning
4. ML density estimation

---

**오토인코더 학습할 때 :**

학습 방법은 비교사(비지도) 학습 방법을 따름 **keyword 1**

Loss는 negative maximal likelihood로 해석 = loss minimize : maximal likelihood function **keyword 4**

---

**학습된 오토인코더에서:**

인코더는 차원 축소 역할을 수행하며, **keyword 2**

디코더는 생성 모델의 역할을 수행한다. **keyword 3**

---

---

## Contents

1. Revisit Deep Neural Networks
   1. Machine learning problem
   2. Loss function viewpoint I : Back-propagation
   3. Loss function viewpoint II : Maximum likelihood
   4. Maximum likelihood for auto-encoders
2. Manifold Learning
   1. Four objectives
   2. Dimension reduction
   3. Density estimation
3. Auto-encoders
   1. Auto-encoders (AE)
   2. Denosing AE(DAE)
   3. Contractive AE(CAE)
4. Variational Auto-encoders
   1. Variational AE(VAE)
   2. Conditional VAE(CVAE)
   3. Adversarial AE(AAE)
5. Applications
   1. Retrieval
   2. Generation
   3. Regression
   4. GAN + VAE

---

# 1. Revisit Deep Neural Networks

## 1.1 Machine learning problem

**Classic Machine Learning?**

1. collect training data
   $$
   x = \{x_1, x_2, ...,x_N\}\\
   y = \{y_1, y_2, ..., y_N\}\\
   \mathcal{D} = \{(x_1,y_1),(x_2,y_2),...,(x_N,y_N)\}
   $$

2. Define functions
   $$
   output : f_{\theta}(x)\\
   Loss : L(f_\theta(x),y)
   $$

3. Learning/Training

   - find the optimal parameter

4. Predicting/Testing

   - Compute optimal function output

---

**Collect training data**

- 알아서....

**Define functions**

> - 모델을 정하고, Loss function을 정하는 과정
>
>   > 데이터와 모델, 그리고 loss function을 정하면 사실상 바꿀 것이 없음. (제약)
>
> - **Loss function** 
>   $$
>   Loss = L(f_\theta(x),y) = \sum_i L(f_\theta(x_i),y_i)
>   $$
>
> - 지금까지의 딥러닝 과정에서는 Loss function을 마음대로 사용할 수 없음. 왜??
>
>   **Why?** Back propagation 때문임. 역전파 과정이 진행되어야 loss 값을 구할 수 있기 때문임.
>
>   - 역전파 알고리즘은 수식 4개로 진행되는데, 이를 진행하기 위한 **2가지 가정**이 있음
>
>     > **Assumption 1.**) 전체 트레인 loss는 각 샘플별 loss의 합과 같다.
>
>     > **Assumption 2.**)  loss 함수를 구할 때, 네트워크의 **출력값과 정답값**만 가지고 계산한다.
>
>   - 이 두가지 가정이 안되면, Back propagation이 불가능

---

**Learning/Training**

> - loss를 정의하면, 이 loss를 minimize하는 파라미터 Theta를 찾는데...
>
> - 이 Theta를 찾는 방식은 대부분 경사하강법을 사용함(Optimal theory의 가장 기초)
>
> - Iterative method를 채택 : 한 단계씩 점점 가까워지는 방식
>
>   > - 여기서 두 가지 질문이 발생함
>   >
>   >   > 1. 어떻게 iterate 할꺼?
>   >   > 2. 언제 멈출꺼?
>   >
>   > - answer 1 : 로스값이 줄어드는 방향이면 무조건 그쪽으로 움직인다.
>   >
>   > - answer 2 : 움직여도 로스값이 변함없을 경우에 멈춤
>   >
>   >   ------
>   >
>   > - 이런 식으로 theta를 바꾸긴 바꿔야하는데... Dimension 클수록 이런 문제가 더더더 발생
>   >
>   >   > 3. 어떻게 theta를 바꿔야 loss가 줄어들어?
>   >
>   > - answer 3 :
>   >
>   >   - Taylor Expansion : 
>   >
>   >     ![](https://github.com/ydy8989/PHM_Study/tree/master/Auto_Encoder/pic/figure1.png)
>   >
>   >   - Approximation : 테일러 expansion처럼 다 확장시키는게 아니라 1차 미분계수만 구해서 계산하는 방식.
>   >
>   >     ![](https://github.com/ydy8989/PHM_Study/tree/master/Auto_Encoder/pic/figure2.png)
>   >
>   >     Learning rate를 사용하여 조금씩 파라미터 값을 바꾸는 것은 로스 함수의 1차 미분항까지만 사용했기 때문에, 아주 좁은 영역(sample data의 인접 지역)에서만 감소 방향이 정확하기 때문임...
>   >
>   >   ------
>   >
>   > - 전체 데이터에 대한 로스 함수가 각 데이터 샘플에 대한 로스의 합으로 구성되어 있기에 미분 계산을 효율적으로 할 수 있다. 
>   >
>   > - 만약 곱으로 구성되어 있으면, 미분을 위해 모든 샘플의 결과를 메모리에 저장하여야 한다.
>   >
>   > - 원래는 모든 데이터에 대한 로스 미분값의 합을 구한 후 파라미터를 갱신해야 하지만, 배치 크기만큼만 로스 미분값의 합을 구한 후 파라미터를 갱신한다. (stochastic 경사하강) =>> 전체와 배치의 경사하강이 같을 거라는 기대

---

---

## 1.2 Loss function viewpoint I : Back-propagation

- **2가지 관점**
  1. 딥러닝 학습 시 **Back-propagation**이 잘 동작하는 관점 : **Cross Entropy**
  2. 네트워크 출력값이 **Continuous value**면 **Mean Square Error** and **Discrete**하면 **Cross Entropy**

****

### Type 1 : Mean Square Error / Quadratic loss



### Type 2 : 

## 1.3 Loss function viewpoint II : Maximum likelihood

