# Logistic Regression

##Introduction

It is widely used in different fields. And it usually handle the two classification problems.  Actually , it also can be seen as a regression algorithm.

It links the features with a probability that the event happens

in the past, we use $ \widehat{y}=f(x) $ but now we use 
$$
\widehat{p}=f(x)
$$
and predict 
$$
\widehat{y}=\left\{\begin{matrix}
0,\: \: \widehat{p}> 0.5\\ 
1,\: \: \widehat{p}\leqslant 0.5
\end{matrix}\right.
$$
In the past, the range of the $ \widehat{y}$ is (-infinity, +infinity). And we want to map it to [0, 1]

so we use **Sigmoid Function**


$$
\sigma \left ( z \right )=\frac{1}{1+e^{-z}}
$$
In the logistic regression,  we have
$$
\widehat{p}=\sigma \left ( \theta ^{T}\cdot x_{b} \right )= \frac{1}{1+e^{-\theta ^{T}\cdot x_{b}}}
$$
and then predict the problem. So how to get the parameters theta?

## Lost Function

We define the cost function
$$
cost= \left\{\begin{matrix}
-log(\widehat{p}),\: \: if \: \: y=1\\ 
-log(1-\widehat{p}),\: \: if \: \: y=0
\end{matrix}\right.
$$


We can see the curve of the cost function.

Then combine them together, we get
$$
cost= -ylog(\widehat{p})-(1-y)log(1-\widehat{p})
$$
So the lost function is
$$
J = -\frac{1}{m}\sum_{i=1}^{m}\: \: y^{(i)}log(\widehat{p}^{(i)})+\left ( 1-y^{(i)} \right )log(1-\widehat{p}^{(i)})
$$
没有公式解,只能使用梯度下降法进行求解

## Using Gradient descent in Logistic Regression

We find that
$$
{log(\sigma \left ( t \right ))}'=1-\sigma (t)
$$
and 
$$
({log(1-\sigma (t))})'=-\sigma (t)
$$
then we get
$$
\frac{\mathrm{d y^{(i)}log(\sigma (X_{b}^{(i)}\theta )) } }{\mathrm{d} \theta_{j}}=y^{(i)}(1-\sigma (X_{b}^{(i)}\theta  ))X_{j}^{(i)}
$$

$$
\frac{\mathrm{d (1-y^{(i)})log(1-\sigma (X_{b}^{(i)}\theta )) } }{\mathrm{d} \theta_{j}}=(1-y^{(i)})(-\sigma (X_{b}^{(i)}\theta  ))X_{j}^{(i)}
$$

and finally
$$
\frac{\partial J(\theta )}{\partial \theta_{j}}=\frac{1}{m}\sum_{i=1}^{m}(\sigma (X_{b}^{(i)}\theta )-y^{(i)})X_{j}^{(i)}=\frac{1}{m}\sum_{i=1}^{m}(\widehat{p}^{(i)}-y^{(i)})X_{j}^{(i)}
$$
so
$$
\bigtriangledown\, J=\frac{1}{m}\begin{Bmatrix}
\sum_{i=1}^{m}(\widehat{p}^{(i)}-y^{(i)})\\ 
\sum_{i=1}^{m}(\widehat{p}^{(i)}-y^{(i)})X_{1}^{(i)}\\ 
\sum_{i=1}^{m}(\widehat{p}^{(i)}-y^{(i)})X_{2}^{(i)}\\ 
...\\ 
\sum_{i=1}^{m}(\widehat{p}^{(i)}-y^{(i)})X_{n}^{(i)}
\end{Bmatrix}=\frac{1}{m}X_{b}^{T}\cdot (\sigma (X_{b}\theta )-y)
$$
compare with gradient in the linear regression:
$$
\bigtriangledown\, J=\frac{2}{m}\begin{Bmatrix}
\sum_{i=1}^{m}(X_{b}^{(i)}\theta -y^{(i)})\\ 
\sum_{i=1}^{m}(X_{b}^{(i)}\theta -y^{(i)})X_{1}^{(i)}\\ 
\sum_{i=1}^{m}(X_{b}^{(i)}\theta -y^{(i)})X_{2}^{(i)}\\ 
...\\ 
\sum_{i=1}^{m}(X_{b}^{(i)}\theta -y^{(i)})X_{n}^{(i)}
\end{Bmatrix}
$$

## Decision Boundary

The decision boundary is
$$
\theta^{T}\cdot x_{b}=0
$$
The decision boundary is a straight line. Actually the logistic regression is a kind of linear regression. 

When we want to show the irregular decision boundary, we can just label lots of  points in the area with different colors for different classes.

## Using Polynomial Features in Logistic Regression

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

def PolynomicalLogisticRegression(degreee):
    return Pipeline([
        ('poly', PolynomialFeatures(degree = degree)),
        ('std_scalar', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])

```

## Logistic Regression in Scikit-learn

When we use polynomial features, we should do regularization in case of overfitting.

We now use $ C\cdot J\left ( \theta  \right )+L_{2} $ instead of $ J\left ( \theta  \right )+\alpha L_{2} $ to do regularization. 

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)
X = np.random.normal(0, 1, size = (200, 2) )
y = np.array( X[:,0]**2 + X[:,1] < 1.5, dtype = 'int' )
for _ in range(20):
    y[np.random.randint(200)] = 1
    
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 666)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

log_reg.score(X_train, y_train)
```

Then we use polynomial features.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

def PolynomicalLogisticRegression(degreee, C, penalty = 'l2'):
    return Pipeline([
        ('poly', PolynomialFeatures(degree = degree)),
        ('std_scalar', StandardScaler()),
        ('log_reg', LogisticRegression(C = C, penalty = penalty))
    ])
```

## OvR and OvO

One VS Rest / One VS All 

![1554327843224](C:\Users\hasee\AppData\Roaming\Typora\typora-user-images\1554327843224.png)

![1554328186934](D:\software\Typora\1554328186934.png)

Generally speaking, OvO is more accurate than OvO.

```python
from sklearn.multiclass import OneVsRestClassifier

ovr = OneVsRestClassifier(log_reg)
ovr.fit(X_train, y_train)
ovr.score(X_test, y_test)

from sklearn.multiclass import OneVsOneClassifier

ovo = OneVsOneClassifier(log_reg)
ovo.fit(X_train, y_train)
ovo.score(X_test, y_test)
```

