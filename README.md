# lightgbm相关总结

## lightgbm原理详解

> lightgbm本质是实现GBDT算法的框架，但在工程实现上了做了很多优化，支持高效率的并行训练，并且具有更快的训练速度、更低的内存消耗、更好的准确率、支持分布式可以快速处理海量数据等优点。因此了解lightgbm原理本质是要了解GBDT的原理，要熟悉GBDT拟合分类和回归的原理，了解相关公式的推导。

> GBDT的原理及相关公式参考：

https://www.cnblogs.com/pinard/p/6140514.html

https://zhuanlan.zhihu.com/p/53980138

https://zhuanlan.zhihu.com/p/29765582


> lightgbm介绍参考：

https://zhuanlan.zhihu.com/p/99069186


## 自定义损失函数说明
> lightgbm支持自定义损失函数，在lightgbm的API中具体如下：

`fobj`: Customized objective function，自定义的损失函数，返回值有两个，grad和hess，grad代表损失函数对 $\hat y$ 的一阶导数，hess代表损失函数对 $\hat y$ 的二阶导数。<font color=Red>**另外，hess and grad both need to be 1-dimensional arrays of the same length as the number of examples in our training data set.**</font>

`feval`: Customized evaluation function，自定义的eval函数

`metric`: 交叉验证时的监控指标 

### 回归实例自定义损失函数

**详细代码参考：[lightgbm_regression.ipynb](https://github.com/ShaoQiBNU/lightgbmTips/blob/main/lightgbm_regression.ipynb)**

#### 自定义mse实现
> lightgbm实现regression时默认采用L2损失函数，metric设置为L2。自定义实现一下mse，对比一下两者的差异，具体如下：

mse损失函数为：
$$Loss= \frac{1}{N} \sum_{i=1}^N (y_{i} - \hat y_{i})^{2}$$

一阶导数和二阶导数为：

$$\frac{\partial Loss}{\partial \hat y_{i}}= \frac{1}{N} \cdot -2 \cdot (y_{i} - \hat y_{i}) = \frac{1}{N} \cdot 2 \cdot (\hat y_{i} - y_{i}) $$

$$\frac{\partial^2 Loss}{\partial^2 \hat y_{i}}= \frac{1}{N} \cdot 2$$

实现lightgbm的`fobj`和`feval`如下：

```python
# custom loss and eval
def custom_mse_loss(y_pred, data):
    y_true = data.get_label()
    error = y_pred-y_true
    
    #1st derivative of loss function
    grad = 2 * error

    #2nd derivative of loss function
    hess = 0 * error + 2
    
    return grad, hess

def custom_mse_eval(y_pred, data):

    y_true = data.get_label()
    error = y_pred-y_true

    return 'l2 loss custom', np.mean(np.square(error)), False
```

对比一下lightgbm默认regression和基于自定义mse的regression的差异，如下：

**lightgbm默认regression**
```
[1]	training's l2: 1.23856	valid_1's l2: 1.2715
Training until validation scores don't improve for 10 rounds.
[2]	training's l2: 1.16057	valid_1's l2: 1.19255
[3]	training's l2: 1.09633	valid_1's l2: 1.12749
[4]	training's l2: 1.03179	valid_1's l2: 1.06198
[5]	training's l2: 0.979685	valid_1's l2: 1.0094
[6]	training's l2: 0.925378	valid_1's l2: 0.953761
[7]	training's l2: 0.876336	valid_1's l2: 0.904049
[8]	training's l2: 0.828646	valid_1's l2: 0.856528
[9]	training's l2: 0.787129	valid_1's l2: 0.814784
[10]	training's l2: 0.74667	valid_1's l2: 0.773438
[11]	training's l2: 0.712037	valid_1's l2: 0.738821
[12]	training's l2: 0.679931	valid_1's l2: 0.706476
[13]	training's l2: 0.651992	valid_1's l2: 0.679253
[14]	training's l2: 0.622372	valid_1's l2: 0.650223
[15]	training's l2: 0.595443	valid_1's l2: 0.623187
[16]	training's l2: 0.570915	valid_1's l2: 0.599406
[17]	training's l2: 0.549103	valid_1's l2: 0.577307
[18]	training's l2: 0.52927	valid_1's l2: 0.55756
[19]	training's l2: 0.510518	valid_1's l2: 0.53932
[20]	training's l2: 0.493825	valid_1's l2: 0.522831
Did not meet early stopping. Best iteration is:
[20]	training's l2: 0.493825	valid_1's l2: 0.522831
```
**自定义mse实现**
```
[1]	training's l2: 5.0927	training's l2 loss custom: 5.0927	valid_1's l2: 5.15175	valid_1's l2 loss custom: 5.15175
Training until validation scores don't improve for 10 rounds.
[2]	training's l2: 4.6387	training's l2 loss custom: 4.6387	valid_1's l2: 4.69574	valid_1's l2 loss custom: 4.69574
[3]	training's l2: 4.23561	training's l2 loss custom: 4.23561	valid_1's l2: 4.29176	valid_1's l2 loss custom: 4.29176
[4]	training's l2: 3.8645	training's l2 loss custom: 3.8645	valid_1's l2: 3.91779	valid_1's l2 loss custom: 3.91779
[5]	training's l2: 3.53621	training's l2 loss custom: 3.53621	valid_1's l2: 3.58798	valid_1's l2 loss custom: 3.58798
[6]	training's l2: 3.23229	training's l2 loss custom: 3.23229	valid_1's l2: 3.28141	valid_1's l2 loss custom: 3.28141
[7]	training's l2: 2.95794	training's l2 loss custom: 2.95794	valid_1's l2: 3.00386	valid_1's l2 loss custom: 3.00386
[8]	training's l2: 2.70724	training's l2 loss custom: 2.70724	valid_1's l2: 2.75177	valid_1's l2 loss custom: 2.75177
[9]	training's l2: 2.48219	training's l2 loss custom: 2.48219	valid_1's l2: 2.52559	valid_1's l2 loss custom: 2.52559
[10]	training's l2: 2.27626	training's l2 loss custom: 2.27626	valid_1's l2: 2.31704	valid_1's l2 loss custom: 2.31704
[11]	training's l2: 2.09297	training's l2 loss custom: 2.09297	valid_1's l2: 2.13314	valid_1's l2 loss custom: 2.13314
[12]	training's l2: 1.92645	training's l2 loss custom: 1.92645	valid_1's l2: 1.96523	valid_1's l2 loss custom: 1.96523
[13]	training's l2: 1.77721	training's l2 loss custom: 1.77721	valid_1's l2: 1.81642	valid_1's l2 loss custom: 1.81642
[14]	training's l2: 1.63816	training's l2 loss custom: 1.63816	valid_1's l2: 1.67793	valid_1's l2 loss custom: 1.67793
[15]	training's l2: 1.51229	training's l2 loss custom: 1.51229	valid_1's l2: 1.55136	valid_1's l2 loss custom: 1.55136
[16]	training's l2: 1.39817	training's l2 loss custom: 1.39817	valid_1's l2: 1.43762	valid_1's l2 loss custom: 1.43762
[17]	training's l2: 1.29552	training's l2 loss custom: 1.29552	valid_1's l2: 1.3341	valid_1's l2 loss custom: 1.3341
[18]	training's l2: 1.20288	training's l2 loss custom: 1.20288	valid_1's l2: 1.24081	valid_1's l2 loss custom: 1.24081
[19]	training's l2: 1.11832	training's l2 loss custom: 1.11832	valid_1's l2: 1.15658	valid_1's l2 loss custom: 1.15658
[20]	training's l2: 1.04232	training's l2 loss custom: 1.04232	valid_1's l2: 1.08042	valid_1's l2 loss custom: 1.08042
[21]	training's l2: 0.973282	training's l2 loss custom: 0.973282	valid_1's l2: 1.01089	valid_1's l2 loss custom: 1.01089
[22]	training's l2: 0.909042	training's l2 loss custom: 0.909042	valid_1's l2: 0.945647	valid_1's l2 loss custom: 0.945647
[23]	training's l2: 0.850424	training's l2 loss custom: 0.850424	valid_1's l2: 0.886551	valid_1's l2 loss custom: 0.886551
[24]	training's l2: 0.796268	training's l2 loss custom: 0.796268	valid_1's l2: 0.832558	valid_1's l2 loss custom: 0.832558
[25]	training's l2: 0.749875	training's l2 loss custom: 0.749875	valid_1's l2: 0.785854	valid_1's l2 loss custom: 0.785854
[26]	training's l2: 0.707394	training's l2 loss custom: 0.707394	valid_1's l2: 0.743053	valid_1's l2 loss custom: 0.743053
[27]	training's l2: 0.669198	training's l2 loss custom: 0.669198	valid_1's l2: 0.704867	valid_1's l2 loss custom: 0.704867
[28]	training's l2: 0.631962	training's l2 loss custom: 0.631962	valid_1's l2: 0.667879	valid_1's l2 loss custom: 0.667879
[29]	training's l2: 0.598686	training's l2 loss custom: 0.598686	valid_1's l2: 0.634531	valid_1's l2 loss custom: 0.63453
[30]	training's l2: 0.569876	training's l2 loss custom: 0.569876	valid_1's l2: 0.605438	valid_1's l2 loss custom: 0.605438
Did not meet early stopping. Best iteration is:
[30]	training's l2: 0.569876	training's l2 loss custom: 0.569876	valid_1's l2: 0.605438	valid_1's l2 loss custom: 0.605438
```

从上面可以看出，自定义的mse和lightgbm默认的L2损失函数数值基本一致，但是为何自定义的mse需要设置更大的num_boost_round才能达到lightgbm默认的训练效果？
待补充

#### 自定义分段mse实现

> 参考的教程里有一个分段的mse，具体可以自行推导一下，具体代码如下：

```python
def custom_asymmetric_train(y_pred, data):
    y_true = data.get_label()
    residual = (y_true - y_pred)
    grad = np.where(residual<0, -2*10.0*residual, -2*residual)
    hess = np.where(residual<0, 2*10.0, 2.0)
    return grad, hess

def custom_asymmetric_valid(y_pred, data):
    y_true = data.get_label()
    residual = (y_true - y_pred)
    loss = np.where(residual < 0, (residual**2)*10.0, residual**2) 
    return "custom_asymmetric_eval", np.mean(loss), False
```

参考：

https://www.datasnips.com/110/lightgbm-custom-loss-function/

https://towardsdatascience.com/custom-loss-functions-for-gradient-boosting-f79c1b40466d

https://github.com/manifoldai/mf-eng-public/blob/master/notebooks/custom_loss_lightgbm.ipynb

### 分类实例自定义损失函数
> lightgbm实现二分类时默认采用对数损失函数，也就是交叉熵损失函数，自定义实现一下交叉熵，对比一下两者的差异，具体如下：

交叉熵损失函数为：
$$Loss= -\sum_{i=1}^N y_{i} \cdot log(p_{i}) +（1-y_{i}) \cdot log(1-p_{i}) $$

$$p_{i} = sigmoid(\hat y_{i}) = \frac{1}{1+exp^{-\hat y_{i}}} $$

一阶导数和二阶导数为：

$$\frac{\partial p_{i}}{\partial \hat y_{i}}= (\frac{1}{1+exp^{-\hat y_{i}}})^{'} = \frac{exp^{-\hat y_{i}}}{(1+exp^{-\hat y_{i}})^{2}} = \frac{1 + exp^{-\hat y_{i}} -1}{(1+exp^{-\hat y_{i}})^{2}} = \frac{1}{1+exp^{-\hat y_{i}}} \cdot (1 - \frac{1}{1+exp^{-\hat y_{i}}}) = p_{i} \cdot (1 - p_{i})$$

$$\frac{\partial Loss}{\partial p_{i}}= - (\frac{y_{i}}{p_{i}} + (1 - y_{i}) \cdot \frac{-1}{1 - p_{i}}) = - \frac{y_{i}}{p_{i}} + \frac{1 - y_{i} }{1 - p_{i}} $$

$$\frac{\partial Loss}{\partial \hat y_{i}}= (- \frac{y_{i}}{p_{i}} + \frac{1 - y_{i} }{1 - p_{i}}) \cdot p_{i} \cdot (1 - p_{i}) = p_{i} - y_{i}$$

$$\frac{\partial^2 Loss}{\partial^2 \hat y_{i}}= p_{i} \cdot (1-p_{i})$$

实现lightgbm的`fobj`和`feval`如下：
```python
## sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

## 自定义损失函数需要提供损失函数的一阶和二阶导数形式
def loglikelood(preds, train_data):
    labels = train_data.get_label()
    preds = sigmoid(preds)
    grad = preds - labels
    hess = preds * (1. - preds)
    return grad, hess

## 自定义评估函数
def binary_error(preds, train_data):
    labels = train_data.get_label()
    preds = sigmoid(preds)
    return 'error', log_loss(labels, preds), False
```
二分类时，`binary_logloss`和`cross_entropy`数值计算上是一样的，这一点从lightgbm默认二分类的结果可以看出，如下：
```
[1]	training's auc: 0.980526	training's xentropy: 0.626414	training's binary_logloss: 0.626414	valid_1's auc: 0.974349	valid_1's xentropy: 0.608637	valid_1's binary_logloss: 0.608637
Training until validation scores don't improve for 10 rounds.
[2]	training's auc: 0.980526	training's xentropy: 0.590854	training's binary_logloss: 0.590854	valid_1's auc: 0.974349	valid_1's xentropy: 0.573687	valid_1's binary_logloss: 0.573687
[3]	training's auc: 0.98234	training's xentropy: 0.560036	training's binary_logloss: 0.560036	valid_1's auc: 0.983354	valid_1's xentropy: 0.545363	valid_1's binary_logloss: 0.545363
[4]	training's auc: 0.989436	training's xentropy: 0.530959	training's binary_logloss: 0.530959	valid_1's auc: 0.985775	valid_1's xentropy: 0.517253	valid_1's binary_logloss: 0.517253
[5]	training's auc: 0.988822	training's xentropy: 0.504266	training's binary_logloss: 0.504266	valid_1's auc: 0.985623	valid_1's xentropy: 0.490874	valid_1's binary_logloss: 0.490874
[6]	training's auc: 0.991023	training's xentropy: 0.479707	training's binary_logloss: 0.479707	valid_1's auc: 0.988574	valid_1's xentropy: 0.467473	valid_1's binary_logloss: 0.467473
[7]	training's auc: 0.992117	training's xentropy: 0.457022	training's binary_logloss: 0.457022	valid_1's auc: 0.99092	valid_1's xentropy: 0.445513	valid_1's binary_logloss: 0.445513
[8]	training's auc: 0.991757	training's xentropy: 0.436177	training's binary_logloss: 0.436177	valid_1's auc: 0.991828	valid_1's xentropy: 0.424863	valid_1's binary_logloss: 0.424863
[9]	training's auc: 0.991543	training's xentropy: 0.416976	training's binary_logloss: 0.416976	valid_1's auc: 0.992433	valid_1's xentropy: 0.405809	valid_1's binary_logloss: 0.405809
[10]	training's auc: 0.991517	training's xentropy: 0.39913	training's binary_logloss: 0.39913	valid_1's auc: 0.993341	valid_1's xentropy: 0.387673	valid_1's binary_logloss: 0.387673
Did not meet early stopping. Best iteration is:
[7]	training's auc: 0.992117	training's xentropy: 0.457022	training's binary_logloss: 0.457022	valid_1's auc: 0.99092	valid_1's xentropy: 0.445513	valid_1's binary_logloss: 0.445513
```
但是自定义的loss函数下，两者却有差异，而且自定义的eval函数，其数值与`binary_logloss`和`cross_entropy`也有差异，参考了lightgbm的[issue](https://github.com/microsoft/LightGBM/issues/3312)做了修改，但是结果还是有差异，待补充？


### 加权Logloss实例自定义损失函数

> 用lightgbm实现加权logloss，主要应用在短视频领域WCE建模时长或者某手树模型规则Ensemble融合，具体参考：
https://github.com/ShaoQiBNU/videoRecTips#wce%E5%8A%A0%E6%9D%83%E5%88%86%E7%B1%BBhttps://mp.weixin.qq.com/s/mxlecZpxXEoOe21UY_UCXQ

WCE损失函数为：
$$Loss= -\sum_{i=1}^N y_{i} \cdot log(p_{i}) + log(1-p_{i}) $$

$$p_{i} = sigmoid(\hat y_{i}) = \frac{1}{1+exp^{-\hat y_{i}}} $$

$$y_{i} 是正样本的权重，业务自定义$$

一阶导数和二阶导数为：

$$\frac{\partial p_{i}}{\partial \hat y_{i}}= (\frac{1}{1+exp^{-\hat y_{i}}})^{'} = \frac{exp^{-\hat y_{i}}}{(1+exp^{-\hat y_{i}})^{2}} = \frac{1 + exp^{-\hat y_{i}} -1}{(1+exp^{-\hat y_{i}})^{2}} = \frac{1}{1+exp^{-\hat y_{i}}} \cdot (1 - \frac{1}{1+exp^{-\hat y_{i}}}) = p_{i} \cdot (1 - p_{i})$$

$$\frac{\partial Loss}{\partial p_{i}}= - (\frac{y_{i}}{p_{i}} + \frac{-1}{1 - p_{i}}) = - \frac{y_{i}}{p_{i}} + \frac{1}{1 - p_{i}} $$

$$\frac{\partial Loss}{\partial \hat y_{i}}= (- \frac{y_{i}}{p_{i}} + \frac{1}{1 - p_{i}}) \cdot p_{i} \cdot (1 - p_{i}) = - y_{i} \cdot (1 - p_{i}) + p_{i}$$

$$\frac{\partial^2 Loss}{\partial^2 \hat y_{i}}= (y_{i} + 1) \cdot p_{i} \cdot (1-p_{i})$$

实现lightgbm的`fobj`和`feval`如下：
```python
## sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

## 自定义损失函数需要提供损失函数的一阶和二阶导数形式
def loglikelood(preds, train_data):
    labels = train_data.get_label()
    preds = sigmoid(preds)
    grad = -labels * (1 - preds) + preds
    hess = (labels + 1) * preds * (1 - preds)
    return grad, hess

## 自定义评估函数
def binary_error(preds, train_data):
    labels = train_data.get_label()
    preds = sigmoid(preds)
    return 'error', -np.average(labels * np.log(preds) + np.log(1 -preds)), False
```

> 数据集是乳腺癌的二分类数据集，因此样本权重采用随机整数的方式构造，为实现方便，将权重直接作为样本的label。由于树模型中复制样本作为负样本会对模型分类产生影响，因此采用修改loss函数的形式来实现，具体见：https://github.com/ShaoQiBNU/lightgbmTips/blob/main/lightgbm_wce.ipynb


参考：

https://www.showmeai.tech/article-detail/205

## lightgbm使用教程

https://www.showmeai.tech/article-detail/205

https://blog.csdn.net/luanpeng825485697/article/details/80236759

https://zhuanlan.zhihu.com/p/76206257