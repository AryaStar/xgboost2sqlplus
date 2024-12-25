# XGBoost模型转sql语句工具包
现在是大数据量的时代，我们开发的模型要应用在特别大的待预测集上，使用单机的python，需要预测2、3天，甚至更久，中途很有可能中断。因此需要通过分布式的方式来预测。这个工具包就是实现了将训练好的python模型，转换成sql语句。将生成的sql语句可以放到大数据环境中进行分布式执行预测，能比单机的python预测快好几个量级


## 思想碰撞

本项目fork来自ZhengRyan的优秀项目xgboost2sql，相关链接如下：

> 仓库地址：https://github.com/ZhengRyan/xgboost2sql
> 
> 微信公众号文章：https://mp.weixin.qq.com/s/z3IjzMFKP7iEoag5KP6nAA
> 
> pipy包：https://pypi.org/project/xgboost2sql/


## 环境准备
可以不用单独创建虚拟环境，因为对包的依赖没有版本要求。本项目在xgboost2sql的基础上，对value is missing的情况进行fix，此外扩充了xgboost有类别（category）作为模型输入时的2sql处理

### `xgboost2sql` 安装
pip install（pip安装）

```bash
pip install xgboost2sql # to install
pip install -U xgboost2sql # to upgrade
```

Source code install（源码安装）

```bash
python setup.py install
```

添加代码相关内容

```bash
将xgboost2sql中的xgboost2sql.py中的内容替换为本项目中的xgboost2sqlplus.py中的所有内容
```

## 运行样例
###【注意：：：核验对比python模型预测出来的结果和sql语句预测出来的结果是否一致请查看教程代码】tutorial_code.ipynb。代码位置："https://github.com/ZhengRyan/xgboost2sql/tree/master/examples"

+ 导入相关依赖
```python
import xgboost as xgb 

from xgboost2sql import XGBoost2Sql
```

+ 训练1个xgboost二分类模型
```python
X, y = make_classification(n_samples=10000,
                           n_features=10,
                           n_informative=3,
                           n_redundant=2,
                           n_repeated=0,
                           n_classes=2,
                           weights=[0.7, 0.3],
                           flip_y=0.1,
                           random_state=1024)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1024)

###训练模型
model = xgb.XGBClassifier(n_estimators=3)
model.fit(X_train, y_train)
#xgb.to_graphviz(model)
```
+ 使用xgboost2sql工具包将模型转换成的sql语句
```python
xgb2sql = XGBoost2Sql()
sql_str = xgb2sql.transform(model)
```

+ 将sql语句保存
```python
xgb2sql.save()
```

+ 将sql语句打印出来
```python
print(sql_str)
```