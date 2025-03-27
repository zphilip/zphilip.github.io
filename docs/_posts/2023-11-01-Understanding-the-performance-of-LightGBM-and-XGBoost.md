---
layout: post
title: Understanding the performance of LightGBM and XGBoost
date: 2023-11-01
categories: coding
tags: AI LightGBM XGBoost
---

## Problem Statement: Predict the pay-range a given employee would belong to, based on the analysis of different attributes of other employees & their respective pay-range.

#### *The dataset is sourced from http://openbook.sfgov.org, which is part of the Government of San Francisco city's initiative in providing open & easily accessible data related to performance & spending across different departments of San Francisco City government. Salary range is the range of pay established by employers to pay to employees performing a particular job or function. Salary range generally has a minimum pay rate, a maximum pay rate, and a series of mid-range opportunities for pay increases.*

#### *The salary range is determined by market pay rates, organization, department, union, type and domain of jobs, established through market pay studies, for people doing similar work in similar industries in the same region of the country. Pay rates and salary ranges are also set up by individual employers and recognize the level of education, knowledge, skill, and experience needed to perform each job. Its a database of the salary and benefits paid to City employees since fiscal year 2013.*

#### *This data is summarized and presented on the Employee Compensation report hosted at http://openbook.sfgov.org*

#### *The target/label comprises of the 3 pay-range:* 
#### *1. Low range salary*
#### *2. Mid range salary*
#### *3. High range salary*

example is get from https://github.com/debajyotid/Understanding-the-performance-of-LightGBM-and-XGBoost

### *The objective of this notebook is to understand the impact of using Imblearn libraries on an imbalanced dataset*


```python
import timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
```


```python
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn import metrics
from sklearn import decomposition
```

#### *The dataset is given in 2 sets: train & test, wherein the test set doesn't have the labels marked. Thus the train set has to be split for actual training & validation*


```python
train=pd.read_csv('./data/train.csv')
test=pd.read_csv('./data/test.csv')
```


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 149087 entries, 0 to 149086
    Data columns (total 21 columns):
     #   Column                   Non-Null Count   Dtype  
    ---  ------                   --------------   -----  
     0   ID                       149087 non-null  int64  
     1   Year Type                149087 non-null  object 
     2   Year                     149087 non-null  int64  
     3   Organization Group Code  149087 non-null  int64  
     4   Organization Group       149087 non-null  object 
     5   Department Code          149087 non-null  object 
     6   Department               149087 non-null  object 
     7   Union Code               149087 non-null  int64  
     8   Union                    149087 non-null  object 
     9   Job Family Code          149087 non-null  object 
     10  Job Family               149087 non-null  object 
     11  Job Code                 149087 non-null  object 
     12  Job                      149087 non-null  object 
     13  Employee Identifier      149087 non-null  int64  
     14  Overtime                 149087 non-null  float64
     15  Other Salaries           149087 non-null  float64
     16  Retirement               149087 non-null  float64
     17  Health/Dental            149087 non-null  float64
     18  Other Benefits           149087 non-null  float64
     19  Total Benefits           149087 non-null  float64
     20  Class                    149087 non-null  int64  
    dtypes: float64(6), int64(6), object(9)
    memory usage: 23.9+ MB



```python
train['Class'].value_counts()
```




    3    50811
    2    49604
    1    48672
    Name: Class, dtype: int64



#### *There are altogether 19 features & 1 label distributed across approx. 150 thousand datapoints. Of these 19 features, some are string/object type while the others are numerical (integer or floating point). Most ML algorithms work well with numerical data, and non-numerical categorical data usually needs to be ENCODED into some numerical form or the other.* 
#### *Some of these features may be redundant, while others may be not so important. We will inspect them individually*
#### *Lastly the target variable comprises of 3 classes and the classes seem to be slightly imbalanced. We can try introducing synthetic methods like y_val & RandomUnderSampler to handle this imbalance, so that the model is capable of classifying all the 3 classes equally well*


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Year Type</th>
      <th>Year</th>
      <th>Organization Group Code</th>
      <th>Organization Group</th>
      <th>Department Code</th>
      <th>Department</th>
      <th>Union Code</th>
      <th>Union</th>
      <th>Job Family Code</th>
      <th>...</th>
      <th>Job Code</th>
      <th>Job</th>
      <th>Employee Identifier</th>
      <th>Overtime</th>
      <th>Other Salaries</th>
      <th>Retirement</th>
      <th>Health/Dental</th>
      <th>Other Benefits</th>
      <th>Total Benefits</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9248</td>
      <td>Fiscal</td>
      <td>2017</td>
      <td>3</td>
      <td>Human Welfare &amp; Neighborhood Development</td>
      <td>DSS</td>
      <td>HSA Human Services Agency</td>
      <td>535</td>
      <td>SEIU - Human Services, Local 1021</td>
      <td>2900</td>
      <td>...</td>
      <td>2905</td>
      <td>Senior Eligibility Worker</td>
      <td>41351</td>
      <td>0.00</td>
      <td>240.00</td>
      <td>11896.36</td>
      <td>13765.55</td>
      <td>5248.43</td>
      <td>30910.34</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44541</td>
      <td>Fiscal</td>
      <td>2014</td>
      <td>6</td>
      <td>General Administration &amp; Finance</td>
      <td>ASR</td>
      <td>ASR Assessor / Recorder</td>
      <td>21</td>
      <td>Prof &amp; Tech Engineers - Miscellaneous, Local 21</td>
      <td>4200</td>
      <td>...</td>
      <td>4222</td>
      <td>Sr Personal Property Auditor</td>
      <td>41792</td>
      <td>0.00</td>
      <td>400.00</td>
      <td>15429.94</td>
      <td>9337.37</td>
      <td>5599.01</td>
      <td>30366.32</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>47031</td>
      <td>Fiscal</td>
      <td>2014</td>
      <td>3</td>
      <td>Human Welfare &amp; Neighborhood Development</td>
      <td>DSS</td>
      <td>HSA Human Services Agency</td>
      <td>535</td>
      <td>SEIU - Human Services, Local 1021</td>
      <td>2900</td>
      <td>...</td>
      <td>2910</td>
      <td>Social Worker</td>
      <td>9357</td>
      <td>0.00</td>
      <td>1080.00</td>
      <td>9682.00</td>
      <td>8848.03</td>
      <td>3463.92</td>
      <td>21993.95</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>139416</td>
      <td>Fiscal</td>
      <td>2014</td>
      <td>1</td>
      <td>Public Protection</td>
      <td>FIR</td>
      <td>FIR Fire Department</td>
      <td>798</td>
      <td>Firefighters - Miscellaneous, Local 798</td>
      <td>H000</td>
      <td>...</td>
      <td>H002</td>
      <td>Firefighter</td>
      <td>28022</td>
      <td>25730.46</td>
      <td>18414.18</td>
      <td>24222.26</td>
      <td>13911.13</td>
      <td>2416.58</td>
      <td>40549.97</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>123780</td>
      <td>Fiscal</td>
      <td>2013</td>
      <td>2</td>
      <td>Public Works, Transportation &amp; Commerce</td>
      <td>MTA</td>
      <td>MTA Municipal Transprtn Agncy</td>
      <td>790</td>
      <td>SEIU - Miscellaneous, Local 1021</td>
      <td>1600</td>
      <td>...</td>
      <td>1224</td>
      <td>Pr Payroll &amp; Personnel Clerk</td>
      <td>51052</td>
      <td>1138.28</td>
      <td>2148.11</td>
      <td>15437.62</td>
      <td>12828.15</td>
      <td>7246.54</td>
      <td>35512.31</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



LabelEncoder是用来对分类型特征值进行编码，即对不连续的数值或文本进行编码。其中包含以下常用方法：

fit(y) ：fit可看做一本空字典，y可看作要塞到字典中的词。
fit_transform(y)：相当于先进行fit再进行transform，即把y塞到字典中去以后再进行transform得到索引值。
inverse_transform(y)：根据索引值y获得原始数据。
transform(y) ：将y转变成索引值。

LabelBinarizer
将对应的数据转换为二进制型，有点类似于onehot编码，这里有几点不同，LabelBinarizer可以处理数值型和类别型数据，输入必须为1D数组，可以自己设置正类和父类的表示方式

OneHotEncoder

有一些特征并不是以连续值的形式给出。例如：人的性别 [“male”, “female”]，来自的国家 [“from Europe”, “from US”, “from Asia”]，使用的浏览器[“uses Firefox”, “uses Chrome”, “uses Safari”, “uses Internet Explorer”]。这种特征可以采用整数的形式进行编码，如： [“male”, “from US”, “uses Internet Explorer”] 可表示成 [0, 1, 3] ，[“female”, “from Asia”, “uses Chrome”] 可表示成[1, 2, 1]。 但是，这些整数形式的表示不能直接作为某些机器学习算法输入，因为有些机器学习算法是需要连续型的输入数据，同一列数据之间数值的大小可代表差异程度。如： [0, 1, 3]与[0,1,0]的特征差异比[0, 1, 3]与[0,1,2]之间的差异要大，但事实上它们的差异是一样的，都是浏览器使用不一样。

一个解决办法就是采用OneHotEncoder，这种表示方式将每一个分类特征变量的m个可能的取值转变成m个二值特征，对于每一条数据这m个值中仅有一个特征值为1，其他的都为0。

> enc = preprocessing.OneHotEncoder()  
> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])  # 注意：第1、2、3列分别有2、3、4个可能的取值  
> OneHotEncoder(categorical_features='all', dtype=<... 'numpy.float64'>,  
>       handle_unknown='error', n_values='auto', sparse=True)  
> enc.transform([[0, 1, 3]]).toarray() #要对[0,1,3]进行编码  
array([[ 1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.]]) # [1,0]对应数值0，[0,1,0]对应数值1，[0,0,0,1]对应数值3  



```python
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()  
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
enc.transform([[0, 1, 3]]).toarray()
```




    array([[1., 0., 0., 1., 0., 0., 0., 0., 1.]])




```python
enc = OneHotEncoder()
lb = LabelEncoder()
tmp = lb.fit_transform([123,456,789])
print(tmp)#输出LabelEncoder的结果
enc.fit(tmp.reshape(-1,1))#将LabelEncoder的结果作为OneHotEncoder特征输入
x_train = enc.transform(lb.transform([123,789]).reshape(-1, 1))
#输出特征[123,789]的OneHotEncoder的编码结果
print(x_train)
```

    [0 1 2]
      (0, 0)	1.0
      (1, 2)	1.0



```python
label_encoder = LabelEncoder()
lbl_binarizer1 = LabelBinarizer()
lbl_binarizer2 = LabelBinarizer()
lbl_binarizer3 = LabelBinarizer()
train['Year']= label_encoder.fit_transform(train['Year'])
train['Department Code']= lbl_binarizer1.fit_transform(train['Department Code'])
train['Job Family Code']= lbl_binarizer2.fit_transform(train['Job Family Code'])
train['Job Code']= lbl_binarizer3.fit_transform(train['Job Code'])
```


```python
test['Year']= label_encoder.fit_transform(test['Year'])
test['Department Code']= lbl_binarizer1.transform(test['Department Code'])
test['Job Family Code']= lbl_binarizer2.transform(test['Job Family Code'])
test['Job Code']= lbl_binarizer3.transform(test['Job Code'])
```

#### *We first fit the encoder objects on the training data & then transform the same. The same object is then also used to transform the test data. As in this case we are using the train data for both training & validation, it would have been necessary to perform any transformative functions like scaling on the training data after splitting the same for training & testing. Fortunately such activity has not been needed due to our choice of ML algorithm, Tree based Classifiers - Random Forest & XGBClassifier, followed by stacking the 2*

#### *LabelBinarizer helps in generating one-hot encoded data from multi-class object/string data. LabelEncoder helps in generating ordered numeric feature by encoding ordinal numeric/object data*

DataFrame.corr(method='pearson', min_periods=1)

参数说明：
method：可选值为{‘pearson’, ‘kendall’, ‘spearman’}
pearson：Pearson相关系数来衡量两个数据集合是否在一条线上面，即针对线性数据的相关系数计算，针对非线性                                           数据便会有误差。
kendall：用于反映分类变量相关性的指标，即针对无序序列的相关系数，非正太分布的数据
spearman：非线性的，非正太分析的数据的相关系数
min_periods：样本最少的数据量
返回值：各类型之间的相关系数DataFrame表格。


```python
fig,ax = plt.subplots(figsize=(10, 10))   
sns.heatmap(train.corr(method='kendall'), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma") 
```




    <AxesSubplot:>




    
![png](/assets/2023-11-01-Understanding-the-performance-of-LightGBM-and-XGBoost_files/2023-11-01-Understanding-the-performance-of-LightGBM-and-XGBoost_17_1.png)
    


#### *From the above description we can see that the columns 'Organization Group','Department','Union','Job Family','Job','Employee Identifier','Year Type' are either redundant or are containing not important information. The same information as 'Organization Group','Department','Union','Job Family', and 'Job' is contained in 'Organization Group Code','Department Code','Union Code','Job Family Code',and 'Job Code' respectively. This data being available in numeric format is more suitable for analysis.*

#### *Additionally, information like 'Year Type','ID' and 'Employee Identifier' are either not unique for different employees or there is no visible strong correlation between the same & the target variable 'Class'. Thus we can drop them also.*


```python
train.drop(columns=['ID','Organization Group','Department','Union',
                    'Job Family','Job','Employee Identifier','Year Type'],axis=1,inplace=True)

test.drop(columns=['ID','Organization Group','Department','Union',
                   'Job Family','Job','Employee Identifier', 'Year Type'],axis=1,inplace=True)
```


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 149087 entries, 0 to 149086
    Data columns (total 13 columns):
     #   Column                   Non-Null Count   Dtype  
    ---  ------                   --------------   -----  
     0   Year                     149087 non-null  int64  
     1   Organization Group Code  149087 non-null  int64  
     2   Department Code          149087 non-null  int64  
     3   Union Code               149087 non-null  int64  
     4   Job Family Code          149087 non-null  int64  
     5   Job Code                 149087 non-null  int64  
     6   Overtime                 149087 non-null  float64
     7   Other Salaries           149087 non-null  float64
     8   Retirement               149087 non-null  float64
     9   Health/Dental            149087 non-null  float64
     10  Other Benefits           149087 non-null  float64
     11  Total Benefits           149087 non-null  float64
     12  Class                    149087 non-null  int64  
    dtypes: float64(6), int64(7)
    memory usage: 14.8 MB



```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Organization Group Code</th>
      <th>Department Code</th>
      <th>Union Code</th>
      <th>Job Family Code</th>
      <th>Job Code</th>
      <th>Overtime</th>
      <th>Other Salaries</th>
      <th>Retirement</th>
      <th>Health/Dental</th>
      <th>Other Benefits</th>
      <th>Total Benefits</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>535</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>240.00</td>
      <td>11896.36</td>
      <td>13765.55</td>
      <td>5248.43</td>
      <td>30910.34</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>21</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>400.00</td>
      <td>15429.94</td>
      <td>9337.37</td>
      <td>5599.01</td>
      <td>30366.32</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>535</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>1080.00</td>
      <td>9682.00</td>
      <td>8848.03</td>
      <td>3463.92</td>
      <td>21993.95</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>798</td>
      <td>0</td>
      <td>0</td>
      <td>25730.46</td>
      <td>18414.18</td>
      <td>24222.26</td>
      <td>13911.13</td>
      <td>2416.58</td>
      <td>40549.97</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>790</td>
      <td>0</td>
      <td>0</td>
      <td>1138.28</td>
      <td>2148.11</td>
      <td>15437.62</td>
      <td>12828.15</td>
      <td>7246.54</td>
      <td>35512.31</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



kdeplot(核密度估计图)

核密度估计(kernel density estimation)是在概率论中用来估计未知的密度函数，属于非参数检验方法之一。通过核密度估计图可以比较直观的看出数据样本本身的分布特征。具体用法如下：


```python
plt.figure(figsize=(20,5))
plt.legend()
plt.xlabel('Overtime')
sns.kdeplot(train[train['Class']==1]['Overtime'],label='Low Range',shade=True)
sns.kdeplot(train[train['Class']==2]['Overtime'],label='Mid Range',shade=True)
sns.kdeplot(train[train['Class']==3]['Overtime'],label='High Range',shade=True)
```

    No handles with labels found to put in legend.





    <AxesSubplot:xlabel='Overtime', ylabel='Density'>




    
![png](/assets/2023-11-01-Understanding-the-performance-of-LightGBM-and-XGBoost_files/2023-11-01-Understanding-the-performance-of-LightGBM-and-XGBoost_23_2.png)
    



```python
plt.figure(figsize=(20,5))
plt.legend()
plt.xlabel('Other Salaries')
sns.kdeplot(train[train['Class']==1]['Other Salaries'],label='Low Range', shade=True)
sns.kdeplot(train[train['Class']==2]['Other Salaries'],label='Mid Range', shade=True)
sns.kdeplot(train[train['Class']==3]['Other Salaries'],label='High Range', shade=True)
```

    No handles with labels found to put in legend.





    <AxesSubplot:xlabel='Other Salaries', ylabel='Density'>




    
![png](/assets/2023-11-01-Understanding-the-performance-of-LightGBM-and-XGBoost_files/2023-11-01-Understanding-the-performance-of-LightGBM-and-XGBoost_24_2.png)
    



```python
plt.figure(figsize=(20,5))
plt.legend()
plt.xlabel('Retirement')
sns.kdeplot(train[train['Class']==1]['Retirement'],label='Low Range', shade=True)
sns.kdeplot(train[train['Class']==2]['Retirement'],label='Mid Range', shade=True)
sns.kdeplot(train[train['Class']==3]['Retirement'],label='High Range', shade=True)
```

    No handles with labels found to put in legend.





    <AxesSubplot:xlabel='Retirement', ylabel='Density'>




    
![png](/assets/2023-11-01-Understanding-the-performance-of-LightGBM-and-XGBoost_files/2023-11-01-Understanding-the-performance-of-LightGBM-and-XGBoost_25_2.png)
    



```python
plt.figure(figsize=(20,5))
plt.legend()
plt.xlabel('Health/Dental')
sns.kdeplot(train[train['Class']==1]['Health/Dental'],label='Low Range', shade=True)
sns.kdeplot(train[train['Class']==2]['Health/Dental'],label='Mid Range', shade=True)
sns.kdeplot(train[train['Class']==3]['Health/Dental'],label='High Range', shade=True)
```

    No handles with labels found to put in legend.





    <AxesSubplot:xlabel='Health/Dental', ylabel='Density'>




    
![png](/assets/2023-11-01-Understanding-the-performance-of-LightGBM-and-XGBoost_files/2023-11-01-Understanding-the-performance-of-LightGBM-and-XGBoost_26_2.png)
    



```python
plt.figure(figsize=(20,5))
plt.legend()
plt.xlabel('Other Benefits')
sns.kdeplot(train[train['Class']==1]['Other Benefits'],label='Low Range', shade=True)
sns.kdeplot(train[train['Class']==2]['Other Benefits'],label='Mid Range', shade=True)
sns.kdeplot(train[train['Class']==3]['Other Benefits'],label='High Range', shade=True)
```

    No handles with labels found to put in legend.





    <AxesSubplot:xlabel='Other Benefits', ylabel='Density'>




    
![png](/assets/2023-11-01-Understanding-the-performance-of-LightGBM-and-XGBoost_files/2023-11-01-Understanding-the-performance-of-LightGBM-and-XGBoost_27_2.png)
    



```python
plt.figure(figsize=(20,5))
plt.legend()
plt.xlabel('Total Benefits')
sns.kdeplot(train[train['Class']==1]['Total Benefits'],label='Low Range', shade=True)
sns.kdeplot(train[train['Class']==2]['Total Benefits'],label='Mid Range', shade=True)
sns.kdeplot(train[train['Class']==3]['Total Benefits'],label='High Range', shade=True)
```

    No handles with labels found to put in legend.





    <AxesSubplot:xlabel='Total Benefits', ylabel='Density'>




    
![png](/assets/2023-11-01-Understanding-the-performance-of-LightGBM-and-XGBoost_files/2023-11-01-Understanding-the-performance-of-LightGBM-and-XGBoost_28_2.png)
    


#### *It seems from the above EDA that Low Range employees receive mostly no Retirement, Other Salaries and Other Benefits. Thus oevrall amount of benefits received by Low Range earners is very less*

#### *In case of Mid Range earners, they seem to be the largest earners of Overtime. Their Other Salaries mostly seem to be zero, same as Low Range & High Range earners. Retirement benefits is less than High Range earners, but there is significant over-lap between the higher range of Retirement benefits of Mid Range earners & lower-range of Retirement benefits of High Range earners. Health/Dental benefits are identical with that of High Range earners while Other Benefits component is significantly higher for Mid Range earners, as compared to the other 2 class*

#### *High-Range earners mostly don't earn any Overtime or Other Salaries. As already seen, they earn slightly better Other Benefits & Retirals than Mid-Range earners, with whom they share similar Health/Dental benefits*


```python
#Dropping the target column 'Class' from the train data & storing it into a new variable
y = train['Class']
train.drop('Class',axis=1,inplace=True)
```

#### *Splitting the train data into 2 sets for training & validation by a 70:30 split (test_size=0.3). Further this split will be done after ensuring that the features are distributed in the same proportions in the 2 datasets, i.e. if if the 3 classes were initially distributed in the main dataset in a 30:40:30 ratio, the same will be maintained in the x_train & x_val also. This is achieved by the 'stratify' command, although it may not provide useful in the current context as we will be employing Up-sampling & Down-sampling techniques.*
#### *The random_state ensures reproducability over multiple execution cycles*


```python
x_train,x_val,y_train,y_val = train_test_split(train,y,test_size=0.3,random_state=42,stratify=y)
```


```python
y_train.value_counts()
```




    3    35567
    2    34723
    1    34070
    Name: Class, dtype: int64




```python
y_train = y_train - 1
y_val = y_val - 1
```


```python
model = XGBClassifier(random_state=42, 
                      booster='gbtree', 
                      objective='multi:softmax', 
                      num_class=3, 
                      n_jobs=-1, 
                      learning_rate=0.3,            #default 
                      n_estimators=200, 
                      max_depth=6,                  #default
                      subsample=0.8,
                      tree_method='hist',
                      grow_policy='lossguide',
                      max_leaves=10,
                      max_bin=256,                  #default
                      eval_metric='mlogloss')
```


```python
x_train.shape
```




    (104360, 12)




```python
start_time = timeit.default_timer()
model.fit(x_train,y_train)
print ('\n','Training Accuracy is: ',model.score(x_train,y_train))
stop_time = timeit.default_timer()
print('\n','Total Training Time: {time} seconds.'.format(time=round(stop_time - start_time, 2)))

start_time = timeit.default_timer()
y_pred_test = model.predict(x_val)
stop_time = timeit.default_timer()
print ('\n','Testing Accuracy is: ',metrics.accuracy_score(y_val,y_pred_test))
print('\n','Total Testing Time: {time} seconds.'.format(time=round(stop_time - start_time, 2)))
```

    
     Training Accuracy is:  0.9930816404752779
    
     Total Training Time: 4.27 seconds.
    
     Testing Accuracy is:  0.9871442305542514
    
     Total Testing Time: 0.18 seconds.



```python
#Viewing the classification report
print (metrics.classification_report(y_val,y_pred_test))
```

                  precision    recall  f1-score   support
    
               0       0.99      0.99      0.99     14602
               1       0.98      0.98      0.98     14881
               2       0.99      0.99      0.99     15244
    
        accuracy                           0.99     44727
       macro avg       0.99      0.99      0.99     44727
    weighted avg       0.99      0.99      0.99     44727
    



```python
model = LGBMClassifier(random_state=42, 
                      boosting_type='rf',
                      num_leaves=31,                #default
                      max_depth=-1,                 #default
                      learning_rate=0.3,
                      n_estimators=200,             
                      objective='multiclass', 
                      class_weight='balanced',      #to handle class imbalance
                      subsample=0.8,                #bagging ratio
                      subsample_freq=5,             #perform bagging at every 5th iteration
                      max_bin=255,                  #default
                      metric='multi_logloss'
                      )
```


```python
start_time = timeit.default_timer()
model.fit(x_train,y_train)
print ('\n','Training Accuracy is: ',model.score(x_train,y_train))
stop_time = timeit.default_timer()
print('\n','Total Training Time: {time} seconds.'.format(time=round(stop_time - start_time, 2)))

start_time = timeit.default_timer()
y_pred_test = model.predict(x_val)
stop_time = timeit.default_timer()
print ('\n','Testing Accuracy is: ',metrics.accuracy_score(y_val,y_pred_test))
print('\n','Total Testing Time: {time} seconds.'.format(time=round(stop_time - start_time, 2)))
```

    
     Training Accuracy is:  0.9659448064392487
    
     Total Training Time: 4.61 seconds.
    
     Testing Accuracy is:  0.9639144141122812
    
     Total Testing Time: 0.3 seconds.



```python
#Viewing the classification report
print (metrics.classification_report(y_val,y_pred_test))
```

                  precision    recall  f1-score   support
    
               0       0.98      0.98      0.98     14602
               1       0.95      0.95      0.95     14881
               2       0.97      0.97      0.97     15244
    
        accuracy                           0.96     44727
       macro avg       0.96      0.96      0.96     44727
    weighted avg       0.96      0.96      0.96     44727
    


### Using PCA to verify if model accuracy can be increased. Later, we will try imbalanced learning on PCA data also


```python
train=pd.read_csv('./data/train.csv')
test=pd.read_csv('./data/test.csv')
```


```python
label_encoder = LabelEncoder()
lbl_binarizer1 = LabelBinarizer()
lbl_binarizer2 = LabelBinarizer()
lbl_binarizer3 = LabelBinarizer()
train['Year']= label_encoder.fit_transform(train['Year'])
train['Department Code']= lbl_binarizer1.fit_transform(train['Department Code'])
train['Job Family Code']= lbl_binarizer2.fit_transform(train['Job Family Code'])
train['Job Code']= lbl_binarizer3.fit_transform(train['Job Code'])
```


```python
test['Year']= label_encoder.fit_transform(test['Year'])
test['Department Code']= lbl_binarizer1.transform(test['Department Code'])
test['Job Family Code']= lbl_binarizer2.transform(test['Job Family Code'])
test['Job Code']= lbl_binarizer3.transform(test['Job Code'])
```


```python
fig,ax = plt.subplots(figsize=(10, 10))   
sns.heatmap(train.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma") 
```




    <AxesSubplot:>




    
![png](/assets/2023-11-01-Understanding-the-performance-of-LightGBM-and-XGBoost_files/2023-11-01-Understanding-the-performance-of-LightGBM-and-XGBoost_48_1.png)
    



```python
train[['Retirement','Health/Dental','Other Benefits','Total Benefits']].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Retirement</th>
      <th>Health/Dental</th>
      <th>Other Benefits</th>
      <th>Total Benefits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Retirement</th>
      <td>1.000000</td>
      <td>0.787526</td>
      <td>0.671137</td>
      <td>0.962205</td>
    </tr>
    <tr>
      <th>Health/Dental</th>
      <td>0.787526</td>
      <td>1.000000</td>
      <td>0.606259</td>
      <td>0.888821</td>
    </tr>
    <tr>
      <th>Other Benefits</th>
      <td>0.671137</td>
      <td>0.606259</td>
      <td>1.000000</td>
      <td>0.796860</td>
    </tr>
    <tr>
      <th>Total Benefits</th>
      <td>0.962205</td>
      <td>0.888821</td>
      <td>0.796860</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### *Looking to run PCA on the above columns which have high collinearity*


```python
plt.figure(figsize=(30,10))
sns.boxplot(data=train[['Retirement','Health/Dental','Other Benefits','Total Benefits']])
```




    <AxesSubplot:>




    
![png](/assets/2023-11-01-Understanding-the-performance-of-LightGBM-and-XGBoost_files/2023-11-01-Understanding-the-performance-of-LightGBM-and-XGBoost_51_1.png)
    



```python
scaler=StandardScaler()
pca=decomposition.PCA(n_components=3)
```


```python
columns=['Retirement','Health/Dental','Other Benefits','Total Benefits']
data_scaled = pd.DataFrame(scaler.fit_transform(train[['Retirement','Health/Dental','Other Benefits','Total Benefits']]),columns=columns)
```


```python
plt.figure(figsize=(30,10))
sns.boxplot(data=data_scaled)
```




    <AxesSubplot:>




    
![png](/assets/2023-11-01-Understanding-the-performance-of-LightGBM-and-XGBoost_files/2023-11-01-Understanding-the-performance-of-LightGBM-and-XGBoost_54_1.png)
    



```python
# PCA
# Step 1 - Create covariance matrix
cov_matrix = np.cov(data_scaled.T)
# Step 2- Get eigen values and eigen vector
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
#Step 3- Understanding cumulative variance
tot = sum(eig_vals)
var_exp = [( i /tot ) * 100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print("Cumulative Variance Explained", cum_var_exp)
```

    Cumulative Variance Explained [ 84.25437992  94.72404974 100.         100.        ]



```python
pcadata_reduced = pca.fit_transform(data_scaled)
```


```python
proj_data_df = pd.DataFrame(pcadata_reduced)  # converting array to dataframe for pairplot
pca_df = proj_data_df.join(train)             # adding back the PCA to the main dataset
```


```python
pca_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>ID</th>
      <th>Year Type</th>
      <th>Year</th>
      <th>Organization Group Code</th>
      <th>Organization Group</th>
      <th>Department Code</th>
      <th>Department</th>
      <th>...</th>
      <th>Job Code</th>
      <th>Job</th>
      <th>Employee Identifier</th>
      <th>Overtime</th>
      <th>Other Salaries</th>
      <th>Retirement</th>
      <th>Health/Dental</th>
      <th>Other Benefits</th>
      <th>Total Benefits</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.624622</td>
      <td>-0.327289</td>
      <td>-0.689272</td>
      <td>9248</td>
      <td>Fiscal</td>
      <td>4</td>
      <td>3</td>
      <td>Human Welfare &amp; Neighborhood Development</td>
      <td>0</td>
      <td>HSA Human Services Agency</td>
      <td>...</td>
      <td>0</td>
      <td>Senior Eligibility Worker</td>
      <td>41351</td>
      <td>0.00</td>
      <td>240.00</td>
      <td>11896.36</td>
      <td>13765.55</td>
      <td>5248.43</td>
      <td>30910.34</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.403335</td>
      <td>0.081200</td>
      <td>0.149623</td>
      <td>44541</td>
      <td>Fiscal</td>
      <td>1</td>
      <td>6</td>
      <td>General Administration &amp; Finance</td>
      <td>0</td>
      <td>ASR Assessor / Recorder</td>
      <td>...</td>
      <td>0</td>
      <td>Sr Personal Property Auditor</td>
      <td>41792</td>
      <td>0.00</td>
      <td>400.00</td>
      <td>15429.94</td>
      <td>9337.37</td>
      <td>5599.01</td>
      <td>30366.32</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.463135</td>
      <td>-0.163763</td>
      <td>-0.155547</td>
      <td>47031</td>
      <td>Fiscal</td>
      <td>1</td>
      <td>3</td>
      <td>Human Welfare &amp; Neighborhood Development</td>
      <td>0</td>
      <td>HSA Human Services Agency</td>
      <td>...</td>
      <td>0</td>
      <td>Social Worker</td>
      <td>9357</td>
      <td>0.00</td>
      <td>1080.00</td>
      <td>9682.00</td>
      <td>8848.03</td>
      <td>3463.92</td>
      <td>21993.95</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.275232</td>
      <td>-1.284027</td>
      <td>0.376024</td>
      <td>139416</td>
      <td>Fiscal</td>
      <td>1</td>
      <td>1</td>
      <td>Public Protection</td>
      <td>0</td>
      <td>FIR Fire Department</td>
      <td>...</td>
      <td>0</td>
      <td>Firefighter</td>
      <td>28022</td>
      <td>25730.46</td>
      <td>18414.18</td>
      <td>24222.26</td>
      <td>13911.13</td>
      <td>2416.58</td>
      <td>40549.97</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.098471</td>
      <td>0.089262</td>
      <td>-0.367685</td>
      <td>123780</td>
      <td>Fiscal</td>
      <td>0</td>
      <td>2</td>
      <td>Public Works, Transportation &amp; Commerce</td>
      <td>0</td>
      <td>MTA Municipal Transprtn Agncy</td>
      <td>...</td>
      <td>0</td>
      <td>Pr Payroll &amp; Personnel Clerk</td>
      <td>51052</td>
      <td>1138.28</td>
      <td>2148.11</td>
      <td>15437.62</td>
      <td>12828.15</td>
      <td>7246.54</td>
      <td>35512.31</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
data_scaled_test = pd.DataFrame(scaler.transform(test[['Retirement','Health/Dental','Other Benefits','Total Benefits']]),columns=columns)
pcadata_reduced_test = pca.transform(data_scaled_test)
```


```python
proj_data_df_test = pd.DataFrame(pcadata_reduced_test)  # converting array to dataframe for pairplot
pca_df_test = proj_data_df_test.join(test) 
```


```python
pca_df.drop(columns=['ID','Organization Group','Department','Union',
                    'Job Family','Job','Employee Identifier','Year Type',
                     'Retirement','Health/Dental','Other Benefits','Total Benefits'],axis=1,inplace=True)
pca_df_test.drop(columns=['ID','Organization Group','Department','Union',
                    'Job Family','Job','Employee Identifier','Year Type',
                     'Retirement','Health/Dental','Other Benefits','Total Benefits'],axis=1,inplace=True)
```


```python
pca_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>Year</th>
      <th>Organization Group Code</th>
      <th>Department Code</th>
      <th>Union Code</th>
      <th>Job Family Code</th>
      <th>Job Code</th>
      <th>Overtime</th>
      <th>Other Salaries</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.624622</td>
      <td>-0.327289</td>
      <td>-0.689272</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>535</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>240.00</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.403335</td>
      <td>0.081200</td>
      <td>0.149623</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>21</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>400.00</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.463135</td>
      <td>-0.163763</td>
      <td>-0.155547</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>535</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>1080.00</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.275232</td>
      <td>-1.284027</td>
      <td>0.376024</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>798</td>
      <td>0</td>
      <td>0</td>
      <td>25730.46</td>
      <td>18414.18</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.098471</td>
      <td>0.089262</td>
      <td>-0.367685</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>790</td>
      <td>0</td>
      <td>0</td>
      <td>1138.28</td>
      <td>2148.11</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
y = pca_df['Class']
pca_df.drop('Class',axis=1,inplace=True)
```


```python
x_train,x_val,y_train,y_val = train_test_split(pca_df,y,test_size=0.3,random_state=42,stratify=y)
```


```python
model2 = LGBMClassifier(random_state=42, 
                      boosting_type='rf',
                      num_leaves=31,                #default
                      max_depth=-1,                 #default
                      learning_rate=0.3,
                      n_estimators=200,             
                      objective='multiclass', 
                      class_weight='balanced',      #to handle class imbalance
                      subsample=0.8,                #bagging ratio
                      subsample_freq=5,             #perform bagging at every 5th iteration
                      max_bin=255,                  #default
                      metric='multi_logloss')
```


```python
start_time = timeit.default_timer()
model2.fit(x_train,y_train)
print ('\n','Training Accuracy is: ',model2.score(x_train,y_train))
stop_time = timeit.default_timer()
print('\n','Total Training Time: {time} seconds.'.format(time=round(stop_time - start_time, 2)))

start_time = timeit.default_timer()
y_pred_test = model2.predict(x_val)
stop_time = timeit.default_timer()
print ('\n','Testing Accuracy is: ',metrics.accuracy_score(y_val,y_pred_test))
print('\n','Total Testing Time: {time} seconds.'.format(time=round(stop_time - start_time, 2)))
```

    
     Training Accuracy is:  0.9529225756995017
    
     Total Training Time: 4.25 seconds.
    
     Testing Accuracy is:  0.9506785610481365
    
     Total Testing Time: 0.33 seconds.



```python
#Viewing the classification report
print (metrics.classification_report(y_val,y_pred_test))
```

                  precision    recall  f1-score   support
    
               1       0.96      0.97      0.97     14602
               2       0.93      0.92      0.93     14881
               3       0.96      0.96      0.96     15244
    
        accuracy                           0.95     44727
       macro avg       0.95      0.95      0.95     44727
    weighted avg       0.95      0.95      0.95     44727
    


### *LightGBM provides a much faster throughput than XGBoost. This is even more accentuated when some of the collinear features are replaced with equivalent Principal Components, with both training time & model accuracy improving considerably*

From my run , there are no major difference in time betwen XgBoost and LightGBM, accuracy Xgboost is better....I guess there are some improvement in the xgboost in the sklearn.
