#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'SVM'))
	print(os.getcwd())
except:
	pass

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# # 数据描述
#%% [markdown]
# 1. variance of Wavelet Transformed image (continuous) 小波变换图像的偏度
# 2. skewness of Wavelet Transformed image (continuous) 图像的方差
# 3. curtosis of Wavelet Transformed image (continuous) 图像的熵
# 4. entropy of image (continuous) 图像的曲率
# 5. class (integer) 
#%% [markdown]
# # 导入数据

#%%
bankdata = pd.read_csv('../dataset/bill_authentication.csv')

#%% [markdown]
# # 探索性数据分析

#%%
bankdata.shape


#%%
bankdata.head()

#%% [markdown]
# # 数据预处理

#%%
X = bankdata.drop('Class', axis=1)
y = bankdata['Class']


#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#%% [markdown]
# # 算法训练

#%%
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

#%% [markdown]
# # 做预测

#%%
y_pred = svclassifier.predict(X_test)

#%% [markdown]
# # 算法评价

#%%
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#%% [markdown]
# ![Confusion_matrix](http://www.ycc.idv.tw/media/mechine_learning_measure/mechine_learning_measure.001.jpeg)

