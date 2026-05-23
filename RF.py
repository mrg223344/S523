import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import pickle
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np
import joblib

os.getcwd() # 当前工作路径
plt.rcParams['font.family'] = 'Times New Roman'
train_data = pd.read_csv("traindata.csv", encoding="GBK")
val_data = pd.read_csv("valdata.csv", encoding="GBK")
print(train_data.info())
continuous_vars = train_data.iloc[:, 32:].columns.tolist()# 命名连续变量
print(train_data.info())

scaler = StandardScaler()
# 训练集连续变量标准化
scaler.fit(train_data[continuous_vars])
train_data[continuous_vars] = scaler.transform(train_data[continuous_vars])
train_data[continuous_vars] = train_data[continuous_vars].round(2)
# 导出标准化后的训练集
train_data.to_csv("1.标准化数据/train_data_scaled.csv")
# 验证集连续变量标准化
val_data[continuous_vars] = scaler.transform(val_data[continuous_vars])
val_data[continuous_vars] = val_data[continuous_vars].round(2)
# 导出标准化后的验证集
val_data.to_csv("1.标准化数据/val_data_scaled.csv")
joblib.dump(scaler, "1.标准化数据/scaler.joblib")

selected_features = ['Aneurysm_Overlap', 'Neck_Center_Distance', 'Parent_Artery_Overlap', 'Diameter']
target_label = 'Outcome'

# 定义训练集和验证集里的自变量和结局
train_data_ml = pd.read_csv("1.标准化数据/train_data_scaled4.csv", encoding="GBK", index_col=0)
val_data_ml = pd.read_csv("1.标准化数据/val_data_scaled4.csv", encoding="GBK", index_col=0)
X_train = train_data_ml[selected_features] # 批量定义自变量
y_train = train_data_ml[target_label]
X_val = val_data_ml[selected_features] # 批量定义自变量
y_val = val_data_ml[target_label]

rf_model_default = RandomForestClassifier(random_state = 123, oob_score = True)
rf_model_default.fit(X_train, y_train)
# 查看模型默认参数
print("模型默认参数:", pd.DataFrame.from_dict(rf_model_default.get_params(), orient='index'))
# 计算默认参数模型的验证集AUC
y_val_pred_prob_rfd = rf_model_default.predict_proba(X_val)[:, 1]
auc_rfd = roc_auc_score(y_val, y_val_pred_prob_rfd)
print("默认参数模型的验证集 AUC:", auc_rfd)

# 进行超参数网格搜索调优
# 1. 定义超参数搜索范围
param_grid = {
    'n_estimators': np.arange(50, 500, 50),  # 树的数量：50 到 450，每次增加 50
    #'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    #'max_features': ['sqrt']
    'max_features': list(range(2, round(np.sqrt(X_train.shape[1])) + 1))  # 每棵树的最大特征使用数：2 到 自变量数
}

# 2. 使用 GridSearchCV 进行网格搜索和 k 折交叉验证
grid_search_rf = GridSearchCV(
    estimator=rf_model_default,  # 使用之前定义的默认参数模型
    param_grid=param_grid,  # 使用之前定义的超参数网格
    scoring='roc_auc',  # 使用 AUC 作为评价指标
    cv=10,  # 5 折交叉验证
    n_jobs=-1,  # 并行计算
    verbose=1
)

grid_search_rf.fit(X_train, y_train)  # 在训练集上进行网格搜索

# 3. 输出最优超参数
best_auc_rf = grid_search_rf.best_score_  # 获取最佳模型的交叉验证 AUC
rf_model_best = grid_search_rf.best_estimator_  # 获取最佳模型
best_params_rf = grid_search_rf.best_params_  # 获取最佳超参数组合
best_ntree = best_params_rf['n_estimators']
best_mtry = best_params_rf['max_features']
print("最佳RF模型参数组合: n_estimators =", best_ntree, ", max_features =", best_mtry)
print("调优后RF模型详细参数:", pd.DataFrame.from_dict(rf_model_best.get_params(), orient='index'))
print("默认参数RF模型的验证集 AUC:", auc_rfd)
print("调优后RF模型最佳模型的交叉验证 AUC", best_auc_rf)
# 保存训练好的模型
with open("2.训练集构建模型/rf_model.pkl", 'wb') as f:
    pickle.dump(rf_model_best, f)
joblib.dump(rf_model_best, "2.训练集构建模型/rf_model.joblib")