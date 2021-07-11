import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

import csv

# import seaborn as sns
DATA_DIR = '../db'
train_file = os.path.join(DATA_DIR, 'train.csv')
train = pd.read_csv(train_file)

# 前処理
train["AG_ratio"].fillna(train["Alb"] / (train["TP"] - train["Alb"]), inplace=True)
train.drop_duplicates(inplace=True)
train.reset_index(drop=True, inplace=True)
train["Gender"] = train["Gender"].apply(lambda x: 1 if x=="Male" else 0)
train = train.drop(["id"], axis=1)
X = train.drop(["disease"], axis=1)
y = train["disease"]
X_target = X.drop(["Gender"], axis=1)
# print(train.duplicated())
print(X.info())
# print(train.dropna())
# train_2 = train.dropna().reset_index(drop=True,inplace=True)
# print(train_2)
# X_train, X_test, y_train, y_test = train_test_split(X_target, y, random_state=1, test_size=0.2)
# model = LogisticRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict_proba(X_test) # 確率
# y_pred_a = model.predict_proba(X_test)[:, 1] # １が出る確率
# # cm = confusion_matrix(y_true= ,y_pred=  )
# # df_cm = pd.DataFrame(np.rot90(cm, 2), index=[], columns=[])
# # sns.heatmap(df_cm)
# # print(y_pred_a)
# auc_score = roc_auc_score(y_true=y_test, y_score=y_pred_a)
# print(auc_score)

# fpr, tpr, thresholds = roc_curve(y_true=y_test , y_score=y_pred_a) 
# plt.plot(fpr, tpr, label='roc curve (area = %0.3f)' % auc_score) # ROC曲線

# plt.show()


polynomial = PolynomialFeatures(degree=2 , include_bias=False)
polynomial_arr = polynomial.fit_transform(X_target)
X_polynomial = pd.DataFrame(polynomial_arr, columns=["poly" + str(x) for x in range(polynomial_arr.shape[1])])

fs_model = LogisticRegression(penalty='l2', random_state=0)
fs_threshold = 'mean'
# mask = selector.get_support()

# selector.fit(X_train, y_train)
selector = SelectFromModel(fs_model, threshold=fs_threshold)

# selector.get_support() # 重要か否か
# X.loc[:, 変数]

# 特徴量選択の実行
selector.fit(X_polynomial, y)
mask = selector.get_support()

# 選択された特徴量だけのサンプル取得
X_polynomial_masked = X_polynomial.loc[:, mask]

# 学習用・評価用データの分割（元の説明変数Xの代わりに、特徴量選択後のX_polynomial_maskedを使う）
X_train, X_test, y_train, y_test = train_test_split(X_polynomial_masked, y, test_size=0.2, random_state=1)
print(X_polynomial_masked)
# モデルの学習・予測
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]

# ROC曲線の描画（偽陽性率、真陽性率、閾値の算出）
fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
plt.plot(fpr, tpr, label='roc curve')
plt.plot([0, 1], [0, 1], linestyle=':', label='random')
plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', label='ideal')
plt.legend()
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
# plt.show()

# AUCスコアの算出
auc_score = roc_auc_score(y_true=y_test, y_score=y_pred)
print("AUC:", auc_score)

# plt.show()
test_file = os.path.join(DATA_DIR, 'test.csv')
test = pd.read_csv(test_file)
test["AG_ratio"].fillna(test["Alb"] / (test["TP"] - test["Alb"]), inplace=True)
test.drop_duplicates(inplace=True)
test.reset_index(drop=True, inplace=True)
test["Gender"] = test["Gender"].apply(lambda x: 1 if x=="Male" else 0)
# train = train.drop(["id"], axis=1)
z = test.drop(["id"], axis=1)
# y = train["disease"]
z_target = z.drop(["Gender"], axis=1)
print(z_target)
polynomial_z = polynomial.fit_transform(z_target)

z_polynomial = pd.DataFrame(polynomial_z, columns=["poly" + str(x) for x in range(polynomial_z.shape[1])])
z_polynomial_masked = z_polynomial.loc[:, mask]
z_pred = model.predict(z_polynomial_masked)

with open('result.csv','w') as f:
  writer = csv.writer(f)
  # print(len(z_pred))
  j = 0
  for i in range(len(z_pred)):
    try:
      # print()
      k = test.at[j, 'id']
    except:
      j += 1
      # print()
      k = test.at[j, 'id']
    # print()
    l = z_pred[i]
    writer.writerow([k, l])
    j += 1

