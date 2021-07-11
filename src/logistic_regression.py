import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
DATA_DIR = '../db'
data_file = os.path.join(DATA_DIR, 'train.tsv')
data = pd.read_csv(data_file)

from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# model.predict_proba(X_test) # 確率
# model.predict_proba(X_test)[: 1] # １が出る確率
from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_true= ,y_pred=  )
# df_cm = pd.DataFrame(np.rot90(cm, 2), index=[], columns=[])
# sns.heatmap(df_cm)
from sklearn.metrics import roc_auc_score, roc_curve
# auc_score = roc_auc_score(y_true= , y_score= )
# fpr, tpr, thresholds = roc_curve(y_true= , y_score= )
# plt.plot(fpr, tpr, label='roc curve (area = %0.3f)' % auc_score) # ROC曲線
from sklearn.preprocessing import PolynomialFeatures
# polynomial = PolynomialFeatures(degree= , include_bias=False)
# polynomial_result = plynomial.fit_transform(data)
from sklearn.feature_selection import SelectFromModel
# fs_model = LogisticRegression(penalty={}, random_state=0)
# fs_threshold = {}
# selector = SelectFromModel(fs_model, threshold=fs_threshold)
# selector.fit(X, y)
# selector.get_support() # 重要か否か
# X.loc[:, 変数]