import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
DATA_DIR = '../db'
data_file = os.path.join(DATA_DIR, 'train.tsv')
data = pd.read_csv(data_file)

# df[df.isnull().any(axis=1)] 一行ごとにboolearnの有無を調べる
# df[].fillna(df[]/(df[] - df[]), inplace=True)
# df.loc[[], :]
# df.duplicated()重複の確認
# df.drop_duplicates(inplace=True) #重複を削除
# df.reset_index(drop=True,inplace=True) #順序整理、インデックス作成
# df.describe(include='all')
# df[].value_counts() # count
# counts.plot(kind='bar')
# plt.tight_layout()
# pd.concat(df, df,axis=1) # 結合
# df.query("カラム名==0")

# lambda x: 1 if(x == ) else 2
# df.apply(処理) # 処理を適用

#bin分割
#X_cut, bin_indice = pd.cut(X, bins= (num or []), retbins=True)
#X_cut = pd.cut(X, bins= (num or []), labels=False)
# pd.get_dummies(X_cut)
