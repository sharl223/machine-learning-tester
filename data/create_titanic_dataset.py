import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

print("タイタニックのデータセットをダウンロード・処理しています...")

# 1. データの読み込み
# seabornからタイタニックのデータセットを読み込む
# この時点での目的変数名は 'survived' (小文字)
df = sns.load_dataset('titanic')

# 2. 簡単な前処理
# 不要な列を削除
df = df.drop(columns=['who', 'adult_male', 'deck', 'embark_town', 'alive', 'class'])

# 欠損値の補完 (inplace=True を使わない書き方に修正)
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# カテゴリカル変数のダミー変数化
df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True, dtype=float)

# 3. データの分割
# 元の目的変数名 'survived' (小文字) を指定して分割
train_df, predict_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['survived'] # 'Survived' (大文字) から 'survived' (小文字) に修正
)

# 4. CSVファイルとして保存
# インデックスをリセットせずに、元の 'passengerid' などをインデックスとして保存
train_df.to_csv('titanic_train.csv')
predict_df.to_csv('titanic_predict.csv')

print("titanic_train.csv と titanic_predict.csv を作成しました。")