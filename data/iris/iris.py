from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
df = pd.DataFrame(iris.data)
df.columns = iris.feature_names
df.to_csv('iris_data.csv')
df['target'] = iris.target
df.to_csv('iris.csv')