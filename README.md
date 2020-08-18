# Decision-Trees

import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\91709\Downloads\Iris.csv")

df.head()

df.shape

x = df.drop('Species',axis=1)
y = df['Species']

train = df[1:100]
test = df[101:]

x_train = x
y_train = y

x_test = x.loc[101:]
y_test = y.loc[101:]

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(x_train,y_train)

pred = dtc.predict(x_test)

pred

dtc.score(x_train,y_train)

dtc.score(x_test,y_test)

dtc.score(x_test,pred)

!pip install pydotplus
!apt-get install graphviz -y

import sklearn.datasets as datasets
iris=datasets.load_iris()
df=pd.DataFrame(iris.data, columns=iris.feature_names)
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

data = StringIO()
export_graphviz(dtc, out_file=data, feature_names=iris.feature_names,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_data(data.getvalue())  
Image(graph.create_png())

