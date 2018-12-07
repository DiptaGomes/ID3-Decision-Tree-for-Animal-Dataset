import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

dataset= pd.read_csv("dataset.csv") 

print("Dataset Lenght:: ", len(dataset))
print("Dataset Shape:: ", dataset.shape)

X = dataset.values[:, 1:16]
Y = dataset.values[:,17]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)
classifier = clf_entropy.fit(X_train,y_train)
y_pred = clf_entropy.predict(X_test)

print(y_pred)

print("Accuracy is ", accuracy_score(y_test,y_pred)*100)

from sklearn.tree import export_graphviz
export_graphviz(classifier, out_file='tree.dot')

# //Generating a txt file containg the graph
# with open("fruit_classifier.txt", "w") as f:
#     f = tree.export_graphviz(clf_entropy, out_file=f)

# To open the .dot file
# dot -Tpdf tree.dot -o tree.pdf