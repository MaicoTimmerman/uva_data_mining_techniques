import os
import pickle

import graphviz
from sklearn import tree

with open("./models/model_tiny_train.pkl", "rb") as f:
    model = pickle.load(f)

path = "./trees"
if not os.path.exists(path):
    os.mkdir(path)

for i, estimator in enumerate(model.estimators_):
    graph = graphviz.Source(tree.export_graphviz(estimator[0]))
    graph.render(os.path.join(path, f"tree{i}"))
