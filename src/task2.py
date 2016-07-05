from sknn.mlp import Regressor, Layer
from sklearn import cross_validation
from sklearn import metrics
import numpy as np

def load_data():
    f = open('../data/649_2015.txt')
    x = []
    y = []
    for line in f:
        split_line = line.split(',')
        x.append([int(split_line[0]), int(split_line[1])])
        y.append(int(split_line[2]))
    return {"x": np.asarray(x), "y": np.asarray(y)}

nn = Regressor(
    layers=[Layer("Linear")],
    learning_rate=2e-7,
    learning_momentum=0.5,
    regularize="L2",
    weight_decay=1e-2,
    n_iter=100)
data = load_data();

cross_validation_sets = 10

mse = cross_validation.cross_val_score(nn, data["x"], data["y"], cv=cross_validation_sets, scoring='mean_squared_error')
rmse = [(-x) ** 0.5 for x in mse]
avrg_rmse = reduce(lambda x,y: x + y, rmse) / cross_validation_sets;

print avrg_rmse