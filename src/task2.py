from sknn.mlp import Regressor, Layer
from sklearn import cross_validation
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_data():
    f = open('../data/649_2015.txt')
    prev = None
    x = []
    y = []
    for line in f:
        split_line = line.split(',')
        data = []
        for i in range(0, 7):
            num = split_line[i].strip()
            data.append(int(num))

        if prev != None:
            x.append([data[0]] + prev[1:])
            result = [];
            a = data[1:]

            for i in range(0, 49):
                result.append(0)
            for num in a:
                result[num - 1] = 1
            y.append(result)
        prev = data

    return {"x": np.asarray(x), "y": np.asarray(y)}

nn = Regressor(
    layers=[
        Layer("Linear", units=49),
        Layer("Linear", units=120),
        Layer("Linear", units=49)],
    learning_rate=2e-6,
    n_iter=50)

# pipe = Pipeline(
#     [MinMaxScaler(feature_range=(0, 1)),
#     nn])

data = load_data();

split_data = cross_validation.train_test_split(data["x"], data["y"], test_size=0.2)
x_train = split_data[0]
x_test = split_data[1]
y_train = split_data[2]
y_test = split_data[3]

nn.fit(x_train, y_train)

predictions = nn.predict(x_test)

avrg = 0.0

for i in range(0, len(x_test)):
    predicted = predictions[i]
    real = y_test[i]
    partition = np.argpartition(predicted, -6)[-6:]
    hitcount = 0
    for num in partition:
        if real[num] == 1:
            hitcount = hitcount + 1
    avrg = avrg + hitcount

avrg = avrg / len(x_test)
print avrg