from sknn.mlp import Regressor, Layer
from sklearn import cross_validation
from sklearn import metrics
from colorsys import rgb_to_hls, rgb_to_hsv
import numpy as np

def load_data():
    f = open('../data/rgb.csv')
    x = []
    y = []
    for line in f:
        split_line = line.split(',')
        red = int(split_line[0])
        green = int(split_line[1])
        blue = int(split_line[2])
        hls = rgb_to_hls(red, green, blue)
        hsv = rgb_to_hsv(red, green, blue)
        gray = (red + green + blue) / 3
        x.append([red, green, blue, gray] + list(hls) + [hsv[2]])
        # x.append([red, green, blue, gray] + list(hls) + [hsv[2]])
        lum = split_line[9]
        warmness = split_line[10]
        primary = split_line[11]
        neuance = split_line[12]
        y.append([lum, warmness, primary, neuance])
    return {"x": np.asarray(x), "y": np.asarray(y)}

def closest(x):
    if x < 0.33:
        return '0.0'     
    elif x < 0.66:
        return '0.5'
    else:
        return '1.0'

data = load_data()

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return str(array[idx])

def luminosity():
    value_map = {'Light': 1.0, 'Medium': 0.5, 'Dark': 0.0}

    X = data["x"][:, [0, 1, 2, 3, 5, 7]]
    maxX = np.amax(X, axis=0)
    minX = np.amax(X, axis=0)
    X = (X - minX) / maxX
    Y = data["y"][:, 0]
    Y = np.asarray([value_map[y] for y in Y])

    split_data = cross_validation.train_test_split(X, Y, test_size=0.2)
    X_train = split_data[0]
    X_test = split_data[1]
    Y_train = split_data[2]
    Y_test = split_data[3]

    nn = Regressor(
        layers=[
            Layer("Rectifier", units=3),
            Layer("Linear")],
        learning_rate=1e-3,
        n_iter=100)

    nn.fit(X_train, Y_train)

    print 'inosity accuracy'
    prediction = nn.predict(X_test)
    prediction = [closest(y[0]) for y in prediction]
    Y_test = [closest(y) for y in Y_test]
    print metrics.accuracy_score(prediction, Y_test)

def gamma():
    value_map = {'warm': 1.0, 'neutral': 0.5, 'cold': 0.0}

    X = data["x"][:, [0, 1, 2, 5, 6]]
    X = np.abs(X)
    maxX = np.amax(X, axis=0)
    minX = np.amax(X, axis=0)
    X = (X - minX) / maxX
    Y = data["y"][:, 1]
    Y = np.asarray([value_map[y] for y in Y])

    split_data = cross_validation.train_test_split(X, Y, test_size=0.2)
    X_train = split_data[0]
    X_test = split_data[1]
    Y_train = split_data[2]
    Y_test = split_data[3]

    nn = Regressor(
        layers=[
            Layer("Rectifier", units=3),
            Layer("Linear")],
        learning_rate=1e-3,
        n_iter=100)

    nn.fit(X_train, Y_train)

    print 'inosity accuracy'
    prediction = nn.predict(X_test)
    prediction = [closest(y[0]) for y in prediction]
    Y_test = [closest(y) for y in Y_test]
    print metrics.accuracy_score(prediction, Y_test)

def colour():
    Y = data["y"][:, 2]
    vals = np.unique(Y);
    value_map = {};
    for i in range(0, len(vals)):
        value_map[vals[i]] = (0.0 + i) / (len(vals) - 1)
    value_values = value_map.values()
    keys = value_map.keys()

    Ya = []
    for a in Y:
        k = []
        for i in range(0, len(keys)):
            k.append(0.0)
        k[keys.index(a)] = 1.0
        Ya.append(k)
    Y = np.asarray(Ya)
    
    X = data["x"][:, [0, 1, 2, 3, 4, 5, 6, 7]]
    X = np.abs(X)
    maxX = np.amax(X, axis=0)
    minX = np.amax(X, axis=0)
    X = (X - minX) / maxX

    split_data = cross_validation.train_test_split(X, Y, test_size=0.2)
    X_train = split_data[0]
    X_test = split_data[1]
    Y_train = split_data[2]
    Y_test = split_data[3]

    nn = Regressor(
        layers=[
            Layer("Linear", units=9),
            Layer("Softmax", units=9)],
        learning_rate=5e-2,
        n_iter=100)

    nn.fit(X_train, Y_train)

    print 'colour accuracy'
    prediction = nn.predict(X_test)
    prediction = [np.argmax(y) for y in prediction]
    Y_test = [np.argmax(y) for y in Y_test]
    print metrics.accuracy_score(prediction, Y_test)

luminosity()
gamma()
colour()