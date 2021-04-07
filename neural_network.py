from datetime import datetime

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def nn_experiment(data):
    start = datetime.now()
    _train_x, _test_x, _train_y, _test_y = train_test_split(
        data.x, data.y, test_size=0.33, random_state=42)
    neural_net_grid = MLPClassifier(activation='tanh', random_state=10, solver='adam')
    best_nn = neural_net_grid.fit(_train_x, _train_y)
    predicted_y_values = best_nn.predict(_test_x)
    result_score = accuracy_score(_test_y, predicted_y_values)
    end = datetime.now()
    time_taken = (end - start).total_seconds()

    return [result_score, time_taken]