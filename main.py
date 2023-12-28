"""Basic adaline usage"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sgd_adaline import SGDAdaline

DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

if __name__ == '__main__':
    df = pd.read_csv(DATASET_URL, header=None, encoding='utf-8')
    inputs = df.iloc[:100, [0, 2]].values
    labels = np.where(df.iloc[:100, 4].values == 'Iris-setosa', 1, 0)

    model = SGDAdaline(learning_rate=0.001, epochs=20)
    model.fit(inputs, labels)

    test_data = np.array([[7, 4.2], [5, 1]])
    test_results = model.predict(test_data)

    x1_min, x1_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1
    x2_min, x2_max = inputs[:, 1].min() - 1, inputs[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02))
    res = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    res = res.reshape(xx1.shape)

    fig, (ada_plt, loss_plt) = plt.subplots(2)

    ada_plt.contourf(xx1, xx2, res, alpha=0.2, cmap=ListedColormap(('green', 'red')))
    ada_plt.set_xlim(xx1.min(), xx1.max())
    ada_plt.set_ylim(xx2.min(), xx2.max())

    ada_plt.scatter(inputs[:50, 0], inputs[:50, 1], color='red', label='Setosa')
    ada_plt.scatter(inputs[50:, 0], inputs[50:, 1], color='green', label='Versicolor')

    ada_plt.scatter(
        test_data[test_results == 1, 0],
        test_data[test_results == 1, 1],
        marker='s', s=100, edgecolors='black', color='red', label='Setosa (predicted)'
    )
    ada_plt.scatter(
        test_data[test_results == 0, 0],
        test_data[test_results == 0, 1],
        marker='s', s=100, edgecolors='black', color='green', label='Versicolor (predicted)'
    )
    ada_plt.set_xlabel('Sepal length')
    ada_plt.set_ylabel('Petal length')
    ada_plt.legend(loc='lower right')

    loss_plt.plot(np.arange(1, len(model.losses) + 1), model.losses, marker='o')
    loss_plt.set_xlabel('Epochs')
    loss_plt.set_ylabel('Average loss')

    plt.show()
