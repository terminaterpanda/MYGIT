import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

if __name__ == '__main__':
    # Data Load (MNIST: 64 dim)
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    # t-SNE(64 to 2)
    tsne = TSNE(n_components=2, random_state=1)
    X_tsne = tsne.fit_transform(X)

    # t-SNE Visualization
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
    plt.show()