import numpy as np
import pandas as pd
import wandb
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
testing_path = "<path to test>"
labels = np.arange(10)
s = ["T-shirt/Top","Trouser","Pull Over","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle Boot"]
test = pd.read_csv(testing_path)
lab_test = test.label
lab_test = lab_test.to_numpy()

hidden_representation = np.load("dlpa4/HiddenRepresentations.npy")
# print(hidden_representation.shape)

tsne = TSNE(n_components=2)
tsne_hidden = tsne.fit_transform(hidden_representation)
plt.figure(figsize=(6, 5))
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple']
for i, c, label in zip(labels, colors,s):
    plt.scatter(tsne_hidden[lab_test == i, 0], tsne_hidden[lab_test == i, 1], c=c, label=label)
plt.xlabel("T-SNE Component 1")
plt.ylabel("T-SNE Component 2")
plt.legend()
plt.savefig("TSNE.png")
wandb.init(project='dlpa4-manoj-shivangi', entity='shiv')
wandb.log({"T-SNE":plt})
 
