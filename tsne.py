# implementation of TSNE visualization
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from numpy import reshape
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

class TSNEGen():
    def __init__(self, x, y = None):
        if torch.is_tensor(x):
            # print(x.shape)
            # if y == None:
            #     self.y = []
            #     for i in range(0, x.shape[0]):
            #         for _ in range(0, x.shape[1]):
            #             self.y.append(i)
            #     self.y = np.array(self.y)
            # else:
            #     self.y = np.array(y).squeeze()

            reshaped_tensor = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

            # Convert the reshaped tensor to a NumPy array
            self.x = reshaped_tensor.numpy()
        else:
            self.x = np.array(x)
        
        if y == None:
            self.y = []
            for i in range(0, x.shape[0]):
                for _ in range(0, x.shape[1]):
                    self.y.append(i)
            self.y = np.array(self.y)
        else:
            self.y = np.array(y[0]).squeeze()
            

        print("x:", self.x.shape)
        print("y:", self.y.shape)

    def generateTSNE(self):
        tsne = TSNE(n_components=2, random_state=42)
        z = tsne.fit_transform(self.x) 
        df = pd.DataFrame()
        df["y"] = self.y
        df["1"] = z[:,0]
        df["2"] = z[:,1]

        sns.scatterplot(x="1", y="2", hue=df.y.tolist(),
                        palette=["red", "green"],
                        data=df).set(title="Rotated synthetic data T-SNE projection") 
        #plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
