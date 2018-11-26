import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv('seeds.csv',sep='\t')
f1 = df['Area'].values
f2 = df['Comp'].values
X = np.array(list(zip(f1, f2)))

plt.scatter(f1, f2, c='black', s=7)
plt.show()