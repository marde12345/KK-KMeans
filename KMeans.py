import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

attr = ['Area','Perimeter','Compactness','Length_of_kernel','Width_of_kernel','Asymmetry_coef','Length_kernel_groove']
df = pd.read_csv('seeds.csv',sep='\t')
df = df[['Area','Perimeter','Compactness','Length_of_kernel','Width_of_kernel','Asymmetry_coef','Length_kernel_groove']]
dataset = df.astype(float).values.tolist()
X = df.values
print(X)
# f1 = df.Area
# f2 = df.Perimeter

# print(f1)
# print(f2)
# ab=np.array([[0,0],[0,0]])
# bc=np.array([[3,4],[0,1]])
# print(np.linalg.norm(ab-bc))

#print blob
# con = 1
# for a1 in df:
#     for a2 in df:
#         if a1 < a2 :
#             f1 = df[a1].values
#             f2 = df[a2].values
#             plt.cla()
#             plt.scatter(f1,f2,c='black',s=5)
#             plt.xlabel(a1)
#             plt.ylabel(a2)
#             plt.savefig('./pict'+str(con)+'.png')
#             con = con + 1
# plt.show()

# f1 = df['Area'].values
# f2 = df['Peri'].values
# f3 = df['Comp'].values



# plt.scatter(f1, f2, c='black', s=5)
# plt.savefig('abc.png')