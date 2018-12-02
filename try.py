import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd 
from random import randint

class K_Means:
	def __init__(self, data, k =3, tolerance = 0.0001, max_iterations = 500):
		self.k = k
		self.tolerance = tolerance
		self.max_iterations = max_iterations
		self.dataset = data
		self.df = data[['Area','Perimeter','Compactness','Length_of_kernel','Width_of_kernel','Asymmetry_coef','Length_kernel_groove']]
		self.data = self.df.values

	def fit(self):

		self.centroids = {}

		for i in range(self.k):
			idx = randint(0,209)
			self.centroids[i] = self.data[idx]
		con = 1

		for i in range(self.max_iterations):
			self.classes = {}
			for i in range(self.k):
				self.classes[i] = []
			#find the distance between the point and cluster; choose the nearest centroid
			for features in self.data:
				distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
				classification = distances.index(min(distances))
				self.classes[classification].append(features)

			previous = dict(self.centroids)

			#average the cluster datapoints to re-calculate the centroids
			for classification in self.classes:
				self.centroids[classification] = np.average(self.classes[classification], axis = 0)

			isOptimal = True

			for centroid in self.centroids:

				original_centroid = previous[centroid]
				curr = self.centroids[centroid]

				if np.sum((curr - original_centroid)/original_centroid * 100.0) > self.tolerance:
					isOptimal = False

			if isOptimal:
				break

	def accuracy(self, cluster):
		acc = 0
		for a in range(self.k):
			tmp = 0
			for b in range(self.k):
				flag = 0
				for c in cluster[a]:
					for d in self.classes[b]:
						if np.array_equiv(c,d) :
							flag += 1
							break
				tmp = max(flag,tmp)
			acc += tmp
		acc /= 210
		return acc

	def print_cluster(self):
		attr = ['Area','Perimeter','Compactness','Length_of_kernel','Width_of_kernel','Asymmetry_coef','Length_kernel_groove']
		colors = ["r", "g", "b"]
		con = 1

		for a0 in self.df:
			for a1 in self.df:
				if a0 < a1 :
					plt.cla()
					plt.xlabel(a0)
					plt.ylabel(a1)
					# for centroid in self.centroids:
					# 	plt.scatter(self.centroids[centroid][attr.index(a0)], self.centroids[centroid][attr.index(a1)], s = 130, marker = "o")
					for classification in self.classes:
						color = colors[classification]
						for features in self.classes[classification]:
							plt.scatter(features[attr.index(a0)], features[attr.index(a1)], color = color,s = 5)
					
					plt.savefig('./pict'+str(con)+'cl.png')
					con += 1


def get_clus(cluster):
	clustered = {}
	k = np.unique(cluster['Class'])
	
	for i in range((len(k))):
		clustered[i] = []
	
	cluster = cluster.values
	for a in range(len(cluster)):
		index = int(cluster[a,7:])-1
		data = np.delete(cluster[a,:],7)
		clustered[index].append(data)
	
	return clustered


def main():
	df = pd.read_csv("seeds.csv",sep='\t')
	
	clas = get_clus(df)

	km = K_Means(df)
	km.fit()

	km.print_cluster()


if __name__ == "__main__":
	main()