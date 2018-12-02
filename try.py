import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd 
from random import randint

class K_Means:
	def __init__(self, k =3, tolerance = 0.0001, max_iterations = 500):
		self.k = k
		self.tolerance = tolerance
		self.max_iterations = max_iterations

	def fit(self, data):

		self.centroids = {}

		#initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
		for i in range(self.k):
			idx = randint(0,209)
			self.centroids[i] = data[idx]
		con = 1

		print(self.centroids)
		#begin iterations
		for i in range(self.max_iterations):
			self.classes = {}
			for i in range(self.k):
				self.classes[i] = []
			#find the distance between the point and cluster; choose the nearest centroid
			for features in data:
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

			#break out of the main loop if the results are optimal, ie. the centroids don't change their positions much(more than our tolerance)
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
	attr = ['Area','Perimeter','Compactness','Length_of_kernel','Width_of_kernel','Asymmetry_coef','Length_kernel_groove']
	df = pd.read_csv("seeds.csv",sep='\t')
	df1 = df[['Area','Perimeter','Compactness','Length_of_kernel','Width_of_kernel','Asymmetry_coef','Length_kernel_groove']]
	X = df1.values #returns a numpy array
	
	clas = get_clus(df)

	km = K_Means(3)
	km.fit(X)

	print(km.accuracy(clas))
	#print(df)
	# colors = ["r", "g", "b"]
	# con = 1

	# for a0 in df1:
	# 	for a1 in df1:
	# 		if a0 < a1 :
	# 			plt.cla()
	# 			plt.xlabel(a0)
	# 			plt.ylabel(a1)
	# 			# for centroid in km.centroids:
	# 			# 	plt.scatter(km.centroids[centroid][attr.index(a0)], km.centroids[centroid][attr.index(a1)], s = 130, marker = "o")
	# 			for classification in km.classes:
	# 				color = colors[classification]
	# 				for features in km.classes[classification]:
	# 					plt.scatter(features[attr.index(a0)], features[attr.index(a1)], color = color,s = 5)
				
	# 			plt.savefig('./pict'+str(con)+'cl.png')
	# 			con += 1


if __name__ == "__main__":
	main()