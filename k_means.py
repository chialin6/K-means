import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

#read files
with open('clus_1.csv', 'rb') as c1:
	rc1 = csv.reader(c1)
	list_c1 = list(rc1)

with open('clus_2.csv', 'rb') as c2:
	rc2 = csv.reader(c2)
	list_c2 = list(rc2)

with open('clus_3.csv', 'rb') as c3:
	rc3 = csv.reader(c3)
	list_c3 = list(rc3)

with open('start.csv', 'rb') as ini:
	rini = csv.reader(ini)
	init_cens = list(rini)
	
c1.close()
c2.close()
c3.close()
ini.close()

points = np.vstack((list_c1, list_c2, list_c3))
init_cens = np.stack(init_cens)
list_c1 = np.stack(list_c1)
list_c2 = np.stack(list_c2)
list_c3 = np.stack(list_c3)

fig = plt.figure()
plt.scatter(points[:,0], points[:,1], color='k')
plt.scatter(init_cens[:,0], init_cens[:,1], color='y')
fig.savefig('ini.jpg')
plt.clf()
##plt.show()

ini_img = np.zeros((512,512,3), np.uint8)
points = points.astype(np.int)
init_cens = init_cens.astype(np.int)
for i in range(points.shape[0]):
	cv2.circle(ini_img, (points[i][0], points[i][1]), 2, (179,179,179), -1)
for i in range(init_cens.shape[0]):
	cv2.circle(ini_img, (init_cens[i][0], init_cens[i][1]), 4, (255,255,255), -1)
##cv2.imwrite('ini_img.jpg', ini_img)
##img = cv2.imread('ini_img.jpg')
cv2.imshow('i', ini_img)
cv2.waitKey(1000)

iter = 0
err = 5.0

cens = init_cens
clus1 = []
clus2 = []
clus3 = []

def dist_square(a0, a1, b0, b1):
	a0 = np.float32(a0)
	a1 = np.float32(a1)
	b0 = np.float32(b0)
	b1 = np.float32(b1)
	return (b0-a0)**2+(b1-a1)**2

while iter<=10 or err>1.0:
	if (iter!=0):
		clus1 = clus1.tolist()
		clus2 = clus2.tolist()
		clus3 = clus3.tolist()
		del clus1[:]
		del clus2[:]
		del clus3[:]
		
	for i in range(points.shape[0]):
		d1 = dist_square(points[i][0], points[i][1], cens[0][0], cens[0][1])
		d2 = dist_square(points[i][0], points[i][1], cens[1][0], cens[1][1])
		d3 = dist_square(points[i][0], points[i][1], cens[2][0], cens[2][1])
		
		if d1<=d2 and d1<=d3:
			clus1.append(points[i])
		elif d2<=d3 and d2<=d1:
			clus2.append(points[i])
		else:
			clus3.append(points[i])
	
	old_cens = cens
	
	clus1 = np.asarray(clus1)
	clus2 = np.asarray(clus2)
	clus3 = np.asarray(clus3)
	
	cens[0] = np.mean(clus1, axis=0, dtype=np.float32)
	cens[1] = np.mean(clus2, axis=0, dtype=np.float32)
	cens[2] = np.mean(clus3, axis=0, dtype=np.float32)
	
	plt.scatter(clus1[:,0], clus1[:,1], color='b')
	plt.scatter(clus2[:,0], clus2[:,1], color='r')
	plt.scatter(clus3[:,0], clus3[:,1], color='g')
	plt.scatter(cens[:,0], cens[:,1], color='y')
	##plt.show()
	fig.savefig('iter_%d.jpg' % iter)
	plt.clf()
	
	img = np.zeros((512,512,3), np.uint8)
	for i in range(clus1.shape[0]):
		cv2.circle(img, (clus1[i][0], clus1[i][1]), 2, (255,0,0), -1)
	for i in range(clus2.shape[0]):
		cv2.circle(img, (clus2[i][0], clus2[i][1]), 2, (0,255,0), -1)
	for i in range(clus3.shape[0]):
		cv2.circle(img, (clus3[i][0], clus3[i][1]), 2, (0,0,255), -1)
	for i in range(cens.shape[0]):
		cv2.circle(img, (cens[i][0], cens[i][1]), 4, (255,255,255), -1)
	
	##img_ = cv2.imread('iter_img_%d.jpg' % iter)
	cv2.imshow('i', img)
	cv2.waitKey(1000)
	##cv2.imwrite('iter_img_%d.jpg' % iter, img)

	err = max(dist_square(old_cens[0][0],old_cens[0][1],cens[0][0],cens[0][1]),
			  dist_square(old_cens[1][0],old_cens[1][1],cens[1][0],cens[1][1]), 
			  dist_square(old_cens[2][0],old_cens[2][1],cens[2][0],cens[2][1]))
	
	iter = iter+1