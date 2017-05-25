





   

Homework1
一、目的和要求
1、sample the function curve of y=sin(x) with Gaussian noise
2、fit degree 3 and 9 curves in 10 samples
3、fit degree 9 curves in 15 and 100 samples(two pic)
4、fit degree 9 curve in 10 samples but with regularization term

二、python代码
import pylab
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.pipeline import Pipeline  
from sklearn.preprocessing import PolynomialFeatures  

x=np.linspace(0,6,100)
y=np.sin(x)
a1=np.random.rand(10)*7
b1=np.sin(a1)+(np.random.rand(10)*2-1)*0.5
plt.figure(1)
plt.plot(x,y,color="g",linewidth=2)
plt.title('M=3,N=10')
plt.scatter(a1,b1,color="b")
c1=np.polyfit(a1,b1,3)
p1=np.poly1d(c1)
plt.plot(x,p1(x),color="r")

plt.figure(2)
plt.scatter(a1,b1,color="b")
plt.plot(x,p1(x),color="g")
c2=np.polyfit(a1,b1,9)
p2=np.poly1d(c2)
plt.plot(x,p2(x),color="r")
plt.title('M=9,N=10')

plt.figure(3)
a2=np.random.rand(15)*7
b2=np.sin(a2)+(np.random.rand(15)*2-1)*0.5
plt.scatter(a2,b2,color="b")
c3=np.polyfit(a2,b2,9)
p3=np.poly1d(c3)
plt.plot(x,p3(x),color="r")
plt.plot(x,y,label="sin(x)",color="g",linewidth=2)
plt.title('M=9,N=15')

plt.figure(4)
a3=np.random.rand(100)*7
b3=np.sin(a3)+(np.random.rand(100)*2-1)*0.5
plt.scatter(a3,b3,color="b")
c4=np.polyfit(a3,b3,9)
p4=np.poly1d(c4)
plt.plot(x,p4(x),color="r")
plt.plot(x,y,label="sin(x)",color="g",linewidth=2)
plt.title('M=9,N=100')

plt.figure(5)
clf = Pipeline([('poly', PolynomialFeatures(degree=9)),
                    ('linear', linear_model.Ridge (alpha=np.e**(-18)))])
clf.fit(a1[:, np.newaxis], b1[:])
plt.plot(x,clf.predict(x[:,np.newaxis]))
plt.title('ln lambda=-18')
plt.plot(x,y,label="sin(x)",color="g",linewidth=2)
plt.scatter(a1,b1,color="b")

三、结果















Homework2

一、目的和要求

1、convert data from the UCI Optical Recognition of Handwritten Digits Data Set 
2、perform PCA over all digit '3' with 2 components
3、plot the PCA results as below (also in page #12 of PCA)

二、python代码
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cPickle

def extract_data(filename,num):
    file = open(filename)
    Y = []
    for line in file:
        listline = map(int,line.split(','))
        if listline[-1] == num:
        	Y.append(listline[0:])
    file.close()
    cPickle.dump(Y,open('save_'+filename,"wb"))

def data_process():

    extract_data('optdigits.tra',3)
    extract_data('optdigits.tes',3)
    data_tra = cPickle.load(open('save_optdigits.tra',"rb"))
    data_tes = cPickle.load(open('save_optdigits.tes',"rb"))
    data_set = np.vstack((data_tra,data_tes))
   
    data_without_code = np.zeros((len(data_set),len(data_set[0])-1))
    data_without_code = data_set[:,:-1]
   
    binary_image = np.zeros((len(data_without_code),8,8))
   
    for i in range(len(data_without_code)):
    	binary_image[i, :, :] = data_without_code[i, :].reshape(8,8)
    for i in range(len(data_without_code)):
    	for j in range(0,8):
    		for k in range(0,8):
    			if binary_image[i,j,k] != 0:
    				binary_image[i,j,k] = 1

  
    return data_without_code, binary_image

def PCA(data_set,feature):
 	mean_data = np.mean(data_set,axis = 0)
 	center = data_set - mean_data
	covariance = np.cov(center,rowvar = 0) 
	eigenvalue,eigenvector = np.linalg.eig(np.mat(covariance))
	eigenvalue_sort = np.argsort(-eigenvalue)
	eigenvalue_feature_index = eigenvalue_sort[0 : feature]
	eigenvector_feature = eigenvector[:, eigenvalue_feature_index]
	reduced_data_set = center * eigenvector_feature
	recovery_data_set = (reduced_data_set * eigenvector_feature.transpose()) + mean_data
	return reduced_data_set, recovery_data_set

def find_nearest_point(point,reduced_data_set):
	min_dist = 999999
	for i in range(len(reduced_data_set)):
		dist = ((point[0]-reduced_data_set[i,0])**2+(point[1]-reduced_data_set[i,1])**2)**0.5
		if dist < min_dist:
			min_dist = dist
			arg = i
	return arg

def find_key_points(reduced_data_set):
	#generate 25 init points
	x = [-20, -10, 0, 10, 20]
	points=[]
	for i in range(5):
		for j in range(5):
			points.append((x[i],x[j]))

	index = []
	for i in range(len(points)):
		arg = find_nearest_point(points[i],reduced_data_set)
		index.append(arg)
	return index

def main():
	data_without_code, binary_image = data_process()
	reduced_data_set, recovery_data_set = PCA(data_without_code,2)
	index = find_key_points(reduced_data_set)

	# show the PCA 
	fig1 = plt.figure(1)
	plt.plot(reduced_data_set[:, 0], reduced_data_set[:, 1], 'bo')
	for i in range(len(index)):
		plt.plot(reduced_data_set[index[i],0],reduced_data_set[index[i],1],'ro')
	plt.xlabel("First Principle Component")
	plt.ylabel("Second Principle Component")
	plt.title("PCA Demo")
	plt.grid(True)   # add grid to the pictur
	plt.show()
	fig1.savefig('figure_PCA_matrix.jpg')

	# show the image 
	fig2 = plt.figure(2)
	for i in range(25):
		subplot = plt.subplot(5,5,i)
		sub_image = recovery_data_set[index[i], :].reshape((8,8))
		plt.imshow(sub_image,cmap=cm.gray_r)
	plt.show()
	fig2.savefig('image_PCA.jpg')
if __name__ == "__main__":
main()

三、结果

+

Homework03
一、目的和要求
implement MOG in 2D case 
Generate 2D Gaussian distribution 
E-M method

二、python代码
import numpy as np
import matplotlib.pyplot as plot
 
N = 50000
T1 = np.random.rand(N)
T2 = np.random.rand(N)
 
r = np.sqrt(-2*np.log(T2))
theta = 2*np.pi*T1
X = r*np.cos(theta)
Y = r*np.sin(theta)
 
heatmap, xedges, yedges = np.histogram2d(X, Y, bins=80)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
 
plot.imshow(heatmap, extent=extent)
plot.show()

from numpy import *  
import time  
import matplotlib.pyplot as plt  
  
  
# calculate Euclidean distance  
def euclDistance(vector1, vector2):  
    return sqrt(sum(power(vector2 - vector1, 2)))  
  
# init centroids with random samples  
def initCentroids(dataSet, k):  
    numSamples, dim = dataSet.shape  
    centroids = zeros((k, dim))  
    for i in range(k):  
        index = int(random.uniform(0, numSamples))  
        centroids[i, :] = dataSet[index, :]  
    return centroids  
  
# k-means cluster  
def kmeans(dataSet, k):  
    numSamples = dataSet.shape[0]   
    clusterAssment = mat(zeros((numSamples, 2)))  
    clusterChanged = True  
  
    ## step 1: init centroids  
    centroids = initCentroids(dataSet, k)  
  
    while clusterChanged:  
        clusterChanged = False  
        ## for each sample  
        for i in xrange(numSamples):  
            minDist  = 100000.0  
            minIndex = 0  
            ## for each centroid  
            ## step 2: find the centroid who is closest  
            for j in range(k):  
                distance = euclDistance(centroids[j, :], dataSet[i, :])  
                if distance < minDist:  
                    minDist  = distance  
                    minIndex = j  
              
            ## step 3: update its cluster  
            if clusterAssment[i, 0] != minIndex:  
                clusterChanged = True  
                clusterAssment[i, :] = minIndex, minDist**2  
  
        ## step 4: update centroids  
        for j in range(k):  
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]  
            centroids[j, :] = mean(pointsInCluster, axis = 0)  
  
    'Congratulations, cluster complete!'  
    return centroids, clusterAssment  
  
# show your cluster only available with 2-D data  
def showCluster(dataSet, k, centroids, clusterAssment):  
    numSamples, dim = dataSet.shape  
    if dim != 2:  
        "Sorry! I can not draw because the dimension of your data is not 2!"  
        return 1  
  
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  
    if k > len(mark):  
        "Sorry! Your k is too large! please contact Zouxy"  
        return 1  
  
    # draw all samples  
    for i in xrange(numSamples):  
        markIndex = int(clusterAssment[i, 0])  
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])  
  
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']  
    # draw the centroids  
    for i in range(k):  
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)  
  
plt.show()  

三、结果





Homework 04
一、目的和要求
Implement the Levenberg-Marquardt method

二、python代码

import numpy as np
from scipy.optimize import leastsq
import pylab as pl

# A*sin(2*pi*k*x + theta)
def function(x, p):
    A, k, theta = p
    return A*np.sin(2*np.pi*k*x+theta)   

def residuals(p, y, x):
    return y - function(x, p)

x = np.linspace(0, -2*np.pi, 100)
A, k, theta = 10, 0.34, np.pi/6 # method param of real data
y0 = function(x, [A, k, theta]) # real data
y1 = y0 + 2 * np.random.randn(len(x)) # experiment data after fussion  

p0 = [7, 0.2, 0] # first gusee of method matching

plsq = leastsq(residuals, p0, args=(y1, x))

print (u"realParams:", [A, k, theta])
print (u"matchingParams", plsq[0]) # params after matching real data
fig1 = pl.figure(1)
pl.plot(x, y0, label=u"real data")
pl.plot(x, y1, label=u"test data")
pl.plot(x, function(x, plsq[0]), label=u"match data")
pl.legend()
pl.show()
fig1.savefig('LM.jpg')

三、结果
realParams: [10, 0.34, 0.5235987755982988]
matchingParams [-9.89552948  0.3379844   3.68369217]

Homework05
一、目的和要求
Implement (simplified) SVM method 
1、input 2D data and their label (in two classes)
2、implement quadratic programming 
3、output (and plot) classification results


二、python代码
from numpy import *  
import time  
import matplotlib.pyplot as plt   
 
 
def calcKernelValue(matrix_x, sample_x, kernelOption):  
    kernelType = kernelOption[0]  
    numSamples = matrix_x.shape[0]  
    kernelValue = mat(zeros((numSamples, 1)))  
      
    if kernelType == 'linear':  
        kernelValue = matrix_x * sample_x.T  
    elif kernelType == 'rbf':  
        sigma = kernelOption[1]  
        if sigma == 0:  
            sigma = 1.0  
        for i in xrange(numSamples):  
            diff = matrix_x[i, :] - sample_x  
            kernelValue[i] = exp(diff * diff.T / (-2.0 * sigma**2))  
    else:  
        raise NameError('Not support kernel type! You can use linear or rbf!')  
    return kernelValue  
  
  
# calculate kernel matrix given train set and kernel type   
def calcKernelMatrix(train_x, kernelOption):  
    numSamples = train_x.shape[0]  
    kernelMatrix = mat(zeros((numSamples, numSamples)))  
    for i in xrange(numSamples):  
        kernelMatrix[:, i] = calcKernelValue(train_x, train_x[i, :], kernelOption)  
    return kernelMatrix  
  
  
# define a struct just for storing variables and data   
class SVMStruct:  
    def __init__(self, dataSet, labels, C, toler, kernelOption):  
        self.train_x = dataSet # each row stands for a sample   
        self.train_y = labels  # corresponding label   
        self.C = C             # slack variable   
        self.toler = toler     # termination condition for iteration   
        self.numSamples = dataSet.shape[0] # number of samples   
        self.alphas = mat(zeros((self.numSamples, 1))) # Lagrange factors for all samples   
        self.b = 0  
        self.errorCache = mat(zeros((self.numSamples, 2)))  
        self.kernelOpt = kernelOption  
        self.kernelMat = calcKernelMatrix(self.train_x, self.kernelOpt)  
  
          
# calculate the error for alpha k   
def calcError(svm, alpha_k):  
    output_k = float(multiply(svm.alphas, svm.train_y).T * svm.kernelMat[:, alpha_k] + svm.b)  
    error_k = output_k - float(svm.train_y[alpha_k])  
    return error_k  
  
  
# update the error cache for alpha k after optimize alpha k   
def updateError(svm, alpha_k):  
    error = calcError(svm, alpha_k)  
    svm.errorCache[alpha_k] = [1, error]  
  
  
# select alpha j which has the biggest step   
def selectAlpha_j(svm, alpha_i, error_i):  
    svm.errorCache[alpha_i] = [1, error_i] # mark as valid(has been optimized)   
    candidateAlphaList = nonzero(svm.errorCache[:, 0].A)[0] # mat.A return array   
    maxStep = 0; alpha_j = 0; error_j = 0  
  
    # find the alpha with max iterative step   
    if len(candidateAlphaList) > 1:  
        for alpha_k in candidateAlphaList:  
            if alpha_k == alpha_i:   
                continue  
            error_k = calcError(svm, alpha_k)  
            if abs(error_k - error_i) > maxStep:  
                maxStep = abs(error_k - error_i)  
                alpha_j = alpha_k  
                error_j = error_k  
    # if came in this loop first time, we select alpha j randomly   
    else:             
        alpha_j = alpha_i  
        while alpha_j == alpha_i:  
            alpha_j = int(random.uniform(0, svm.numSamples))  
        error_j = calcError(svm, alpha_j)  
      
    return alpha_j, error_j  
  
  
# the inner loop for optimizing alpha i and alpha j   
def innerLoop(svm, alpha_i):  
    error_i = calcError(svm, alpha_i)  
 
    if ((svm.train_y[alpha_i] * error_i < -svm.toler) and (svm.alphas[alpha_i] < svm.C) or (svm.train_y[alpha_i] * error_i > svm.toler) and (svm.alphas[alpha_i] > 0)):  
   
        alpha_j, error_j = selectAlpha_j(svm, alpha_i, error_i)  
        alpha_i_old = svm.alphas[alpha_i].copy()  
        alpha_j_old = svm.alphas[alpha_j].copy()  
  
        if svm.train_y[alpha_i] != svm.train_y[alpha_j]:  
            L = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])  
            H = min(svm.C, svm.C + svm.alphas[alpha_j] - svm.alphas[alpha_i])  
        else:  
            L = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.C)  
            H = min(svm.C, svm.alphas[alpha_j] + svm.alphas[alpha_i])  
        if L == H:  
            return 0  
 
        eta = 2.0 * svm.kernelMat[alpha_i, alpha_j] - svm.kernelMat[alpha_i, alpha_i] - svm.kernelMat[alpha_j, alpha_j]  
        if eta >= 0:  
            return 0  
  
        svm.alphas[alpha_j] -= svm.train_y[alpha_j] * (error_i - error_j) / eta  
     
        if svm.alphas[alpha_j] > H:  
            svm.alphas[alpha_j] = H  
        if svm.alphas[alpha_j] < L:  
            svm.alphas[alpha_j] = L  
       
        if abs(alpha_j_old - svm.alphas[alpha_j]) < 0.00001:  
            updateError(svm, alpha_j)  
            return 0  
    
        svm.alphas[alpha_i] += svm.train_y[alpha_i] * svm.train_y[alpha_j] * (alpha_j_old - svm.alphas[alpha_j])  
  
        b1 = svm.b - error_i - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
                                                    * svm.kernelMat[alpha_i, alpha_i] \
                             - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
                                                    * svm.kernelMat[alpha_i, alpha_j]
        b2 = svm.b - error_j - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
                                                    * svm.kernelMat[alpha_i, alpha_j] \
                             - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
                                                    * svm.kernelMat[alpha_j, alpha_j]  
        if (0 < svm.alphas[alpha_i]) and (svm.alphas[alpha_i] < svm.C):  
            svm.b = b1  
        elif (0 < svm.alphas[alpha_j]) and (svm.alphas[alpha_j] < svm.C):  
            svm.b = b2  
        else:  
            svm.b = (b1 + b2) / 2.0  
     
        updateError(svm, alpha_j)  
        updateError(svm, alpha_i)  
  
        return 1  
    else:  
        return 0  
  
  
# the main training procedure   
def trainSVM(train_x, train_y, C, toler, maxIter, kernelOption = ('rbf', 1.0)):  
    # calculate training time   
    startTime = time.time()  
  
    # init data struct for svm   
    svm = SVMStruct(mat(train_x), mat(train_y), C, toler, kernelOption)  
      
    # start training   
    entireSet = True  
    alphaPairsChanged = 0  
    iterCount = 0  
    # Iteration termination condition:   
    #   Condition 1: reach max iteration   
    #   Condition 2: no alpha changed after going through all samples,   
    #                in other words, all alpha (samples) fit KKT condition   
    while (iterCount < maxIter) and ((alphaPairsChanged > 0) or entireSet):  
        alphaPairsChanged = 0  
  
        # update alphas over all training examples   
        if entireSet:  
            for i in xrange(svm.numSamples):  
                alphaPairsChanged += innerLoop(svm, i)  
            print '---iter:%d entire set, alpha pairs changed:%d' % (iterCount, alphaPairsChanged)  
            iterCount += 1  
        # update alphas over examples where alpha is not 0 & not C (not on boundary)   
        else:  
            nonBoundAlphasList = nonzero((svm.alphas.A > 0) * (svm.alphas.A < svm.C))[0]  
            for i in nonBoundAlphasList:  
                alphaPairsChanged += innerLoop(svm, i)  
            print '---iter:%d non boundary, alpha pairs changed:%d' % (iterCount, alphaPairsChanged)  
            iterCount += 1  
  
        # alternate loop over all examples and non-boundary examples   
        if entireSet:  
            entireSet = False  
        elif alphaPairsChanged == 0:  
            entireSet = True  
  
    print 'Congratulations, training complete! Took %fs!' % (time.time() - startTime)  
    return svm  
  
  
  
  
# show your trained svm model only available with 2-D data   
def showSVM(svm):  
    if svm.train_x.shape[1] != 2:  
        print "Sorry! I can not draw because the dimension of your data is not 2!"  
        return 1  
	
    # draw all samples   
    for i in xrange(svm.numSamples):  
        if svm.train_y[i] == -1:  
            plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'xr',35.0)  
        elif svm.train_y[i] == 1:  
            plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'ob',35.0)  
  
    # mark support vectors   
    supportVectorsIndex = nonzero(svm.alphas.A > 0)[0]  
    for i in supportVectorsIndex:  
        plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'oy')  
      
    # draw the classify line   
    w = zeros((2, 1))  
    for i in supportVectorsIndex:  
        w += multiply(svm.alphas[i] * svm.train_y[i], svm.train_x[i, :].T)   
    min_x = min(svm.train_x[:, 0])[0, 0]  
    max_x = max(svm.train_x[:, 0])[0, 0]  
    y_min_x = float(-svm.b - w[0] * min_x) / w[1]  
    y_max_x = float(-svm.b - w[0] * max_x) / w[1]  
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')  
    plt.axis([-15, 15, -15, 15])
    plt.show()
    

def genData(N, dim = 2):
	rng = [-10,10]
	X = random.random((N,dim))*(rng[1] - rng[0]) + rng[0]
	X = hstack((ones((N,1)),X))
	while  True:
		sample = random.random((dim,dim))*(rng[1] - rng[0]) + rng[0]
		sample = hstack((ones((dim,1)),sample))
		W = cross(sample[0],sample[1])
		y = sign(dot(X,W.T))
		if all(y):
			break
	X = X[:,1:]
	return X,y

if __name__ =='__main__':
	# generate and show data
	fig1 = plt.figure(1);
	X,y = genData(100)
	
	idx_1 = (y == 1)
	idx_2 = (y == -1)
	plt.scatter(X[idx_1,0],X[idx_1,1],marker = 'o', label = '1', color = 'b' ,s = 35)
	plt.scatter(X[idx_2,0],X[idx_2,1],marker = 'x', label = '-1', color = 'r', s = 35)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend()
	plt.show()
	fig1.savefig("original_data.jpg");

	# train SVM
	X = mat(X)
	y = mat(y).T
	X_train = X[0:80,:]
	y_train = y[0:80]

	X_test = X[81:,:]
	y_test = y[81:]

	C = 1
	toler = 0.001
	maxIter = 50
	svmClassifier = trainSVM(X_train, y_train, C, toler, maxIter, kernelOption = ('linear', 0)) 

	# show result
	print "show the result..."
	fig2 = plt.figure(2)     
	showSVM(svmClassifier)
fig2.savefig("After_SVM.jpg");
三、结果

Original data:

After SVM:
