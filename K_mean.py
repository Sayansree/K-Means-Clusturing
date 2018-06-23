import numpy as np 
from base64 import b64decode
from json import loads
import matplotlib.pyplot as plt  

""" algorithm to classify unlabeled data into K classes using K_means_clustering algorithm
    devloper -> sayansree paria 
    bilding the K means clusturing machine learning model from scrach
    """
class k_mean_cluster:

    def __init__(self):
        """default constructor"""
        pass

    def fit(self,data):
        """input your training dataset to model"""
        self.clusters=data                                              #store our training data set as a 2D Tensor(matrix) features in rows and samples in columns

    
    def eulidian_norm(self,x,y):
        """calculate distance between two n dimention points in eulidian space"""
        return np.sqrt(np.dot((x-y)**2,np.ones(x.shape[0])))               #returns the scalar magnitude of distance 

    
    def asign_centeroids(self):
        """assign nearest centeroid to each of the datapoints as we assign it to centeroids cluster""" 
        for i,cluster in enumerate(self.clusters):                      #iterate through each data points 
            distance=np.zeros((self.k,1))                               #vector to represent distance of each point from every centeroid
            for j,centroid in enumerate(self.centroids):                #itereate through every centeroid points
                distance[j]=self.eulidian_norm(cluster,centroid)        #calculate their distance wrt the the current data point
            minm=np.argmin(distance)                                    #select the closest of the all centeroid
            self.cluster_index_list[minm]=self.cluster_index_list[minm]+[i] #assign the datapoint to the closet centeroid's cluster


    def center_of_mass(self,cluster_list):
        """to calculate mean vector for position of centeroid among its cluster or at its center of mass"""
        ndpoint=np.zeros(self.centroids.shape[1])                       #to store the mean position vector of a group of cluster 
        for i in cluster_list:                                          #iterate through all the points in the the cluster of the centeroid 
            ndpoint=ndpoint+self.clusters[i]                            #vector sumation of all their positions 
        return ndpoint/len(cluster_list)                                #return their arithmetic mean

    
    def move_centroids(self):
        """to calculate mean vector for position of centeroid among its cluster or at its center of mass"""
        for i,cluster_list in enumerate(self.cluster_index_list):       #iterate through all centeroid points
            if not len(cluster_list)==0:                                #if the centeroid has a cluster of datapoints
                self.centroids[i] = self.center_of_mass(cluster_list)   #update the centeroid location to the center of mass

    
    def initial_cent(self):
        """to assign initial guessed (not random) position to centeroids and expected cluster to datapoints
        this boosts the performance and reduces overall iteration in most cases"""
        dist={}                                                         #dictionary to store datapoint index(key) vs distance from origin(value)
        origin=np.zeros(self.centroids.shape[1])                        #cordinate of origin (zero)
        for i,cluster in enumerate(self.clusters):                      #iterate through each data points
            dist[i]=self.eulidian_norm(origin,cluster)                  #compute and store their distance from origin
        itr=0                                                           #monitor iteration 
        for key, value in sorted(dist.items(), key=lambda x: x[1]):     #iterate through sorted dictionary of distances(values)
            j=itr*self.k//self.clusters.shape[0]                        #distribute ito groups and assign to each cluster centeroid
            self.cluster_index_list[j]=self.cluster_index_list[j]+[key] #assign datapoints to rspective clustering centeroid
            itr=itr+1                                                   #update counter
        self.move_centroids()                                           #wemove the centeroids to the mean position of the assigned cluste

    
    def train(self,k,maxitr=30):
        """to train our model to gnerate 'k' cluster of data points"""
        self.k=k                                                        #store the required no of cluster in the process
        self.centroids=np.zeros((k,self.clusters.shape[1]),dtype=float) #stores centeroid location
        self.cluster_index_list=[[]]*k                                  # stores index of assignedclusters for a particuar centroid
        self.initial_cent()                                             #assigns initial centeroid positions
        prev=None                                                       #compare changes to verify convergence
        itr=0                                                           #count iteration
        while itr<maxitr and prev!=self.cluster_index_list:             #iterate while we are in max itration budget until convergence hen model is trained
            prev=list(self.cluster_index_list)                          #store current values for comparison            
            self.cluster_index_list=[[]]*k                              #empty list for storing new centeroid positions
            self.asign_centeroids()                                     #assign cluster of data to each centeroid 
            self.move_centroids()                                       #move the centeroids to their center of cluster
            itr=itr+1                                                   #update iteration
        return   self.centroids,self.cluster_index_list,itr             #return our calculated results
    
    def predict(self,point):
        """to predict some objects belonging to any clusters"""
        distance=np.zeros((self.k,1))
        for j,centroid in enumerate(self.centroids):                #itereate through every centeroid points
            distance[j]=self.eulidian_norm(cluster,centroid)        #calculate their distance wrt the the current data point
        class_index=np.argmin(distance)                             #compute nearest cluster centeroid
        return class_index                                          #return the cluster index



def parse(x):
    """"decode our file data """
    digit = loads(x)
    array = np.fromstring(b64decode(digit["data"]),dtype=np.ubyte)
    array = array.astype(np.float64)
    return array

digits=[]                                                               #we will store all image data as array (1D)
with open("C:\\Users\\S.PARIA\\Documents\\python\AI\\basic_math\\k mean clustering\\digits.base64.json","r") as f: #open our MNIST handwritten dataset of images
    for s in f.readlines():                                             #for each image in the file
        digits.append(parse(s))                                         #add the image to list as a list of 28*28(784) size

digits=np.array(digits)                                                 # convert the  image datas into a 2D array 

def display_digit(digit, k):
    """this function displays few images from 'digit' 
    and 'k' contains index of the images to be displayed""" 
    m=int(np.sqrt(len(k)))                                              #fix our rows
    n=len(k)//m   +1                                                    #fix our columns
    plt.figure()
    for i,loc in enumerate(k):                                          #for all images
        plt.subplot(m,n,i+1)                                            #in the next subplot locatiom
        image = digit[loc]                                              
        fig = plt.imshow(image.reshape(28,28))                          #reshape image into its 2D format 28x28
        fig.set_cmap('gray_r')
    plt.show()                                                          #show our images


kmc=k_mean_cluster()                                                    #create object for storing our clusturing model
kmc.fit(digits[:10000])                                                 #choose first 10,000 images of total 60,000 for training to make training fast
c,cil,itr=kmc.train(10)                                                 #train on the dataset making 10 clusters in actual MNIST data set
display_digit(digits,cil[0][:20])                                       #display some 20 digits belongs to our first cluuster