import numpy as np
import operator
import pandas


#bu kısımda csv dosyası okunuyor
def load_csv(filename):
    dataset = np.array(pandas.read_csv(filename,sep=',')) 
    for i in range(len(dataset)):    
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

#bu kısımda euclidean distance hesaplanıyor    
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return np.sqrt(distance)

#en yakın k tane komşusu bulunuyor
def find_knn(k,item,data):
    dist=[]
    for i in range(len(data)):
        d=euclideanDistance(item,data[i],len(data[0])-1)
        dist.append((i,d))
    dist.sort(key=operator.itemgetter(1))
    return dist[0:k]

#knnClassifier uygulanıyor    
def knnClassifier(X,Y,k=1):
    acc = 0
    flag = 0
    for y in Y:
        idx = find_knn(k,y,X)
        for i in idx:
            #print(i,'|',X[i[0]][-1],'|',y[-1])
            if(X[i[0]][-1]==y[-1]):
                flag+=1
            #print(flag , k)
        if(flag>=int(k/2)+1):
            acc+=1
        flag=0
    return acc /len(Y)
    
X,Y=load_csv('playStoreGamesTrain.csv'),load_csv('playStoreGamesTest.csv')


print('accuracy is: %',knnClassifier(X,Y,1)*100)