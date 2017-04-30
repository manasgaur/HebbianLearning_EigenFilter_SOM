from numpy import genfromtxt, linalg
from numpy import zeros, apply_along_axis
import numpy as np
from mySOM import MiniSom
from pylab import plot, axis, show, pcolor, colorbar, bone

features = []
labels = []
codfile = open('Gaur_animal_map.cod','w')
with open('animals_from_jcg.dat') as input_file:
    for line in input_file:
        features.append(map(int,line.split('\t')[0:13]))
        labels.append((line.split('\t')[-1]).rstrip('\n'))
labelarray = np.asarray(labels, dtype=str)
# performing normalization on the feature dataset
norm_data = apply_along_axis(lambda x: x/linalg.norm(x),1,np.asarray(features))
som = MiniSom(10,10,13,sigma=1.0, learning_rate=0.02)
som.random_weights_init(norm_data)
''' cod file creation '''
for wlst in som.weights:
    for i in range(len(wlst)):
        for ele in wlst[i]:
            codfile.write(str(ele))
            codfile.write(' ')
        codfile.write('\n')
codfile.close()
som.train_random(norm_data,10000) # 10000 iterations

'''**************************** Ploting *******************************'''
bone()
''' Distance map as a background '''
pcolor(som.distance_map().T)
colorbar()
''' taking the labels'''
target=zeros(len(labelarray),dtype=int)
target[labelarray == 'dove'] = 0
target[labelarray =='hen'] = 1
target[labelarray =='duck']= 2
target[labelarray =='goose']= 3
target[labelarray =='owl']= 4
target[labelarray =='hawk']= 5
target[labelarray =='eagle']= 6
target[labelarray =='fox']= 7
target[labelarray =='dog']= 8
target[labelarray =='wolf']= 9
target[labelarray =='cat']= 10
target[labelarray =='tiger']= 11
target[labelarray =='lion']= 12
target[labelarray =='horse']= 13
target[labelarray =='zebra']= 14
target[labelarray =='cow']= 15
markers = ['o','v','<','>','1','2','3','4','8','s','p','*','h','+','x','D']
colors =  ['b','g','r','c','m','y','w','b','g','r','c','m','y','w','b','g']

for count, sample in enumerate(norm_data):
    W = som.winner(sample)
    try:
        plot(W[0]+.5,W[1]+.5,markers[target[count]],markerfacecolor='None',markeredgecolor=colors[target[count]], markersize=12,markeredgewidth=2)
    except IndexError:
        newcount=count%(len(target)-1)
        plot(W[0]+.5,W[1]+.5,markers[target[newcount]],markerfacecolor='None',markeredgecolor=colors[target[newcount]], markersize=12,markeredgewidth=2)


axis([0,som.weights.shape[0],0,som.weights.shape[1]])
show()



