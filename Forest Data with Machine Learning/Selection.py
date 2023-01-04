import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression
from numpy import array 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed
from numpy import asarray
from numpy import exp
import random
from scipy.stats import kurtosis
from scipy.stats import skew

kurtosis.mode=True
veri=pd.read_csv('train1.csv', sep=';')



#veride, sütun olarak işlevsiz olanlar bulundu(hepsi 0 olan sütunlar) ve veriden çıkartıldı
# print(veri.describe())

veri= veri.drop(['Id','Soil_Type7','Soil_Type15'],axis=1)


Y=veri["Cover_Type"]
Y=Y.to_frame()
X = veri.drop('Cover_Type',axis=1)


allFeatures=X.columns



X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)


def SelectBestFeature(sFunction,k):
   
    
    # sFunction=chi2 || f_regression
    select = SelectKBest(sFunction, k=k)
    z = select.fit_transform(X,Y)
    zDF=pd.DataFrame(z,columns=select.get_feature_names_out())
    
    
    
    X_train, X_test= train_test_split(zDF,test_size=0.3,random_state=42)
    
    return X_train, X_test,zDF.columns;

def knn(n,func,p,X_train,X_test):
          
   modelKnn= KNeighborsClassifier(n_neighbors=n, metric=func, p = p)# p=1 manhattan_distance p=2 euclidean
   
   modelKnn.fit(X_train, Y_train)
   
   accuracy=modelKnn.score(X_test, Y_test)
   
   return modelKnn,accuracy;
    

def svm(X_train,X_test):
    modelSvm=SVC()
     #kernel='poly'=61.5--kernel='rbf'=62.56
     #C=1.0 ise 62.56
     #C=10.0 ise 68.38
     #C=20.0 ise 70.08
     #C=50.0 ise 71.50
     #C=100.0 ise 72.81
     #C=5000.0 ise 76.43
    modelSvm.fit(X_train, Y_train)
    
    accuracySvm=modelSvm.score(X_test, Y_test)
    
    return modelSvm,accuracySvm;
    

def dt(X_train,X_test):
    modelDt=DecisionTreeClassifier()
    
    modelDt.fit(X_train,Y_train)
    
    accuracyDt=modelDt.score(X_test,Y_test)
    
    return modelDt,accuracyDt;

def knnAcc(X_curr_train,X_curr_test):
          
   modelKnn= KNeighborsClassifier(n_neighbors=1, metric='minkowski', p = 2)# p=1 manhattan_distance p=2 euclidean
   
   
   modelKnn.fit(X_curr_train,Y_train)
   
   # modelKnnPred=modelKnn.predict(X_test)
   
   accuracy=modelKnn.score(X_curr_test,Y_test)
   
   return accuracy




def f(function_name):
    best=0
    
    for i in range(1,10):
        for j in range(1,10):
            ZX_train, ZX_test,bestFeatures =SelectBestFeature(chi2,i)
            modelKnnZ,accuracyKnnZ=function_name(j,'minkowski',2,ZX_train, ZX_test)
            acc=accuracyKnnZ
            
            if(acc>best):
                best=acc
                best_i=i
                best_j=j
                best_Features=bestFeatures
            
      
    print("En iyi feature seçme sayısı","En iyi komşuluk",best_i,best_j)#19,1
    return best,best_Features        
            
                
            
            
    
# bestAccKnn,bestFeaturesKnn=f(knn)



modelKnn,accuracyKnn=knn(1,'euclidean',2,X_train,X_test)
modelKnn2,accuracyKnn2=knn(2,'minkowski',2,X_train,X_test)

# ZX_train, ZX_test,bestFeatures =SelectBestFeature(chi2,19)
# ZX_train, ZX_test,bestFeatures =SelectBestFeature(f_regression,19)

# modelKnnZ,accuracyKnnZ=knn(50,'minkowski',1,ZX_train, ZX_test)



'''
ZX_train, ZX_test,zDF =SelectBestFeature(chi2,20)
modelKnnZ,accuracyKnnZ=knn(1,'minkowski',2,ZX_train, ZX_test) 

modelKnn,accuracyKnn=knn(1,'minkowski',2,X_train,X_test)


modelDt,accuracyDt=dt(X_train,X_test)

modelDtZ,accuracyDtZ=dt(ZX_train,ZX_test)


# modelSvmZ,accuracySvmZ=svm(ZX_train, ZX_test)
    
# modelSvm,accuracySvm=svm(X_train,X_test)
'''
# sns.scatterplot(data=veri, x='Elevation', y='Hillshade_3pm', hue='Cover_Type')

##############################################
# clf_rf_5 = RandomForestClassifier()      
# clf_rf_5.fit(X_train,Y_train)
# importances = clf_rf_5.feature_importances_

# std = np.std([tree.feature_importances_ for tree in clf_rf_5.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]

# print("Feature ranking:")

# for f in range(X_train.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# plt.figure(1, figsize=(14, 13))
# plt.title("Feature importances")
# plt.bar(range(X_train.shape[1]), importances[indices],
#        color="g", yerr=std[indices], align="center")
# plt.xticks(range(X_train.shape[1]), X_train.columns[indices],rotation=90)
# plt.xlim([-1, X_train.shape[1]])
# plt.show()

##############################################
'''
full_set=set(np.arange(len(X_train.columns)))
best = set(random.sample(list(full_set), round(0.5 * len(full_set))))
def simulated_annealing1(objective, bounds, n_iterations, step_size, temp):
	full_set=set(np.arange(len(X_train.columns)))
    # generate an initial point
	best = set(random.sample(list(full_set), round(0.5 * len(full_set))))
	# evaluate the initial point
	best_eval = objective(best)
	# current working solution
	curr, curr_eval = best, best_eval
	# run the algorithm
	for i in range(n_iterations):
		# take a step
		candidate = curr + randn(len(bounds)) * step_size
		# evaluate candidate point
		candidate_eval = objective(candidate)
		# check for new best solution
		if candidate_eval < best_eval:
			# store new best point
			best, best_eval = candidate, candidate_eval
			# report progress
			print('>%d f(%s) = %.5f' % (i, best, best_eval))
		# difference between candidate and current point evaluation
		diff = candidate_eval - curr_eval
		# calculate temperature for current epoch
		t = temp / float(i + 1)
		# calculate metropolis acceptance criterion
		metropolis = exp(-diff / t)
		# check if we should keep the new point
		if diff < 0 or rand() < metropolis:
			# store the new current point
			curr, curr_eval = candidate, candidate_eval
	return [best, best_eval]
 
# seed the pseudorandom number generator
seed(1)
# define range for input
bounds = asarray([[-5.0, 5.0]])
# define the total iterations
n_iterations = 1000
# define the maximum step size
step_size = 0.1
# initial temperature
temp = 10
# perform the simulated annealing search
# best, score = simulated_annealing(knnAcc, bounds, n_iterations, step_size, temp)
# print('Done!')
# print('f(%s) = %f' % (best, score))
'''
##############################################


def simulated_annealing(X_train,
                        y_train,
                        maxiters=75,
                        alpha=0.85,
                        beta=1,
                        T_0=1,
                        update_iters=1,
                        temp_reduction='geometric'):
    
    best_subset = None
    hash_values = set()
    T = T_0
    full_set = set(np.arange(len(X_train.columns)))

    n=1
    func='minkowski'
    p=2
    # Generate initial random subset based on ~50% of columns
    curr_subset = set(random.sample(list(full_set), round(0.5 * len(full_set))))
    X_curr_train = X_train.iloc[:, list(curr_subset)]
    X_curr_test = X_test.iloc[:, list(curr_subset)]
    prev_metric = knnAcc(X_curr_train,X_curr_test)
    best_metric = prev_metric

    for i in range(maxiters):
        if T < 0.01:
            print(f'Temperature {T} below threshold. Termination condition met')
            break

        while True:
            if len(curr_subset) == len(full_set): 
                move = 'Remove'
            elif len(curr_subset) == 2: # Not to go below 2 features
                move = random.choice(['Add', 'Replace'])
            else:
                move = random.choice(['Add', 'Replace', 'Remove'])
            
            pending_cols = full_set.difference(curr_subset) 
            new_subset = curr_subset.copy()   

            if move == 'Add':        
                new_subset.add(random.choice(list(pending_cols)))
            elif move == 'Replace': 
                new_subset.remove(random.choice(list(curr_subset)))
                new_subset.add(random.choice(list(pending_cols)))
            else:
                new_subset.remove(random.choice(list(curr_subset)))
                
            if new_subset in hash_values:
                print('Subset already visited')
            else:
                hash_values.add(frozenset(new_subset))
                break

        X_new_train = X_train.iloc[:, list(new_subset)]
        X_new_test = X_test.iloc[:, list(new_subset)]
        metric = knnAcc(X_new_train,X_new_test)

        if metric > prev_metric:
            print('Local improvement in metric from {:8.4f} to {:8.4f} '
                  .format(prev_metric, metric) + ' - New subset accepted')
            outcome = 'Improved'
            accept_prob, rnd = '-', '-'
            prev_metric = metric
            curr_subset = new_subset.copy()
            
            if metric > best_metric:
                print('Global improvement in metric from {:8.4f} to {:8.4f} '
                      .format(best_metric, metric) + ' - Best subset updated')
                best_metric = metric
                best_subset = new_subset.copy()
                
        else:
            rnd = np.random.uniform()
            diff = prev_metric - metric
            accept_prob = np.exp(-beta * diff / T)

            if rnd < accept_prob:
                print('New subset has worse performance but still accept. Metric change' +
                      ':{:8.4f}, Acceptance probability:{:6.4f}, Random number:{:6.4f}'
                      .format(diff, accept_prob, rnd))
                outcome = 'Accept'
                prev_metric = metric
                curr_subset = new_subset.copy()
            else:
                print('New subset has worse performance, therefore reject. Metric change' +
                      ':{:8.4f}, Acceptance probability:{:6.4f}, Random number:{:6.4f}'
                      .format(diff, accept_prob, rnd))
                outcome = 'Reject'

        

        # Temperature cooling schedule
        if i % update_iters == 0:
            if temp_reduction == 'geometric':
                T = alpha * T
            elif temp_reduction == 'linear':
                T -= alpha
            elif temp_reduction == 'slow decrease':
                b = 5 # Arbitrary constant
                T = T / (1 + b * T)
            else:
                raise Exception("Temperature reduction strategy not recognized")
    
    if(best_subset!=None):
        best_subset_cols = [list(X_train.columns)[i] for i in list(best_subset)]            
    # best_subset_cols = [list(X_train.columns)[i] for i in list(best_subset)]
    

    return  best_metric, best_subset_cols



# best_metric,best_subset_cols=simulated_annealing(X_train,Y_train,maxiters=50,alpha=0.85,beta=1,T_0=1,update_iters=1,temp_reduction='geometric')





















# #correlation matrix çizdirme
# size = 11 #İlk 11 özellik alındı. Diğer özellikler Soil_type
# corrmatrix = veri.iloc[:,:size].corr()
# f,ax = plt.subplots(figsize=(10,10))
# sns.heatmap(corrmatrix,vmax=0.8,square = True)






#istenilen veri tipinden ,kaç adet veriden kaç adet var
# target = 'Cover_Type'
# def plot(df, name):
#     count = df[target].value_counts.sort_index()
#     plt.ticklabel_format(useOffset=False, style='plain')
#     plt.bar(count.index, count)
#     plt.xlabel('label/class')
#     plt.title(name)
#     plt.show()
# plot(veri,'veri')
# plot(Y_train,'Y_train')




# def objective_function(solution):
#   model = KNeighborsClassifier()
#   #a=sum(solution,keepdims=True)
#   if(np.sum(solution ,keepdims=True)==0):
#     return 0
#   model.fit(X_train.loc[:,solution],Y_train)
#   accuracy = model.score(X_test.loc[:,solution],Y_test)
#   return accuracy




# best_obj = 0

# for i in range(10):
#   solution = np.random.random(55)>0.5
#   obj_val = objective_function(solution)
#   if(obj_val>best_obj):
#     best_obj = obj_val
#     best_sol = solution.copy()
#   print(best_obj)
#   print(solution)

#Normalizasyon

# mms = MinMaxScaler()
# X_train_S = mms.fit_transform(X_train)
# X_test_S = mms.transform(X_test)
# Y_train_S=mms.fit_transform(Y_train)
# Y_test_S = mms.transform(Y_test)

# X_train_S=pd.DataFrame(X_train_S)
# X_test_S=pd.DataFrame(X_test_S)
# Y_train_S=pd.DataFrame(Y_train_S)
# Y_test_S=pd.DataFrame(Y_test_S)

#Normalizasyonlu modelKnn
# modelKnn_S= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)
# modelKnn_S.fit(X_train_S, Y_train_S)
# accuracyKnn_S=modelKnn.score(X_test_S, Y_test_S)
