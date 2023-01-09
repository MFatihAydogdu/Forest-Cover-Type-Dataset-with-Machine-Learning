import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.preprocessing import MinMaxScaler
# from numpy import array 
# from sklearn.svm import SVC
# from sklearn import svm, datasets
# from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier
# from numpy.random import randn
# from numpy.random import rand
# from numpy.random import seed
# from numpy import asarray
# from numpy import exp
# from scipy.stats import kurtosis
# from scipy.stats import skew
# from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
import random


veri=pd.read_csv('train1.csv', sep=';')


veri_describe=veri.describe()
#veride, sütun olarak işlevsiz olanlar bulundu(hepsi 0 olan sütunlar) ve veriden çıkartıldı
print(veri.describe())

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
    

def Svm(X_train,X_test):
    
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
   
   accuracyKnn=modelKnn.score(X_curr_test,Y_test)
   
   return accuracyKnn



def SvmAcc(X_train,X_test):
    
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
    
    return accuracySvm;



def dtAcc(X_train,X_test):
    
    modelDt=DecisionTreeClassifier()
    
    modelDt.fit(X_train,Y_train)
    
    accuracyDt=modelDt.score(X_test,Y_test)
    
    return accuracyDt;


# Knn için featurelar ile komşuluk arasındaki en iyi oranı buluyor.
def findBbestKnnVariation():
    
    best=0
    
    for i in range(1,20):
        for j in range(1,20):
            
            ZX_train, ZX_test,bestFeatures =SelectBestFeature(chi2,i)
            
            modelKnnZ,accuracyKnnZ=knn(j,'minkowski',2,ZX_train, ZX_test)
            
            acc=accuracyKnnZ
            
            if(acc>best):
                
                best=acc
                
                best_i=i
                
                best_j=j
                
                best_Features=bestFeatures
            
    
      
    print("En iyi feature seçme sayısı","En iyi komşuluk",best_i,best_j)#19,1
    
    return best,best_Features        



def simulated_annealing(algorithm,X_train,
                        y_train,
                        maxiters=50,
                        alpha=0.85,
                        beta=1,
                        T_0=1,
                        update_iters=1,
                        temp_reduction='geometric'):
    
    
    columns = ['Iteration', 'Feature Count', 'Feature Set', 'Metric', 'Best Metric',
               'Acceptance Probability', 'Random Number', 'Outcome']
    
    results = pd.DataFrame(index=range(maxiters), columns=columns)
    best_subset = None
    hash_values = set()
    T = T_0
    full_set = set(np.arange(len(X_train.columns)))

    # feature sütunlarının %50 si alınarak ilk X_train oluşturuluyor.
    curr_subset = set(random.sample(list(full_set), round(0.5 * len(full_set))))
    X_curr_train = X_train.iloc[:, list(curr_subset)]
    X_curr_test = X_test.iloc[:, list(curr_subset)]
    
        
    prev_metric = algorithm(X_curr_train,X_curr_test)
    best_metric = prev_metric
        
    
        # sıcaklık istenilen değerden daha az bir değere düşerse duruyor.
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

        # Feature işlemleri bittikten sonra yeni set ile eğitim yapılıyor.
        X_new_train = X_train.iloc[:, list(new_subset)]
        X_new_test = X_test.iloc[:, list(new_subset)]
        
        
        
        metric = algorithm(X_new_train,X_new_test)

        if metric > prev_metric:
            print('Local improvement in metric from {:8.4f} to {:8.4f} '
                  .format(prev_metric, metric) + ' - New subset accepted')
            outcome = 'Improved'
            accept_prob, rnd = '-', '-'
            prev_metric = metric
            curr_subset = new_subset.copy()
            
            if metric > best_metric:
                # print('Global improvement in metric from {:8.4f} to {:8.4f} '
                #       .format(best_metric, metric) + ' - Best subset updated')
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

        
        
        results.loc[i, 'Iteration'] = i+1
        results.loc[i, 'Feature Count'] = len(curr_subset)
        results.loc[i, 'Feature Set'] = sorted(curr_subset)
        results.loc[i, 'Metric'] = metric
        results.loc[i, 'Best Metric'] = best_metric
        results.loc[i, 'Acceptance Probability'] = accept_prob
        results.loc[i, 'Random Number'] = rnd
        results.loc[i, 'Outcome'] = outcome
        
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
    
    results = results.dropna(axis=0, how='all')
    return  results,best_metric, best_subset_cols          
                
            
            
            
            
'''
ZX_train, ZX_test,bestFeatures =SelectBestFeature(chi2,19)    

modelKnn,accuracyKnn=knn(1,'minkowski',2,X_train,X_test)

modelKnnZ,accuracyKnnZ=knn(1,'minkowski',2,ZX_train, ZX_test)

modelSvm,accuracySvm=Svm(X_train, X_test)

modelDt,accuracyDt=dt(X_train, X_test)

modelSvmZ,accuracySvmZ=Svm(ZX_train, ZX_test)

modelDtZ,accuracyDtZ=dt(ZX_train, ZX_test)

'''
# results_knnAcc,best_metric_knnAcc,best_subset_cols_knnAcc=simulated_annealing(knnAcc,X_train,Y_train,maxiters=50,alpha=0.85,beta=0.5,T_0=1,update_iters=1,temp_reduction='geometric')

# return  results,best_metric, best_subset_cols
# UnboundLocalError: local variable 'best_subset_cols' referenced before assignment
# resultSvm,best_metric_Svm,best_subset_cols_Svm=simulated_annealing(SvmAcc,X_train,Y_train,maxiters=50,alpha=0.85,beta=0.5,T_0=1,update_iters=1,temp_reduction='geometric')

# result_dtAcc,best_metric_dt,best_subset_cols_dt=simulated_annealing(dtAcc,X_train,Y_train,maxiters=50,alpha=0.85,beta=0.5,T_0=1,update_iters=1,temp_reduction='geometric')

# bestAccKnn,bestFeaturesKnn=findBbestKnnVariation()


'''
# Confusion Matrix Çizdirme
modelKnnPred=modelKnn.predict(X_test)
confusion_matrixx = confusion_matrix(Y_test, modelKnnPred)
plt.figure(figsize = (12,12))
sns.heatmap(confusion_matrixx,annot=True)
# sns.scatterplot(data=veri, x='Elevation', y='Hillshade_3pm', hue='Cover_Type')
'''


'''
 #correlation matrix çizdirme
size = 11 #İlk 11 özellik alındı. Diğer özellikler Soil_type
corrmatrix = veri.iloc[:,:size].corr()
f,ax = plt.subplots(figsize=(10,8))
sns.heatmap(corrmatrix,vmax=0.8,annot = True)
'''

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


# plt.figure(figsize = (10,10))
# sns.scatterplot("Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology",hue = "Cover_Type",data = veri)






