#!/usr/bin/python3.8
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import collections
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import sys
import time
import codecs
sys.stdout = codecs.getwriter('utf8')(sys.stdout.buffer)


### Przygotowanie danych ###
print("\n\n### Przygotowanie danych ###")

#Załadowanie danych z pliku
df=pd.read_csv('ad.data', low_memory=False, header=None)
print(u"Przegląd danych od razu po ich wczytaniu:")
print("{}".format(df.head()))

#Zmiana nazw szczegolnych kolumn (ratio - (width/height))
df.rename({0:'height', 1:'width', 2:'ratio', 3:'local', 1558:'target'}, axis=1, inplace=True)
print(u'\nZmiana nagłówków szczególnych kolumn:')
print("{}".format(df.head()))

#Generacja wykresu braków
newdf=df.iloc[:,[0,1,2,3]]
newdf=newdf.applymap(lambda x:'?' in x)
plt.figure(figsize=(17,5))
plt.title("Missing values")
sns.heatmap(newdf,cbar=False,yticklabels=False,cmap='viridis')
plt.savefig('missing_values.png')
print(u"\nWykres braków został wygenerowany")

#Braki w liczbach
print(u"\nBraki w liczbach:")
dataset=df.applymap(lambda x:np.nan if isinstance(x,str) and '?' in x else x)
results=dataset.iloc[:,:].isnull().sum()
null_columns=[]
for index, value in results.iteritems():
    if value !=0:
        print("{} : {}".format(index, value))
        null_columns.append(index)

#Czyszczenie danych
def replace_missing(df):
    for i in df:
        df[i]=df[i].replace('[?]',np.NAN,regex=True).astype('float')
        df[i]=df[i].fillna(df[i].mean())
    return df

df[['height', 'width', 'ratio', 'local']]=replace_missing(df.iloc[:,[0, 1, 2, 3]].copy()).values
df['local']=df['local'].apply(lambda x:round(x))
print(u"\nPrzegląd danych po wyczyszczeniu braków i przekonwertowaniu liczb na float:")
print("{}".format(df.head()))

#Sprawdzenie czy braki nadal występują
print("\nSprawdzenie czy zostały wyczyszczone wszystkie braki:")
results=df.iloc[:,[0, 1, 2, 3]].isnull().sum()
for index, value in results.iteritems():
        print("{} : {}".format(index, value))

#pairplot
sns.pairplot(data=df.iloc[:,[0,1,2,3,1558]], hue='target')
plt.savefig('pairplot.png')
print(u"\nWykres pairplot został wygenerowany")

#Podział na atrybuty i etykiety
features=df.iloc[:,:-1]
labels=df.iloc[:,-1]

#Skalowanie - standaryzacja danych na zyczenie
scaled=StandardScaler()
if len(sys.argv) == 2 and str(sys.argv[1]) == 'true':
    features=scaled.fit_transform(features)
    print(u"\nPrzegląd danych po standaryzacji, gtowych do trenowania:")
else:
    print(u"\nPrzegląd danych, gtowych do trenowania:")
print("{}".format(df.head()))

### Trening i validacja ###

#Podział danych na treningowe i testowe
print(u"\n\n\n### Trening i validacja ###")
xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.30, random_state=8)

#Funkcja uruchamia klasyfikatory z listy i dla każdego generuje raport, mierzy czas klasyfikacji i generacji
#raportu. Na koniec funkcja mierzy wydajność modelu za pomocą algorytmu GridSearch
def do_with_normal_data(classifiers, xtrain, ytrain, xtest, ytest, pga):
    models=collections.OrderedDict()
    for constructor in classifiers:
        model_name = str(str(constructor).split(':')[0])
        t1 = time.perf_counter()
        obj=constructor()
        obj.fit(xtrain, ytrain)

        print('__________________________________________________')
        print('the model - '+str(str(constructor).split(':')[0]))
        print(classification_report(ytest, obj.predict(xtest)))
        t2 = time.perf_counter()
        print(f"time: {t2 - t1:0.4f} seconds")

        gc = GridSearchCV(estimator=obj, param_grid=pga[model_name], scoring ='accuracy', cv=5).fit(xtrain, ytrain)
        print("Best score is: {} with params: {}".format(gc.best_score_, gc.best_params_))

#Funkcja dla każdego klasyfikatora uruchamia selekcję po czym uruchamia klasyfikator i generuje raport, mierzy
#czas selekcji, klasyfikacji i generacji raportu. Na koniec funkcja mierzy wydajność modelu za pomocą algorytmu
#GridSearch
def do_with_filter_data(classifiers, xtrain, ytrain, xtest, ytest, pga):
    models=collections.OrderedDict()
    for constructor in classifiers:
        model_name = str(str(constructor).split(':')[0])
        t1 = time.perf_counter()
        obj=constructor()

        sfs1 = SFS(obj, k_features=3, forward=True, floating=False, scoring='accuracy', cv=5)
        sfs1 = sfs1.fit(xtrain, ytrain)
        xtrain_sfs = sfs1.transform(xtrain)
        xtest_sfs = sfs1.transform(xtest)

        obj.fit(xtrain_sfs, ytrain)
        
        print('__________________________________________________')
        print('the model - ' + model_name + ' with filter')
        print(classification_report(ytest, obj.predict(xtest_sfs)))
        t2 = time.perf_counter()
        print(f"time: {t2 - t1:0.4f} seconds")

        gc = GridSearchCV(estimator=obj, param_grid=pga[model_name], scoring ='accuracy', cv=5).fit(xtrain_sfs, ytrain)
        print("Best score is: {} with params: {}".format(gc.best_score_, gc.best_params_))

#Lista wybranych klasyfikatorów
classifiers=[GaussianNB,SVC,KNeighborsClassifier]

#Tworzenie listy zbiorów parametrów do oceny najlepszego zestawu dla jakości modeli
param_grid_array=dict()
param_grid_array[str(classifiers[0]).split(':')[0]] = [
    {}
]
param_grid_array[str(classifiers[1]).split(':')[0]] = [
    {
        'kernel':['linear'],'random_state':[0]
    },
    {
        'kernel':['rbf'],'random_state':[0]
    },
    {
        'kernel':['poly'],'degree':[1,2,3,4],'random_state':[0]
    }
]
param_grid_array[str(classifiers[2]).split(':')[0]] = [
     {
        'n_neighbors':np.arange(1,50),
        'p':[2]
    }
]

#Klasyfikacja, raport i GridSearch
print(u"\nKlasyfikacja, generacja raportu oraz ocena modelu algorytmem GridSearch:")
do_with_normal_data(classifiers, xtrain, ytrain, xtest, ytest, param_grid_array)

#Selekcja, klasyfikacja, raport i GridSearch
print(u"\nSelekcja, klasyfikacja, generacja raportu oraz ocena modelu algorytmem GridSearch:")
do_with_filter_data(classifiers, xtrain, ytrain, xtest, ytest, param_grid_array)


#Najlepsza klasyfikacja SCV z najlepszymi parametrami 
#(Bez selekcji, bez standaryzacji, kernel linear, random_state 0)
print(u"\n\n### Ostateczny wynik ###")
print(u"\nGeneracja raportu z najlepszą konfiguracją:")
obj=SVC(kernel='linear', random_state=0)
obj.fit(xtrain, ytrain)

print('__________________________________________________')
print('the model - '+str(str(SVC).split(':')[0]))
print(classification_report(ytest, obj.predict(xtest)))

sns.heatmap(pd.crosstab(ytest, obj.predict(xtest)),cmap='coolwarm')
plt.xlabel('predicted')
plt.ylabel('actual')
plt.savefig('confusion_matrix.png')


