# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 18:30:29 2021

@author: Coumbiss
"""

import pandas as pd
import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn import preprocessing 
from sklearn import neighbors, metrics


iris = datasets.load_iris()

print(iris)
print(iris.data)
print(iris.feature_names)
print(iris.target)
print(iris.target_names)

x=pd.DataFrame(iris.data)
# définir les noms de colonnes
x.columns=['Sepal_Length','Sepal_width','Petal_Length','Petal_width']
y=pd.DataFrame(iris.target)
y.columns=['Targets']
#########################################################################
new_target =  np.where(iris.target<1, 0, 1)
colormap =np.array(['BLUE','GREEN','CYAN'])
plt.scatter(x.Sepal_Length, x.Sepal_width,c=colormap[new_target],s=40)

new_data = x[['Sepal_Length','Sepal_width']]
#new_data = new_data.astype({new_data['Sepal_Length']:'float64', new_data['Sepal_width']:'float64', new_target:'category'})

plt.scatter(new_data.Sepal_Length, new_data.Sepal_width,
            c=colormap[new_target],s=40)


#svm = SVC(C=1)
svm = SVC(C=1,kernel='linear')
svm.fit(new_data, new_target)
svm.support_vectors_

# 1. Tracer l'hyperplan de marge maximale séparateur
ax = plt.gca()
ax.scatter(new_data.Sepal_Length, new_data.Sepal_width,
            c=colormap[new_target],s=40)
xlim = ax.get_xlim()
w = svm.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(xlim[0], xlim[1])
yy = a*xx - (svm.intercept_[0]/w[1])
plt.plot(xx,yy)
plt.show()

# 2. Evaluer l'algorithme de classification
svm.score(new_data.to_numpy()[:,0:2],new_target.astype('int'))
svm.predict(new_data.to_numpy()[:,0:2].astype('int'))
sm.confusion_matrix(new_target.astype('int'), svm.predict(new_data.to_numpy()[:,0:2]))
sm.plot_confusion_matrix(svm,new_data.to_numpy()[:,0:2],new_target.astype('int'))  

# 3. Choisir le C optimal en utilisant la Validation Croisée 
X_train, X_test, y_train, y_test = model_selection.train_test_split(new_data, new_target, test_size=0.3)
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)

#Fixer les vvaleurs des hyperparamètres à tester
param_grid = {'n_neighbors':[3,5,7,9,11,13,15]}
#Choisir le score à optimiser
score = "accuracy"
clf = model_selection.GridSearchCV(
        neighbors.KNeighborsClassifier(),
        param_grid,
        cv=5,
        scoring=score)
clf.fit(X_train_std, y_train)

print("Meilleur(s) hyperparamètre(s) sur le jeu d'entrainement :")
print(clf.best_params_)

print("Résultats de la validation croisée :")
for mean, std, params in zip(clf.cv_results_['mean_test_score'],
                             clf.cv_results_['std_test_score'],
                             clf.cv_results_['params']) :
    print("{},{:.3f},{:.03f} for{}".format(score, mean, std*2, params ))

y_pred = clf.predict(X_test_std)
print("\nSur le jeu de test:{:.3f}".format(metrics.accuracy_score))
