# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 20:02:01 2021

@author: Coumbiss
"""
#Importation des librairies
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn as sns
from sklearn.linear_model import RidgeCV, LassoCV
sns.set_style('darkgrid')


#QUESTION 1
#fonction regression(X, Y) qui renvoie l’estimateur des moindre carrés

def regression(X,Y):
    return np.dot( np.linalg.pinv(np.dot(X.transpose(),X)),np.dot(X.transpose(), Y) )

#Utiliser votre fonction de régression sur le jeu de données Boston House Prices
    
donnees = datasets.load_boston()

#Exploration des données

print(donnees)
print(donnees.data)
print(donnees.feature_names)
print(donnees.target)
X, Y = donnees.data, donnees.target
X.shape
Y.shape
Y = Y.reshape(Y.shape[0], 1)
identity = np.ones((X.shape[0],1))
X = np.insert(identity,[-1], X, axis=1)
regression(X,Y)

alpha = regression(X,Y)[:-1]
beta = regression(X,Y)[-1]

lm = linear_model.LinearRegression()
lm.fit(X,Y)
lm.coef_
lm.intercept_

#Comparaison
#alpha = lm.coef_
#beta = lm.intercept_

#QUESTION 2
#Écrire une fonction regress(X,alpha, beta) qui renvoie le vecteur Y_chapeau

def regress(X, alpha, beta):
    return np.dot(X, alpha) + beta


#QUESTION 3
#Calculer l'erreur au sens des moindres carrés du regresseur
Y_chapeau = regress(donnees.data, alpha, beta)

def error(Y, Y_chapeau):
    return np.sum((Y - Y_chapeau)**2)

model_erreur = error(Y, Y_chapeau)
model_erreur
#QUESTION 4
# (a) Programmez une fonction ridge_regression(X, Y , lambda_)

I = np.identity(X.shape[1])

def ridge_regression(X,Y,lambda_):
    return np.dot(np.dot(np.linalg.pinv(np.dot(X.T,X)+(lambda_*I)), X.T), Y)

#Application sur le jeu de données

lambda_ = 1
ridge_regression(X, Y, lambda_)

alpha_ridge = ridge_regression(X,Y, lambda_)[:-1]
beta_ridge = ridge_regression(X,Y, lambda_)[-1]

rm = linear_model.Ridge()
rm.fit(X,Y)
rm.coef_
rm.intercept_

#Comparaison
#rm.coef_ sensiblement égal à alpha_ridge
#rm.intercept_ = 2* beta_ridge


#(b) Tracez l’évolution des coefficients du vecteur alpha en fonction du paramètre de régularisation
#lambda pour des valeurs entre 0.001 et 1000
n_alphas = 200
alphas = np.logspace(-2, 3, n_alphas)

    
coefs = []
for l_ in alphas:  
    ridge = linear_model.Ridge(alpha=l_, fit_intercept=False)
    ridge.fit(X, Y)
    coefs.append(ridge.coef_)
    
coefs = np.array(coefs) 
coefs = coefs.reshape(coefs.shape[0], -1)



ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('lambda')
plt.ylabel('coefficients')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.legend(labels = ['CRIM', 'ZN', 'INDUS',  'CHAS',  'NOX','RM', 'AGE', 'DIS', 'RAD', 'TAX',  'PTRATIO', 'B' ,'LSTAT' , 'Price'])
plt.show()
#b_Variables les plus influentes
d = pd.DataFrame(donnees.data, columns = donnees.feature_names)
d['Price'] = donnees.target

corr = d.corr()
#Plot figsize
fig, ax = plt.subplots(figsize=(8, 8))
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(corr, cmap='RdBu', annot=True, fmt=".2f")
#Apply xticks
plt.xticks(range(len(corr.columns)), corr.columns);
#Apply yticks
plt.yticks(range(len(corr.columns)), corr.columns)
#show plot
plt.show()

"""les variables plus influentes sont respectivement: CHAS(0.18),
 DIS(0.25), B(0.33), ZN(0.36) et RM(0.70)"""
 
#c) Meilleure valeur pour le paramétre lambda
 
regr_cv = RidgeCV(alphas )
# Fit the linear regression
model_cv = regr_cv.fit(X, Y) 
# View alpha
model_cv.alpha_

#Calculez l'erreur au sens des moindres carrés

Y_pred = model_cv.predict(X) 
ridge_error = error(Y, Y_pred)
ridge_error
 

#QUESTION 5
#(a) En utilisant la classe linear_model.Lasso, tracez l’évolution des coefficients du vecteur alpha
#en fonction de la valeur du paramètre lambda.
 
 
 
n_alphas = 200
alphas = np.logspace(-2, 3, n_alphas)

    
coefs_lasso = []
for alpha in alphas:  
    lasso = linear_model.Lasso(alpha=alpha, fit_intercept=False)
    lasso.fit(X, Y)
    coefs_lasso.append(ridge.coef_)
    
coefs_lasso = np.array(coefs_lasso) 
coefs_lasso = coefs_lasso.reshape(coefs_lasso.shape[0], -1)



ax = plt.gca()
ax.plot(alphas, coefs_lasso)
ax.set_xscale('log')
plt.xlabel('lambda')
plt.ylabel('coefficients')
plt.title('Lasso coefficients as a function of the regularization')
plt.axis('tight')
plt.legend(labels = ['CRIM', 'ZN', 'INDUS',  'CHAS',  'NOX','RM', 'AGE', 'DIS', 'RAD' 'TAX',  'PTRATIO', 'B' ,'LSTAT' , 'Price'])
plt.show()

 
"""les variables plus influentes sont respectivement: CHAS(0.18),
DIS(0.25), B(0.33), ZN(0.36) et RM(0.70).
Oui, elles sont les mêmes que celles trouvées au niveau de 
la question précédente.
Les autres variables restent constantes lorsque la valeur de lambda augmente."""

#b) Meilleure valeur pour le paramétre lambda

regr_lasso = LassoCV(cv = 20 )
# Fit the linear regression
model_lasso = regr_lasso.fit(X, Y) 

# View alpha
model_lasso.alpha_ 

#Calculez l'erreur au sens des moindres carrés
Y_predLasso = model_lasso.predict(X)
lasso_error = error(Y, Y_predLasso)
lasso_error
 
 
 
 
 
 
 
 
 
 
 