"Esse projeto é para estimar qual time de basquete de ensino médio vai vencer um campeonato!"
"E assim, comparar a precisão dos diferentes métodos"
#What are our lables? Round where the given team was eliminated or where their season ended
#(R68 = First Four, R64 = Round of 64, R32 = Round of 32, S16 = Sweet Sixteen, E8 = Elite Eight, F4 = Final Four, 2ND = Runner-up, Champion = Winner of the NCAA March Madness Tournament for that given year)

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
import seaborn as sns
from sklearn.tree import export_graphviz

#Loading Data
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%206/cbb.csv')

#Next we'll add a column that will contain "true" if the wins above bubble are over 7 and "false" if not.
#We'll call this column Win Index or "windex" for short.
df['windex'] = np.where(df.WAB > 7, 'True', 'False')

#Data visualization and pre-processing
df1 = df.loc[df['POSTSEASON'].str.contains('F4|S16|E8', na=False)]

print(df1.head())

print(".")
print("Visualize alguns dados de pós-temporada : ")
print(".")

print(df1['POSTSEASON'].value_counts())

#Veja alguns dados :
bins = np.linspace(df1.BARTHAG.min(), df1.BARTHAG.max(), 10)
g = sns.FacetGrid(df1, col="windex", hue="POSTSEASON", palette="Set1", col_wrap=6)
g.map(plt.hist, 'BARTHAG', bins=bins, ec="k")

g.axes[-1].legend()
#plt.show()

bins = np.linspace(df1.ADJOE.min(), df1.ADJOE.max(), 10)
g = sns.FacetGrid(df1, col="windex", hue="POSTSEASON", palette="Set1", col_wrap=2)
g.map(plt.hist, 'ADJOE', bins=bins, ec="k")

g.axes[-1].legend()
#plt.show()

#Pre-processing:  Feature selection/extraction

#We see that this data point doesn't impact the ability of a team to get into the Final Four. 
bins = np.linspace(df1.ADJDE.min(), df1.ADJDE.max(), 10)
g = sns.FacetGrid(df1, col="windex", hue="POSTSEASON", palette="Set1", col_wrap=2)
g.map(plt.hist, 'ADJDE', bins=bins, ec="k")
g.axes[-1].legend()
#plt.show()

#Convert Categorical features to numerical values

#Dando uma olhada na pós-temporada :
print(df1.groupby(['windex'])['POSTSEASON'].value_counts(normalize=True))

# Faça uma cópia do DataFrame para garantir que você esteja trabalhando no original
df1 = df1.copy()

# Substitua os valores 'False' e 'True' na coluna 'windex' pelos valores 0 e 1, respectivamente
df1['windex'].replace(to_replace=['False', 'True'], value=[0, 1], inplace=True)

#Feature selection : Let's define feature sets, X
X = df1[['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D',
       'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O',
       '3P_D', 'ADJ_T', 'WAB', 'SEED', 'windex']]

#Defining Labels
y = df1['POSTSEASON'].values

#Normalize Data
X= preprocessing.StandardScaler().fit(X).transform(X)

#Training and Validation
# We split the X into train and test to find the best k
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Validation set:', X_val.shape,  y_val.shape)

#Classification
print("")
print("Knearest")
print("")
#Knearest :
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

#Definindo os K médios
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

#Mede Precisão
#Predict
yhat_Kn=neigh.predict(X_val)

mean_acc_Kn = np.zeros((5-1))
std_acc_Kn = np.zeros((5-1))

#Use isso para extrair as métricas de análise
from sklearn import metrics

mean_acc_Kn = metrics.accuracy_score(y_val, yhat_Kn)
print("precisão média : ", mean_acc_Kn)

#jaccard indice
from sklearn.metrics import jaccard_score
print("jaccard index : ", jaccard_score(y_val, yhat_Kn, average='micro'))

#F1 score
from sklearn.metrics import f1_score
print("F1 score : ",f1_score(y_val, yhat_Kn, average='micro'))

#Classification
print("")
print("Decision Tree")
print("")
#Decision Tree :

from sklearn.tree import DecisionTreeClassifier

#Max_depth pode ser ajustado além de outros fatores
Tree = DecisionTreeClassifier(criterion="entropy", max_depth = 5)

Tree.fit(X_train, y_train)

predTree = Tree.predict(X_val)

print("precisão média : ", metrics.accuracy_score(y_val, predTree))

#jaccard indice
from sklearn.metrics import jaccard_score
print("jaccard index : ", jaccard_score(y_val, predTree, average='micro'))

#F1 score
from sklearn.metrics import f1_score
print("F1 score : ",f1_score(y_val, predTree, average='micro'))

#Crie uma forma de vizualização da árvore :
# export_graphviz(Tree, out_file='Projeto Final/treeFinalProject.dot', filled=True, feature_names=['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D', 'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O', '3P_D', 'ADJ_T', 'WAB', 'SEED', 'windex'])

#Depois de rodar o código execute o seguinte no cmd :
# dot -Tpng Downloads\ProgramsToHelpPeople\IAs\TestesCurso\ProjetoFinal\treeFinalProject.dot -o Downloads\ProgramsToHelpPeople\IAs\TestesCurso\ProjetoFinal\treeFinalProject.png
#Para passar a imagem de .dot para .png

#Classification
print("")
print("Support Vector Machine")
print("")
#SVM :

#Modeling with scikit learn
#a tronsformação matematica pode assumir vários formatos, como linear, polinomial, radial e sigmoid
from sklearn import svm

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
precisao_SVM = []
jaccard_SVM = []
F1score_SVM = []

for k in kernels :
       clf_SVM = svm.SVC(kernel=k)
       clf_SVM.fit(X_train, y_train)
       yhat_SVM = clf_SVM.predict(X_val)
       precisao_SVM.append(metrics.accuracy_score(y_val, yhat_SVM))
       jaccard_SVM.append(jaccard_score(y_val, yhat_SVM, average='micro'))
       F1score_SVM.append(f1_score(y_val, yhat_SVM, average='micro'))

precisao_max_SVM = max(precisao_SVM)
posicao_max_SVM = precisao_SVM.index(precisao_max_SVM)

print("O modelo melhor foi : ", kernels[posicao_max_SVM])

print("precisão média : ", precisao_SVM[posicao_max_SVM])

print("jaccard index : ", jaccard_SVM[posicao_max_SVM])

print("F1 score : ", F1score_SVM[posicao_max_SVM])

#Classification
print("")
print("Logistic Regression")
print("")
#Logistic Regression :

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

#Previsão
yhat_LR = LR.predict(X_val)

#Avaliação

print("precisão média : ", metrics.accuracy_score(y_val, yhat_LR))

#jaccard indice
from sklearn.metrics import jaccard_score
print("jaccard index : ", jaccard_score(y_val, yhat_LR, average='micro'))

#F1 score
from sklearn.metrics import f1_score
print("F1 score : ",f1_score(y_val, yhat_LR, average='micro'))

#log loss
yhat_prob = LR.predict_proba(X_val)
from sklearn.metrics import log_loss
print("Log Loss : ", log_loss(y_val, yhat_prob))

#Confusion Matrix é uma forma interessante, também, de mensurar a precisão, mas para esse caso, ficaria muito difícil de fazer, sem tanta necessidade