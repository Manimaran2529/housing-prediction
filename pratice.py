import pandas as pd
from sklearn.preprocessing import StandardScaler 
from matplotlib import pyplot  as plt
import seaborn  as sns
mm=pd.read_csv("Housing.csv")
list1=["bathrooms","stories"]
mm[list1]=mm[list1].fillna("2")
mm["airconditioning"]=mm["airconditioning"].fillna("yes")
list2=["price","area"]
mm[list2]=StandardScaler().fit_transform(mm[list2])
list3=["bathrooms","stories","prefarea","hotwaterheating","guestroom", "basement","mainroad","airconditioning"]
mm[list3]=mm[list3].replace({"yes":1,"no":0})
mm=pd.get_dummies(mm, columns=["furnishingstatus"] , dtype=int)

corr=mm.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr,annot=True,cmap="coolwarm")
plt.title("heat map")
plt.show()

x=["area","bedrooms","bathrooms","airconditioning","stories","parking","prefarea"]
y=["price"]
