import pandas as pd
from sklearn.preprocessing import StandardScaler 
from  matplotlib import  pyplot as plt
import seaborn as sns
mm=pd.read_csv("Housing.csv")
list1=["airconditioning",]
mm[list1]=mm[list1].fillna("yes")
list2=["bathrooms","stories"]
mm[list2]=mm[list2].fillna("1")
list3=["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]
mm[list3]=mm[list3].replace({"yes":1,"no":0})

mm=pd.get_dummies(mm,columns=["furnishingstatus"],dtype=int)

list4=["price","area"]
mm[list4]=StandardScaler().fit_transform(mm[list4])
print(mm.head(10))

corr=mm.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr,annot=True,cmap="coolwarm")
plt.show()

