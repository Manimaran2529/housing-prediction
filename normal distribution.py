import pandas as pd
from matplotlib  import pyplot as plt
import seaborn as sns
from sklearn .preprocessing import PowerTransformer ,StandardScaler

mm=pd.read_csv("Housing.csv")
list1=["bathrooms","stories"]
mm[list1]=mm[list1].fillna(2)
mm["airconditioning"]=mm["airconditioning"].fillna("yes")



ts=PowerTransformer(method="yeo-johnson")
list2=["price","area",]
mm[list2]=ts.fit_transform(mm[list2])

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.histplot(mm["price"])
plt.title("Price Distribution")

plt.subplot(1,2,2)
sns.histplot(mm["area"])
plt.title("Area Distribution")



list3=["bathrooms","stories","prefarea","hotwaterheating","guestroom", "basement","mainroad","airconditioning"] 
mm[list3]=mm[list3].replace({"yes":1,"no":0})

mm=pd.get_dummies(mm,columns=["furnishingstatus"],dtype=int)

plt.figure(figsize=(10,10))

corr=mm.corr()
sns.heatmap(corr,annot=True,cmap="coolwarm")
plt.show()


x=["area","bedrooms","bathrooms","stories","mainroad","hotwaterheating","airconditioning","parking","prefarea"]
y=["price"]


