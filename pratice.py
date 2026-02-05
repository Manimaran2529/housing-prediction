import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
house=pd.read_csv("Housing.csv")#read a file

list1=["bathrooms","stories"] # list for numeric data

house[list1]=house[list1].fillna(2)# fill the missing value
house["airconditioning"]=house["airconditioning"].dropna()# remove a missing values

# covert 0 and 1 for the model learn we use one heart,ordinary,replace,map

list2=["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]
house[list2]=house[list2].replace({"yes":1,"no":0})
house=pd.get_dummies(house ,columns=["furnishingstatus"],dtype=int)


cor=house.corr()
plt.figure(figsize=(10,10))
sns.heatmap(cor,annot=True ,cmap="coolwarm")
plt.title("correlation map")
plt.show()