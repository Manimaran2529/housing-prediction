import pandas as pd
from sklearn .preprocessing import StandardScaler
mm=pd.read_csv("Housing.csv")

list1=["mainroad","basement","hotwaterheating","airconditioning","prefarea","guestroom"]
mm[list1]=mm[list1].replace({"yes":1, "no":0})
mm = pd.get_dummies(mm, columns=["furnishingstatus"])

list2=[ "price","area"]
mm[list2]= StandardScaler().fit_transform(mm[list2])
mm[list2].round(2)
