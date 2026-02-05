import pandas as pd
house=pd.read_csv("Housing.csv")#read a file

list1=["bathrooms","stories"] # list for numeric data

house[list1]=house[list1].fillna(2)# fill the missing value
house["airconditioning"]=house["airconditioning"].dropna# remove a missing values

# covert 0 and 1 for the model learn we use one heart,ordinary,replace,map

