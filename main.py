# step1: read a file

import pandas as pd #  its a library for read a file 
from sklearn.preprocessing import StandardScaler # us for convert max number to mean
from matplotlib import pyplot as plt
import seaborn as sns
mm=pd.read_csv("Housing.csv")# used for read a csv file
list1=["bathrooms","stories"]# tthere is null value in the two columns soo we create a list
mm[list1]=mm[list1].fillna(1)# we fill 1 forr the misiing values
list2=["airconditioning"]# in this columns also there is null valeus in the string types 
mm[list2]=mm[list2].fillna("yes") # we add a yes to the null valus 
# print(mm[mm.duplicated()])# this is used for check the duplicate values and rows
# mani=mm.drop_duplicates() # this is used for remove the duplicates 

#step2:change string to numbers  by using ordinary encoder,one heart encoder,nlp,replace functions
list3=["bathrooms","stories","prefarea","hotwaterheating","guestroom", "basement","mainroad","airconditioning"]#we create a list for convert 0 ans 1
mm[list3]=mm[list3].replace({"yes":0,"no":1})# this is used for convert 1 and 0 


#step3  we used a one heart encoder for convert furnishingstatus into decimal why we  use onehaert encoder iths has more than 2 columns

#mm=pd.get_dummies(mm, columns=["furnishingstatus"] )
mm = pd.get_dummies(mm, columns=["furnishingstatus"],dtype=int)# its onehot encoder to convert into 0 and 1s


#step4  i the datasets there is long number so  he model accuracy is misiing so convert  numbers into  mean by using a normalization and standscalar 
list4=["price","area"]# this a datasets we want to change
mm[list4]=StandardScaler().fit_transform(mm[list4])# we use a standarscaler for change 
#print(mm.head(10)) # round is used for print the laste 2 decimnals only

#corr=mm.corr()
#plt.figure(figsize=(10,10))
#sns.heatmap(
  #  corr,annot=True,cmap="coolwarm",

#)
##plt.title("relation map")

X = mm[["area","bathrooms","stories","parking","bedrooms"]]
y = mm["price"]
sns.scatterplot(
    data=mm,
    x="area",
    y="price",
    hue="mainroad"
)
plt.title("Price vs Area by Mainroad")
plt.show()
