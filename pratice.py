import pandas as pd
from sklearn.preprocessing import StandardScaler,PowerTransformer
from sklearn.model_selection import train_test_split

from matplotlib  import pyplot as plt
import seaborn as sns
mm=pd.read_csv("Housing.csv")


list1=["bathrooms","stories"]
mm[list1]=mm[list1].fillna(2)

mm["airconditioning"]=mm["airconditioning"].fillna("yes")



list2=["prefarea","hotwaterheating","guestroom", "basement","mainroad","airconditioning"]#we create a list for convert 0 ans 1
mm[list2]=mm[list2].replace({"yes":1,"no":0})


mm=pd.get_dummies(mm ,columns=["furnishingstatus"],dtype=int)


#cor=mm.corr()
#plt.figure(figsize=(10,10))
#sns.heatmap(cor ,annot=True,cmap="coolwarm")
##plt.title("correlation map")

x=mm[["area","bedrooms","bathrooms","stories","mainroad","airconditioning","parking","prefarea"]]
y=mm["price"]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=45)

pt=PowerTransformer(method="yeo-johnson")
X_train=pt.fit_transform(x_train)
X_test=pt.transform(x_test)



pt_x = PowerTransformer(method="yeo-johnson")

Y_train_resize=y_train.values.reshape(-1,1)
Y_test_resize=y_test.values.reshape(-1,1)

Y_train_pt=pt.fit_transform(Y_train_resize)
Y_test_pt=pt.transform(Y_test_resize)

sns.histplot(Y_train_pt)
plt.title("after PowerTransformer")
plt.show()


