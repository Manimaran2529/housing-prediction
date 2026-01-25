import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from matplotlib  import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

mm=pd.read_csv("Housing.csv")
list1=["bathrooms","stories"]
mm[list1]=mm[list1].fillna(2)

mm["airconditioning"]=mm["airconditioning"].fillna("yes")



list2=["prefarea","hotwaterheating","guestroom", "basement","mainroad","airconditioning"]#we create a list for convert 0 ans 1
mm[list2] = mm[list2].replace({"yes": 1, "no": 0}).infer_objects(copy=False)



mm=pd.get_dummies(mm ,columns=["furnishingstatus"],dtype=int)


#cor=mm.corr()
#plt.figure(figsize=(10,10))
#sns.heatmap(cor ,annot=True,cmap="coolwarm")
##plt.title("correlation map")

x=mm[["area","bedrooms","bathrooms","stories","mainroad","airconditioning","parking","prefarea"]]
y=mm["price"]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)

pipe=Pipeline([
    ("PowerTransformer",PowerTransformer(method="yeo-johnson")),
   ("LinearRegression",LinearRegression())
])


pipe.fit(x_train,y_train)

y_pre=pipe.predict(x_test)

print("r2score:",r2_score(y_test,y_pre))

