import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
mm=pd.read_csv("HOUsing.csv")

list1=["bathrooms"]
list1.append("stories")
mm[list1]=mm[list1].fillna(1)
mm["airconditioning"]=mm["airconditioning"].fillna("yes")

list2=["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]
mm[list2]=mm[list2].replace({"yes":1,"no":0})
mm=pd.get_dummies(mm,columns=["furnishingstatus"],dtype=int)
#print(mm.dtypes)

#cor=mm.corr()
#plt.figure(figsize=(10,10))
#sns.heatmap(cor,annot=True,cmap="coolwarm")
#plt.show()


x = mm.drop(columns=["price"])
y = mm["price"]


#sns.histplot(mm["area"],bins=20)
#plt.title("area")
#plt.ylabel("price")
#plt.xlabel("area")
#plt.show()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

DecisionTree=DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
    )
DecisionTree.fit(x_train,y_train)
y_test_pd=DecisionTree.predict(x_test)
y_train_pd=DecisionTree.predict(x_train)


print(" train r2_score",r2_score(y_train,y_train_pd))
print("test r2_score",r2_score(y_test,y_test_pd))

corrs=cross_val_score(
    DecisionTree,x,y,cv=5,scoring="r2"
)

print(corrs)