import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
mm=pd.read_csv("Housing.csv")


list1=["bathrooms","stories",]
mm[list1]=mm[list1].fillna(2)

mm["airconditioning"]=mm["airconditioning"].fillna("yes")

list2=["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]
mm[list2] = mm[list2].replace({"yes": 1, "no": 0})
print(mm["furnishingstatus"].unique())
mm=pd.get_dummies(mm,columns=["furnishingstatus"],dtype=int)



#cor=mm.corr()
#plt.figure(figsize=(10,10))
#sns.heatmap(cor,annot=True,cmap="coolwarm")
#plt.title("corelation map")
#plt.show()


x=mm[["area","bedrooms","bathrooms","stories","airconditioning","parking","prefarea","mainroad"]]
y=mm["price"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)



pd_x=PowerTransformer(method="yeo-johnson")
x_train_pd=pd_x.fit_transform(x_train)
x_test_pd=pd_x.transform(x_test)

pd_y=PowerTransformer(method="yeo-johnson")
y_train_resize=y_train.values.reshape(-1,1)
y_test_resize=y_test.values.reshape(-1,1)

y_train_pd=pd_y.fit_transform(y_train_resize)
y_test_pd=pd_y.transform(y_test_resize)


lr=LinearRegression()
lr.fit(x_train_pd,y_train_pd)

y_pre=lr.predict(x_test_pd)

print("r2score",r2_score(y_test_pd,y_pre))
area = float(input("Enter area (sqft): "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))
stories = int(input("Enter number of stories: "))
airconditioning = int(input("Do you want AC? yes=1, no=0: "))
parking = int(input("Enter parking (0/1/2): "))
prefarea = int(input("Preferred area? yes=1, no=0: "))
mainroad = int(input("Near main road? yes=1, no=0: "))
mani = [[
    area,
    bedrooms,
    bathrooms,
    stories,
    airconditioning,
    parking,
    prefarea,
    mainroad
]]

pd_mani=pd_x.transform(mani)

pd_predict=lr.predict(pd_mani)

maran=pd_y.inverse_transform(pd_predict)

print("house price",maran[0][0])