import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
mm=pd.read_csv("Housing.csv")

#bivaritane graph 

# histogram is used for figure out the one  numerical data

#sns.histplot(mm["bedrooms"],bins=20 ,kde=True)
#plt.title("number of a bedrooms in a house")
#plt.xlabel("bedroomns present")
#plt.ylabel("number of a house")
#plt.show()

#pie chart i used for a single caterogical data we can not use the seabron in there 

#pie=mm["mainroad"].value_counts()
#plt.pie(pie,labels=["maniroad have ","main road did not have"],autopct="%1.1f%%") # here auto pic used for a show a precentage in a pie chart
#plt.title("how many house have a main road")
#plt.show()

#scatter plot is used for figure out the numerical vs numerical data 


#sns.scatterplot(x="bedrooms",y="price",data=mm)
#plt.title("Number of bedrooms and price   in a house ")
#plt.xlabel("number of a bedrooms")
#plt.ylabel("price of a hosue")
#plt.show()

#box chart is used for compare a caterogical data vs numerical data before that we want to conver the datasets into a 0 and 1

    
#sns.boxplot(
# x="mainroad",
#    y="price",
#    data=mm
#)

#plt.title("House Price vs Main Road")
#plt.xlabel("Main Road (0 = No, 1 = Yes)")
#plt.ylabel("House Price")
#plt.show()


#grouped bar chat is used for the categorical  vs categorical data 
#sns.countplot(
#    x="mainroad",
 #   hue="guestroom",
  #  data=mm
#)

#plt.title("Main Road vs Guest Room (Grouped Bar Chart)")
#plt.xlabel("Main Road (0 = No, 1 = Yes)")
#plt.ylabel("Number of Houses")
#plt.show()


#multivarite chart more than 2 figure 


