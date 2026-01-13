import pandas as pd
mm=pd.read_csv("Housing.csv")

sns.scatterplot(
    data=mm,
    x="area",
    y="price",
    hue="mainroad"
)
plt.title("Price vs Area by Mainroad")
plt.show()
