import matplotlib.pyplot as plt

area = [500, 800, 1000, 1500, 2000]
price = [20, 30, 45, 70, 100]

plt.scatter(area, price)

plt.xlabel("Area of House (sq ft)")
plt.ylabel("Price of House (Lakhs)")
plt.title("Simple Example: Area vs Price")

plt.show()

