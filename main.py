import matplotlib.pyplot as plt

# Python list generator
x = [i for i in range(10)]
y = [2*i for i in range(10)]

# Plotting data points
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
# Show datapoints -> scatter
plt.scatter(x, y)
# Show regression lines -> plot
plt.plot(x, y)
plt.show()




