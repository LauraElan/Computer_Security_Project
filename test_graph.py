import matplotlib.pyplot as plt
fig= plt.figure(figsize=(6,3))

axes= fig.add_axes([0.1,0.1,0.8,0.8])

x= [1,2,3,4,5]

y=[x**2 for x in x]

axes.plot(x,y)
plt.savefig('test_graph.png')
plt.show()