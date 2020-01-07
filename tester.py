# -*- coding: utf-8 -*-
"""
Created on Tue Jan 07 18:02:10 2020

@author: S3684970
"""

import mlp_trial as mlp
import matplotlib.pyplot as plt

nn = mlp.MLP(2,5000,1)

for i in range(1000):
    nn.train([0.,0.],[0.])
    nn.train([1.,1.],[0.])
    
    nn.train([1.,0.],[1.])
    nn.train([0.,1.],[1.])

res = mlp.np.zeros((100,100))
for i in range(100):
    for j in range(100):
        res[i][j] = nn.feedforward([i/100,j/100])

print(res)
plt.imshow(res,cmap="Greys")
plt.show()