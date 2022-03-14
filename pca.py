import matplotlib.pyplot as plt
import torch
import pandas as pd


tbl = pd.read_csv("/Users/natalarykova/Desktop/mbnb.tsv", sep="\t")
a = torch.tensor(tbl[["csMBC5","csMBC6","ncsMBC3","NBCB7","NBCB8","NBCB9"]].to_numpy())
a = a.transpose(0,1)
[u,s,v] = torch.pca_lowrank(a)

print(f"a:{a.size()}")
print(u.size())
print(s.size())
print(f"v:{v.size()}")

proj=torch.matmul(a,v[:,:2])
print(proj.size())
plt.scatter(proj[:,0], proj[:,1])
plt.show()