gt = [4, 4, 4, 3, 3, 6, 1, 2, 2]
pred = [[1, 5], [2, 4], [3, 3], [1, 3], [2, 2], [4, 2], [1, 1], [3, 1], [2, 0]]

arr = []
for i in range(len(pred)):
    arr.append(gt[i] - (1 + (((pred[i][0] + pred[i][1]) / 2) ** 2)))
print(sum(arr) / len(gt))


import pandas as pd
df = pd.read_csv("test.csv", header=None)

print(df[2])