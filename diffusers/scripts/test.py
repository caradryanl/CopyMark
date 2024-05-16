import numpy as np

y = np.array([0, 1, 0, 1, 1])
y_pred = np.array([0, 1, 0, 0, 1])

print(y[y_pred>0.5])
print(y_pred[y>0.5])