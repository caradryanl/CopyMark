import numpy as np
from sklearn import preprocessing

member_features = np.array([np.nan, np.inf, -np.inf, 1000, 0, 4, 30000, -899])
membermax, membermin = member_features[~np.isposinf(member_features)].max(), member_features[~np.isneginf(member_features)].min()
member_features = np.nan_to_num(member_features, nan=0, posinf=membermax, neginf=membermin)

x = preprocessing.scale(member_features)
x = np.nan_to_num(member_features, nan=0)
print(x, member_features)