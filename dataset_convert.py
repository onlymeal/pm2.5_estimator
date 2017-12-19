import numpy as np
import pickle

with open('../00.data/v10_170713_5x5_dataset.pickle','rb') as handle:
    data  = pickle.load(handle)
with open('../00.data/v10_170713_5x5_label.pickle','rb') as handle:
    label = pickle.load(handle)

print(data.shape)
print(label.shape)

np.savez('sample_dataset/5x5_data', x=data[:100])
np.savez('sample_dataset/5x5_label', y=label[:100])
