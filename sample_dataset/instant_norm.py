import numpy as np

data_path = '5x5_sample_dataset.npz'
map_range = 5

data  = np.load(data_path)['x']
data = data.reshape(data.shape[0], map_range*map_range, 74)

print("Before instance normalization")
print(data.reshape(data.shape[0], map_range, map_range, 74)[0, :, :, 2])

mask = np.ma.array(data, mask= np.isnan(data))
mean = np.mean(mask, 1)

coor = np.argwhere(np.isnan(data))

for i in range(coor.shape[0]):
	data[coor[i,0], coor[i,1], coor[i,2]] = mean[coor[i, 0], coor[i, 2]]

"""
cnt = 0
for i in range(data.shape[0]) :
	for j in range(data.shape[1]) :
		for k in range(data.shape[2]) :
			if np.isnan(data[i, j, k]) :
				if cnt % 10000 == 0 :
					print("\r %d / %d"%(cnt, data.shape[0] * data.shape[1] * data.shape[2])),
				data[i, j, k] = mean[i, k]
				cnt += 1
"""
coor = np.argwhere(np.isnan(data))
if coor.shape[0] != 0 :
	print("Error! wrong calculation", coor.shape)
else :
	print("dataset saving...")
	np.savez(data_path.split('.')[0]+"_instance_norm", x=data.reshape(data.shape[0], data.shape[1]*data.shape[2]))

data_path = data_path.split('.')[0]+"_instance_norm.npz"
data = np.load(data_path)['x']

print("After instance normalization")
print(data.reshape(data.shape[0], map_range, map_range, 74)[0, :, :, 2])