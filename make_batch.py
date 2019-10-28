import numpy as np
import torch

def make_batch(batch):
	all_time = np.sort(batch[:, :, 0].flatten())
	sync_batch = np.zeros((batch.shape[0], all_time.shape[0], batch.shape[2]))
	sync_batch[:, :, 0] = all_time
	for i in range(sync_batch.shape[0]):
		cur = -1
		for j in range(sync_batch.shape[1]):
			if cur == batch.shape[1] - 1:
				sync_batch[i, j, 1] = sync_batch[i, j - 1, 1]
				continue
			if sync_batch[i, j, 0] >= batch[i, cur + 1, 0]:
				cur += 1
			sync_batch[i, j, 1] = batch[i, max(0, cur), 1]
	return sync_batch