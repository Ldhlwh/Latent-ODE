import numpy as np
import torch

def eq(a, b):
	return np.fabs(a - b) < 1e-8

def make_batch_mask(batch, param):
	all_time = np.sort(np.unique(batch[:, :, 0]).flatten())
	sync_batch = np.zeros((batch.shape[0], all_time.shape[0], batch.shape[2]))
	mask = np.zeros(sync_batch.shape[0:2])
	sync_batch[:, :, 0] = all_time
	for i in range(sync_batch.shape[0]):
		cur = 0
		for j in range(sync_batch.shape[1]):
			if cur == batch.shape[1]:
				continue
			if eq(sync_batch[i, j, 0], batch[i, cur, 0]):
				sync_batch[i, j, 1] = batch[i, cur, 1]
				mask[i, j] = 1
				cur += 1
	return torch.tensor(sync_batch, dtype = torch.float32, device = param['device']), torch.tensor(mask, dtype = torch.float32, device = param['device'])