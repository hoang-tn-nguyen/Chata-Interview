import torch
import torch.nn as nn

class CELoss(nn.Module):
	def __init__(self, ignore_index=-1):
		super().__init__()
		self.CELoss = nn.CrossEntropyLoss(ignore_index=ignore_index)

	def forward(self, output, target):
		'''
		Output: (N,*,C) \n
		Target: (N,*) \n
		'''
		output = torch.log(output)  # Invert softmax
		output = output.reshape(-1, output.shape[-1])  # (*,C)
		target = target.reshape(-1).long()  # (*)
		return self.CELoss(output, target)

class CELossShift(nn.Module):
	def __init__(self, ignore_index=-1):
		super().__init__()
		self.CELoss = CELoss(ignore_index=ignore_index)

	def forward(self, output, target):
		'''
		Output: (N,*,C) \n
		Target: (N,*) \n
		'''
		output = output[:,:-1,:] # (* - 1,C)
		target = target[:,1:] # (* - 1)
		return self.CELoss(output, target)