# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Perceptron en pytorch (en utilisant juste les tenseurs)
# Écrit par Mathieu Lefort
#
# Distribué sous licence BSD.
# ------------------------------------------------------------------------

import gzip, numpy, torch
from time import sleep
    
class parameters:
	def __init__(self, val,max,interval):
		self.val = val
		self.max = max
		self.interval = interval

	def iterate(self):
		val = self.val
		max = self.max
		interval = self.interval

		while val <= max:
			yield val
			val += interval	

class hyperparametersGenerator:
	def __init__(self, batch_size, nb_epochs, eta, w_min, w_max, nb_neurons):
		self.batch_size_generator = batch_size
		self.nb_epochs_generator = nb_epochs
		self.eta_generator = eta
		self.w_min_generator = w_min
		self.w_max_generator = w_max
		self.nb_neurons_generator = nb_neurons

	def iterate(self):
		for batch_size in self.batch_size_generator.iterate():
			for nb_epochs in self.nb_epochs_generator.iterate():
				for eta in self.eta_generator.iterate():
					for w_min in self.w_min_generator.iterate():
						for w_max in self.w_max_generator.iterate():
							for nb_neurons in self.nb_neurons_generator.iterate():
								yield hyperparameters(batch_size, nb_epochs, eta, w_min, w_max, nb_neurons)
							
class hyperparameters:
	def __init__(self, batch_size, nb_epochs, eta, w_min, w_max, nb_neurons):
		self.batch_size = batch_size
		self.nb_epochs = nb_epochs
		self.eta = eta
		self.w_min = w_min
		self.w_max = w_max
		self.nb_neurons = nb_neurons


generator = hyperparametersGenerator(
	parameters(5, 10, 1),
	parameters(10, 10, 1),
	parameters(0.00001, 1, 0.25),
	parameters(-0.001, 0, 0.25),
	parameters(0.001, 1, 0.25),
	parameters(1, 2048, 1000)
)

if __name__ == '__main__':
	# on lit les données
	((data_train,label_train),(data_test,label_test)) = torch.load(gzip.open('mnist.pkl.gz'))
	datasetTrain = torch.utils.data.TensorDataset(data_train,label_train)
	datasetTest = torch.utils.data.TensorDataset(data_test,label_test)

	for hyperparameter in generator.iterate():
		batch_size = 5 # nombre de données lues à chaque fois
		nb_epochs = 10 # nombre de fois que la base de données sera lue
		eta = 0.001 # taux d'apprentissage
		w_min = -0.001 # poids min
		w_max = 0.001 # poids max

		model = torch.nn.Sequential(
			torch.nn.Conv2d(1,6,5,padding=2),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d((2,2)),
			torch.nn.Conv2d(6,16,5),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d((2,2)),
			torch.nn.Flatten(),
			torch.nn.Linear(16*5*5,120),
			torch.nn.ReLU(),
			torch.nn.Linear(120,84),
			torch.nn.ReLU(),
			torch.nn.Linear(84,10)
		)

		loaderTrain = torch.utils.data.DataLoader(datasetTrain, batch_size=batch_size, shuffle=True)
		loaderTest = torch.utils.data.DataLoader(datasetTest, batch_size=1, shuffle=False)		

		torch.nn.init.uniform_(model[0].weight,w_min,w_max)
		torch.nn.init.uniform_(model[3].weight,w_min,w_max)
		torch.nn.init.uniform_(model[7].weight,w_min,w_max)
		torch.nn.init.uniform_(model[9].weight,w_min,w_max)
		torch.nn.init.uniform_(model[11].weight,w_min,w_max)

		loss_func = torch.nn.MSELoss(reduction='sum')
		optim = torch.optim.SGD(model.parameters(), lr=eta)

		for n in range(nb_epochs):
			for x,t in loaderTrain:
				y = model(x)
				loss = loss_func(t,y)
				loss.backward()
				optim.step()
				optim.zero_grad()
			acc = 0.
			for x,t in loaderTest:
				y = model(x)
				acc += torch.argmax(y,1) == torch.argmax(t,1)

			print((acc/data_test.shape[0]).item())
