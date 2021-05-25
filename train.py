import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
import wandb
import argparse

model = linear_model.LogisticRegression(multi_class="multinomial",max_iter= 5000)

def parse_arguments(parser):
    parser.add_argument("--lr", type=float, default=0.0006772587145077029)
    parser.add_argument("--epoch",type=int, default=7)
    parser.add_argument("--hidden_layer_dim",type=int, default=512)
    parser.add_argument("--iteration_conv",type=int, default=10)
    parser.add_argument("--steps_chain",type=int, default=100)
    parser.add_argument("--vis",type=int, default=784)
    parser.add_argument("--type1",type=int, default=1)
    return parser.parse_args()

def crossentropy(z, y):
	n = z.shape[0]
	#print("n",n)
	ce = 0
	for i, j in zip(z, y):
	  ce += -np.log(i[j])
	ce /= n
	return ce

def sigmoid(k):
  return 1/(1+np.exp(-k))

def forwardcompute(W,C,data,hidden):
  hid = np.zeros((data.shape[0], hidden))
  for i in range(len(data)):
    k = np.matmul(np.transpose(W), np.expand_dims(data[i], axis=-1)) + C
    # print("sk",np.matmul(np.transpose(W), data[i]).shape)
    # print("hid",k.shape)
    hid[i] = np.squeeze(sigmoid(k), axis=-1)
  return hid
def binary(data):
  for i in range(len(data)):
    for j in range(len(data[i])):
      if data[i,j] >= 127:
        data[i,j] = 1
      else:
        data[i,j] = 0
  return data



class RBM:
	def __init__(self,args):
		# print(args)
		self.visible_layer = args.vis
		self.hidden_layer = args.hidden_layer_dim
		self.output_layer = 10
		self.c = np.random.rand(self.hidden_layer,1)
		self.b = np.random.rand(self.visible_layer,1)
		self.weight = np.random.rand(self.visible_layer, self.hidden_layer)
		self.weigh_gibbs = np.random.rand(self.visible_layer, self.hidden_layer) #784*50
		self.c_gibbs = np.random.rand(self.hidden_layer,1) #50*1
		self.b_gibbs = np.random.rand(self.visible_layer,1) #784*1
		self.steps = args.steps_chain
		self.lr = args.lr
		self.epochs = args.epoch
		self.itersum = args.iteration_conv
		self.type1 = args.type1


	def load_data(self):
		mnist = pd.read_csv("/data/DLAssignments/dlpa4/archive/fashion-mnist_train.csv")
		mnist = mnist.sample(frac = 1)
		lab = mnist.label
		lab = lab.to_numpy()
		mnist.drop('label',axis='columns', inplace=True)
		mnist = mnist.to_numpy()
		n_rows, n_cols = mnist.shape
		mnist = binary(mnist)
		self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(mnist, lab, test_size = 0.2)
		test = pd.read_csv("/data/DLAssignments/dlpa4/archive/fashion-mnist_test.csv")
		lab_test = test.label
		self.lab_test = lab_test.to_numpy()
		test.drop('label',axis='columns', inplace=True)
		test = test.to_numpy()
		self.test = binary(test)
		# print(self.lab_test.shape)


	def train(self):
		self.load_data()
		if self.type1 == 1:
			return self.__contrastiveDivergence(self.X_val)
		else:
			return self.__gibbsSampling(self.X_val)

	def __sigmoid(self, k):
		return 1/(1+np.exp(-k))

	def __sampling(self, data):
		dim = data.shape[0]
		true_idx = np.random.uniform(0, 1, dim).reshape(dim, 1) <= data
		sampled = np.zeros((dim, 1))
		sampled[true_idx] = 1  # [n, 1]
		return sampled

	def __forward(self, data):
		k = np.matmul(np.transpose(self.weight), data) + self.c
		k = self.__sigmoid(k)
		return k

	def __backward(self, data):
		k = np.matmul(self.weight, data) + self.b
		k = self.__sigmoid(k)
		return k

	def __forwardGibbs(self, data):
		k = np.matmul(np.transpose(self.weigh_gibbs), data) + self.c_gibbs #50*784 784*1 = 50*1 + 50*1 = 50*1
		k = self.__sigmoid(k)
		return k #50*1

	def __backwardGibbs(self, data):
		k = np.matmul(self.weigh_gibbs, data) + self.b_gibbs #784*50 50*1 = 784*1 + 784*1 = 784*1
		k = self.__sigmoid(k)
		return k #784*1


	def __CD_compute(self,d):
		# self.v_p = self.v_p.reshape(-1,1)
		d = d.reshape(-1,1)
		for i in range(self.steps):
			h = self.__forward(self.v_p)
			h = self.__sampling(h)
			self.v_p = self.__backward(h)
			self.v_p = self.__sampling(self.v_p)
			h_p = self.__forward(self.v_p)
			h_p = self.__sampling(h_p)
    
    
		self.weight += self.lr*((np.matmul(d, h.T)) - (np.matmul(self.v_p, h_p.T)))
		self.c += self.lr*(h - h_p)
		self.b += self.lr*(d - self.v_p)
		
  
	def __contrastiveDivergence(self, data):
		self.v_p = data[0].reshape(-1,1)
		for i in range(self.epochs):
			wandb.log({"Epoch":(i+1)})

			for j in data:
				# print(i)
				self.__CD_compute(j)
			hid = forwardcompute(self.weight,self.c,self.X_val,self.hidden_layer)
			hid_test = forwardcompute(self.weight,self.c,self.test,self.hidden_layer)
			model.fit(hid,self.y_val)
			# wandb.sklearn.plot_learning_curve(model,hid,self.y_val)
			y_pred_train = model.predict(hid)
			y_pred = model.predict(hid_test)
			y_pred_probs = model.predict_proba(hid_test)
			wandb.log({"Validation Accuracy:":metrics.accuracy_score(self.y_val, y_pred_train)*100})
			wandb.log({"Testing Accuracy:":metrics.accuracy_score(self.lab_test, y_pred)*100})
			# print()
			wandb.log({"Cross Entropy Loss":metrics.log_loss(self.lab_test, y_pred_probs)})
		# tsne = sklearn.manifold.TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
		# tsne.fit_transform(hid_test)



	def __gibbsCompute(self, data):
		vd = data.reshape(-1,1) #784*1
		# v = vd #784*1
		c0=0
		c1=0
		c2=0
		for t in range(self.steps):
			h_t = self.__forwardGibbs(self.v) #50*1
			h = self.__sampling(h_t)
			# print("h",h.shape)
			v_t = self.__backwardGibbs(h) #784*1
			self.v = self.__sampling(v_t)
			# print("v",v.shape)
			h_tt = self.__forwardGibbs(self.v) #50*1
			h_tt = self.__sampling(h_tt)
		# print("h_tt",h_tt.shape)
			if t>= self.itersum:
				c0 += np.matmul(self.v,h_tt.T) #784*1 1*50 = 784*50
				c1 += self.v #784*1
				c2 += h_tt #50*1
		self.weigh_gibbs += self.lr*((np.matmul(vd, h.T)) - (1/(self.steps-self.itersum))*c0 ) #784*1 1*50 = 784*50 - 784*50
		self.b_gibbs += self.lr*(vd - (1/(self.steps-self.itersum))*c1 )
		self.c_gibbs += self.lr*(h - (1/(self.steps-self.itersum))*c2 )


	def __gibbsSampling(self, data):
		self.v = data[0].reshape(-1,1)
		for j in range(self.epochs):
			for i in data:
				self.__gibbsCompute(i)
			hid = forwardcompute(self.weigh_gibbs,self.c_gibbs,self.X_val,self.hidden_layer)
			hid_test = forwardcompute(self.weigh_gibbs,self.c_gibbs,self.test,self.hidden_layer)
			model.fit(hid,self.y_val)
			# wandb.sklearn.plot_learning_curve(model,hid,self.y_val)
			y_pred_train = model.predict(hid)
			y_pred = model.predict(hid_test)
			y_pred_probs = model.predict_proba(hid_test)
			wandb.log({"Validation Accuracy:":metrics.accuracy_score(self.y_val, y_pred_train)*100})
			wandb.log({"Testing Accuracy:":metrics.accuracy_score(self.lab_test, y_pred)*100})
			# print()
			wandb.log({"Cross Entropy Loss":metrics.log_loss(self.lab_test, y_pred_probs)})

		

  # def __train_logistic_reg(self,data):
  #   inp = self.__forward(data)




if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	args = parse_arguments(parser)
	# print(args)
	hyperparameter_defaults =dict(
	vis = args.vis,
	hidden_layer_dim = args.hidden_layer_dim,
	steps_chain = args.steps_chain,
	lr = args.lr,
	epoch = args.epoch,
	iteration_conv = args.iteration_conv,
	type1 = args.type1)
	wandb.init(project='dlpa4-manoj-shivangi', entity='shiv',config=hyperparameter_defaults)
	args = wandb.config

	

	rbm = RBM(args)
	rbm.train()

#reference: https://github.com/FengZiYjun/Restricted-Boltzmann-Machine/blob/master/rbm.py
