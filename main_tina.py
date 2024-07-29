import sys
sys.path.append("./utils")
sys.path.append("./models")

import torch
from transformer import TransformerModel
from timeseries_dataset import TimeSeriesDataset
from progress_bar import SimpleProgressBar
import TSFEDL.models_pytorch as tsfedl
from tsfedl_top_module import TSFEDL_TopModule

import pickle
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, TensorDataset
from attn import FullAttention, ProbAttention, LocalAttention, AttentionLayer
import torch.nn as nn
from time import time
import pytorch_lightning as pl
import argparse

import sklearn.metrics as metrics
import matplotlib.pyplot as plt

def generate_sintetic_data():
	"""
	Function to generate sintetic data
	"""
	data = np.random.uniform(0, 1, (10000, 13))
	return data, np.array([])

def load_tina_data():
	"""
	Function to load TiNA data
	"""
	route = "./data/tina/"
	days = os.listdir(route)
	data = []
	for day in tqdm(days, desc="Loading data"):
		df = pd.read_parquet(route+day+"/data.parquet")
		data.append(df)
	data = pd.concat(data)
	labels = data[["m_id", "m_subid", "alarms"]]
	data = data.drop(["m_id", "m_subid", "alarms"], axis=1)
	# Remove categorical features
	data = data.drop(["FEATURE76", "FEATURE87"], axis=1)
	return data.values.astype(np.float64), labels

def processTiNA(data, labels, w_size=360):
	"""
	Function to process TiNA data. It returns normal data and anomalies data

	Parameters
	----------
	data: numpy array
		Data to process
	labels: pandas dataframe
		Labels of the data
	w_size: int
		Window size to process the data

	Returns
	-------
	normal_data: numpy array
		Normal data
	normal_labels: pandas dataframe
		Normal labels
	anomalies_data: numpy array
		Anomalies data
	anomalies_labels: pandas dataframe
		Anomalies labels
	"""
	# Normal data has "none" in m_id, m_subid and alarms
	# Anomalies has another value in at least one of them
	# data is a numpy array and labels is a pandas dataframe with timestamp as index

	# Get anomalies
	anomalies_events = np.where((labels["m_id"]!="none") | (labels["m_subid"]!="none") | (labels["alarms"]!="none"))[0]
	anomalies_events_and_previous = np.zeros(len(labels))
	cont=0
	for a in anomalies_events:
		if cont%100==0:
			print("Processing TiNA data: ", cont, "/", len(anomalies_events), " - Perc: ", cont/len(anomalies_events)*100, end="\r")
		cont+=1
		if a-w_size*3>=0:
			anomalies_events_and_previous[a-w_size:a] = 1
		else:
			anomalies_events_and_previous[:a] = 1
	print()

	anomalies_indexes = np.where(anomalies_events_and_previous==1)[0]
	anomalies_labels = labels.iloc[anomalies_indexes]
	# Get normal data
	normal_indexes = np.where(anomalies_events_and_previous==0)[0]
	normal_labels = labels.iloc[normal_indexes]
	# Get normal data from data
	normal_data = data[normal_indexes]
	# Get anomalies from data
	anomalies_data = data[anomalies_indexes]

	normal_labels = labels.iloc[normal_indexes]
	anomalies_labels = labels.iloc[anomalies_indexes]
	return normal_data, normal_labels, anomalies_data, anomalies_labels


def load_data(dataset, train_percentage=0.7, window_size=360):
	"""
	Function to load data. It returns train data and test data depending on the train_percentage

	Parameters
	----------
	dataset: str
		Dataset to load
	train_percentage: float
		Percentage of data to use as train data
	window_size: int
		Window size to process the data for removing anomalies and previous data
	"""
	if dataset=="synthetic":
		return generate_sintetic_data()
	elif dataset=="tina":
		print("Loading TiNA data")
		data, labels = load_tina_data()
		data_train, labels_train = data[:int(train_percentage*len(data))], labels[:int(train_percentage*len(labels))]
		data_test, labels_test = data[int(train_percentage*len(data)):], labels[int(train_percentage*len(labels)):]
		del data, labels
		print("Processing TiNA data")
		data_train, labels_train, _, _ = processTiNA(data_train, labels_train, window_size)
		return data_train, labels_train, data_test, labels_test
	else:
		raise Exception("Dataset not found")

def load_model(name):
	"""
	Function to load a model from the checkpoints folder

	Parameters
	----------
	name: str
		Name of the model to load

	Returns
	-------
	model: pytorch model
		Model loaded
	"""
	model = init_model(name)
	if not os.path.exists('./checkpoints/tina/'+name+'/'):
		raise Exception("Model not found")
	checkpoints = os.listdir('./checkpoints/tina/'+name+'/')
	if len(checkpoints)==0:
		raise Exception("Model not found")
	ckpt = torch.load('./checkpoints/tina/'+name+'/'+checkpoints[0])
	model.load_state_dict(ckpt['state_dict'])
	return model

def init_model(model):
	"""
	Function to initialize a model.

	Parameters
	----------
	model: str
		Name of the model to initialize
	
	Returns
	-------
	model: pytorch model
		Model initialized
	"""
	if not model in ["transformer", "OhShuLih", "KhanZulfiqar", "ZhengZhenyu", "WangKejun",	
					 "KimTaeYoung", "FuJiangmeng", "ShiHaotian",
					 "LihOhShu", "HtetMyetLynn", "YaoQihang"]:
		raise Exception("Model not found")
	
	if model=="transformer":
		return TransformerModel(103, n_window=360, 
						loss = torch.nn.MSELoss(), 
						num_encoder_layers = 2,
						num_decoder_layers = 2,
						optimizers = torch.optim.Adam,
						attn = LocalAttention,
						attn_params = {},
						metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	elif model=="OhShuLih":
		return tsfedl.OhShuLih(in_features=103, 
						  loss = torch.nn.MSELoss(),
						  optimizer = torch.optim.Adam,
						  top_module = TSFEDL_TopModule(in_features=20, out_features=103, npred=1),
						  metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	elif model=="KhanZulfiqar":
		return tsfedl.KhanZulfiqar(in_features=103, 
						  loss = torch.nn.MSELoss(),
						  optimizer = torch.optim.Adam,
						  top_module = TSFEDL_TopModule(in_features=10, out_features=103, npred=1),
						  metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	elif model=="ZhengZhenyu":
		return tsfedl.ZhengZhenyu(in_features=103, 
						  loss = torch.nn.MSELoss(),
						  optimizer = torch.optim.Adam,
						  top_module = TSFEDL_TopModule(in_features=256, out_features=103, npred=1),
						  metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	elif model=="WangKejun":
		return tsfedl.WangKejun(in_features=103, 
						  loss = torch.nn.MSELoss(),
						  optimizer = torch.optim.Adam,
						  top_module = TSFEDL_TopModule(in_features=256, out_features=103, npred=1),
						  metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	elif model=="KimTaeYoung":
		return tsfedl.KimTaeYoung(in_features=103, 
						  loss = torch.nn.MSELoss(),
						  optimizer = torch.optim.Adam,
						  top_module = TSFEDL_TopModule(in_features=64, out_features=103, npred=1),
						  metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	elif model=="FuJiangmeng":
		return tsfedl.FuJiangmeng(in_features=103, 
						  loss = torch.nn.MSELoss(),
						  optimizer = torch.optim.Adam,
						  top_module = TSFEDL_TopModule(in_features=256, out_features=103, npred=1),
						  metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	elif model=="ShiHaotian":
		return tsfedl.ShiHaotian(in_features=103, 
						  loss = torch.nn.MSELoss(),
						  optimizer = torch.optim.Adam,
						  top_module = TSFEDL_TopModule(in_features=32, out_features=103, npred=1),
						  metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	elif model=="LihOhShu":
		return tsfedl.LihOhShu(in_features=103, 
						  loss = torch.nn.MSELoss(),
						  optimizer = torch.optim.Adam,
						  top_module = TSFEDL_TopModule(in_features=10, out_features=103, npred=1),
						  metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	elif model=="HtetMyetLynn":
		return tsfedl.HtetMyetLynn(in_features=103, 
						  loss = torch.nn.MSELoss(),
						  optimizer = torch.optim.Adam,
						  top_module = TSFEDL_TopModule(in_features=80, out_features=103, npred=1),
						  metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	elif model=="YaoQihang":
		return tsfedl.YaoQihang(in_features=103, 
						  loss = torch.nn.MSELoss(),
						  optimizer = torch.optim.Adam,
						  top_module = TSFEDL_TopModule(in_features=32, out_features=103, npred=1),
						  metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	return None

if __name__ == '__main__':

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", choices=["train", "test", "labels", "metrics"], default="train")
	parser.add_argument("--model", choices=["transformer", "OhShuLih", "KhanZulfiqar", "ZhengZhenyu", "WangKejun",
					 						"KimTaeYoung", "FuJiangmeng", "ShiHaotian",
					 						"LihOhShu", "HtetMyetLynn", "YaoQihang", "all"], default="transformer")
	args = parser.parse_args()
	print("Parameters chosen: ")
	print("Mode: ", args.mode)
	print("Model: ", args.model)

	valid_names_tsfedl = ["OhShuLih", "KhanZulfiqar", "ZhengZhenyu", "WangKejun", 
					"KimTaeYoung", "FuJiangmeng", "ShiHaotian", 
					"LihOhShu", "HtetMyetLynn", "YaoQihang"]
	valid_names = ["transformer"] + valid_names_tsfedl
	if args.model!="all":
		valid_names = [args.model]
	mode = args.mode

	window_size = 360

	# Load dataset
	data_train, labels_train, data_test, labels_test = None, None, None, None
	if mode=="train":
		data_train, labels_train, _, _ = load_data("tina", train_percentage=0.7, window_size=window_size)
	elif mode=="test":
		_, _, data_test, labels_test = load_data("tina", train_percentage=0.7, window_size=window_size)
	elif mode=="metrics":
		_, _, _, labels_test = load_data("tina", train_percentage=0.7, window_size=window_size)
	
	for name in valid_names:
		model = None
		if mode=="train" or mode=="test":
			# Init model or load model depending on the mode
			model = init_model(name).double() if mode=="train" else load_model(name).double()
			print(model)

		# Set parameters depending on the model
		for_forecasting = True if name in valid_names_tsfedl else False
		for_transformer = True if name=="transformer" else False
		for_tsfedl = True if name in valid_names_tsfedl else False

		if mode=="train":
			# Split train and validation
			validation_percentage = 0.2
			data_val, labels_val = data_train[:int(validation_percentage*len(data_train))], labels_train[:int(validation_percentage*len(labels_train))]
			data_train, labels_train = data_train[int(validation_percentage*len(data_train)):], labels_train[int(validation_percentage*len(labels_train)):]
			print("Data shape:", data_train.shape)
			print("Labels shape:", labels_train.shape)
			# Create dataset
			timestamps_train = pd.to_datetime(labels_train.index, unit="s").values
			timestamps_val = pd.to_datetime(labels_val.index, unit="s").values

			# Create dataset for train and validation
			dataset_train = TimeSeriesDataset(torch.from_numpy(data_train), timestamps_train, 
												sequence_length=window_size, npred=1, for_forecasting=for_forecasting, 
												for_transformer=for_transformer, for_tsfedl=for_tsfedl)
			dataset_val = TimeSeriesDataset(torch.from_numpy(data_val), timestamps_val, 
											sequence_length=window_size, npred=1, for_forecasting=for_forecasting, 
											for_transformer=for_transformer, for_tsfedl=for_tsfedl)
			# Create dataloader
			#bs 95 for ProbAttention and LocalAttention 8 splits
			#bs 16 for FullAttention
			#bs 150 for LocalAttention 18 splits
			dataloader_train = DataLoader(dataset_train, batch_size=150, shuffle=True, num_workers=64, pin_memory=True)
			dataloader_val = DataLoader(dataset_val, batch_size=150, shuffle=False, num_workers=64, pin_memory=True)
			# Create callbacks
			early_stopping = pl.callbacks.EarlyStopping('val_loss', min_delta=0.0001, 
														patience=5, verbose=True, mode='min')
			
			# Create checkpoint if it does not exist
			if not os.path.exists('./checkpoints/'):
				os.makedirs('./checkpoints/')
			if not os.path.exists('./checkpoints/'+"tina"+'/'):
				os.makedirs('./checkpoints/'+"tina"+'/')
			# Create checkpoint for model if it does not exist
			if not os.path.exists('./checkpoints/tina/'+name+'/'):
				os.makedirs('./checkpoints/tina/'+name+'/')
			# Define checkpoint callback, save only one checkpoint (the best one)
			model_checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_loss', dirpath='./checkpoints/tina/'+name+'/', 
															filename=name+'-{epoch:02d}-{val_loss:.10f}', 
															save_top_k=1, mode='min')
															
			# Create trainer
			print("Training: ", name)
			trainer = pl.Trainer(max_epochs=100, callbacks=[SimpleProgressBar(model_name=name), early_stopping, model_checkpoint])
			# Train
			trainer.fit(model, dataloader_train, dataloader_val)
		elif mode=="test":
			# With no gradients to save memory
			with torch.no_grad():
				# Test
				print("Testing: ", name)
				model.eval()

				# Create dataset for test
				timestamps_test = pd.to_datetime(labels_test.index, unit="s").values
				errors = np.zeros(len(timestamps_test))
				dataset_test = TimeSeriesDataset(torch.from_numpy(data_test), timestamps_test, 
												sequence_length=window_size, npred=1, for_forecasting=for_forecasting, 
												for_transformer=for_transformer, for_tsfedl=for_tsfedl, for_test=True)
				dataloader_test = DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=16, pin_memory=True)
				
				# Test
				for btch in tqdm(dataloader_test, desc="Testing"):
					# If for_transformer, src and tgt are the input and target sequences
					if for_transformer:
						src,tgt,y,_,_, idx = btch
					# If for_tsfedl, x and y are the input and target sequences
					else:
						x, y, _, _, idx = btch
					target_index = idx[1]
					y_hat = model(x) if not for_transformer else model(src, tgt)
					if for_forecasting:
						# Designed to predict 1 step
						if len(y_hat.shape)>2:
							y_hat = y_hat[:,-1,:]
						y = y.reshape((y.shape[0],y.shape[2]))
						error = np.linalg.norm((y_hat-y), axis=1)
					else:
						error = np.linalg.norm((y_hat-y), axis=2)
					# Add error to the corresponding indexes
					for cont,(b,e) in enumerate(zip(target_index[0], target_index[1])):
						errors[b:e] += error[cont]

				# Save results	
				result_df = pd.DataFrame(columns=["unix", "error"])
				result_df.unix = labels_test.index
				result_df.error = errors
				result_df = result_df.set_index("unix").sort_index()
				if not os.path.exists('./results/'):
					os.makedirs('./results/')
				result_df.to_csv('./results/'+name+'.csv')
		elif mode=="labels":
			print("Obtaining labels: ", name)
			results = pd.read_csv('./results/'+name+'.csv')
			results = results.set_index("unix").sort_index()
			# Obtain labels by sliding window of 360 points
			# Compute the mean and deviation of the previous 360 points
			# If the error is greater than 3 times the deviation, it is an anomaly
			# If the error is greater than 2 times the deviation, it is a warning

			# Compute mean and deviation
			mean = np.zeros(len(results))
			deviation = np.zeros(len(results))
			for i in range(window_size, len(results)):
				if i%10000==0:
					print("Computing mean and deviation for model ", name, " at point ", i, "/", len(results), " - Perc: ", i/len(results)*100, end="\r")
				mean[i] = np.mean(results.error[i-window_size:i])
				deviation[i] = np.std(results.error[i-window_size:i])
			# Place as mean and deviation of the first 360 points the mean and deviation of the whole dataset
			mean[:window_size] = np.mean(results.error)
			deviation[:window_size] = np.std(results.error)
			print()
			print("Mean and deviation computed for model ", name)

			# Obtain labels
			labels = np.zeros(len(results))
			for i in range(window_size, len(results)):
				if i%10000==0:
					print("Obtaining labels for model ", name, " at point ", i, "/", len(results), " - Perc: ", i/len(results)*100, end="\r")
				if results.iloc[i].error > mean[i] + 3*deviation[i]:
					labels[i] = 2
				elif results.iloc[i].error > mean[i] + 2*deviation[i]:
					labels[i] = 1
			
			print()
			print("Labels obtained for model ", name)

			# Save labels
			labels_df = pd.DataFrame(columns=["unix", "label", "mean", "deviation"])
			labels_df.unix = results.index
			labels_df.label = labels
			labels_df.mean = mean
			labels_df.deviation = deviation
			labels_df = labels_df.set_index("unix").sort_index()
			if not os.path.exists('./labels/'):
				os.makedirs('./labels/')
			labels_df.to_csv('./labels/'+name+'.csv')
			print("Labels saved for model ", name)
		elif mode=="metrics":
			print("Computing metrics for model ", name)
			scores = pd.read_csv('./results/'+name+'.csv')
			scores = scores.set_index("unix").sort_index()
			labels = pd.read_csv('./labels/'+name+'.csv')
			labels = labels.set_index("unix").sort_index()
			labels.label = labels.label.astype("int")
			# There are labels with values 0,1 and 2
			# Parse all 2 values to 1
			labels.label = labels.label.replace(2, 1)

			# There is an anomaly to detect if labels_test.m_id is not "none"
			# We need to identify the maintenance periods, this is, the periods where labels_test.m_id is not "none" consecutively
			# If our model detects an anomaly in a maintenance period or window_size before, it is a true positive

			# Obtain maintenance periods
			maintenance_identifiers = (labels_test.m_id.values!="none").astype(int)
			maintenance_events = np.where(maintenance_identifiers==1)[0]
			target_labels = np.zeros(len(labels))
			cont=0
			for m in maintenance_events:
				if cont%100==0:
					print("Computing metrics for model ", name, " at point ", cont, "/", len(maintenance_events), " - Perc: ", cont/len(maintenance_events)*100, end="\r")
				cont+=1
				if m-window_size*2>=0:
					target_labels[m-window_size*2:m] = 1
				else:
					target_labels[:m] = 1

			print()

			# Obtain blocks of maintenance periods
			prev_value = target_labels[0]
			block_ind1 = 0
			block_ind2 = 0
			# If there is more than 10% of anomalies in the maintenance period, it is a true positive
			for i in range(1, len(target_labels)):
				if i%10000==0:
					print("Analyzing maintenances with model ", name, " at point ", i, "/", len(target_labels), " - Perc: ", i/len(target_labels)*100, end="\r")
				if target_labels[i]==prev_value:
					block_ind2 = i
				else:
					if prev_value==1:
						anomalies = np.sum(labels.label.values[block_ind1:block_ind2])/(block_ind2-block_ind1)
						# If there is more than 10% of anomalies in the maintenance period, it is a true positive
						if anomalies>0.1:
							labels.label.values[block_ind1:block_ind2] = 1
					block_ind1 = i
					block_ind2 = i
					prev_value = target_labels[i]
			
			# Compute acc, precision, recall, f1, auc
			acc = metrics.accuracy_score(target_labels, labels.label.values.astype("int"))
			precision = metrics.precision_score(target_labels, labels.label.values.astype("int"), average="macro")
			recall = metrics.recall_score(target_labels, labels.label.values.astype("int"), average="macro")
			f1 = metrics.f1_score(target_labels, labels.label.values.astype("int"), average="macro")
			auc = metrics.roc_auc_score(target_labels, scores.error.values)

			# Plot ROC curve in a detailed plot
			fpr, tpr, _ = metrics.roc_curve(target_labels, scores.error.values)
			plt.figure()
			plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % auc)
			plt.plot([0, 1], [0, 1], 'k--')
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.05])
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.title('ROC curve ' + name)
			plt.legend(loc="lower right")
			if not os.path.exists('./plots/'):
				os.makedirs('./plots/')
			plt.savefig('./plots/'+name+'_roc.png')

			# Save metrics
			metrics_df = pd.DataFrame(columns=["acc", "precision", "recall", "f1", "auc"])
			metrics_df.loc[0] = [acc, precision, recall, f1, auc]
			if not os.path.exists('./metrics/'):
				os.makedirs('./metrics/')
			metrics_df.to_csv('./metrics/'+name+'.csv')

		else:
			raise Exception("Mode not found")