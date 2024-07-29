import sys
sys.path.append("./utils")
sys.path.append("./models")
sys.path.append("./models/informer")

import torch
from transformer import TransformerModel
from progress_bar import SimpleProgressBar
import TSFEDL.models_pytorch as tsfedl
from tsfedl_top_module import TSFEDL_TopModule
from uci_dataset import TimeSeriesDataset
from metrics_informer import metric
from model_informer import Informer

import pickle
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pyreadr

from torch.utils.data import Dataset, DataLoader, TensorDataset
from attn import FullAttention, ProbAttention, LocalAttention, AttentionLayer
import torch.nn as nn
from time import time
import pytorch_lightning as pl
import argparse

import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime

EXPERIMENTATION_NAME = "informer_experimentation_paper_uci"
ROUTE_TO_DATASETS = "/mnt/homeGPU/naguiler/Forecasting-Datasets/"

def load_model(name, dataset, in_features, out_features, n_window=96, n_pred=1, label_len=0):
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
	model = init_model(name, in_features, out_features, n_window, n_pred)
	if not os.path.exists('./checkpoints/'+EXPERIMENTATION_NAME+'/'+dataset+'/'+name+'/'):
		raise Exception("Model not found")
	checkpoints = os.listdir('./checkpoints/'+EXPERIMENTATION_NAME+'/'+dataset+'/'+name+'/')
	if len(checkpoints)==0:
		raise Exception("Model not found")
	desired_checkpoint = None
	for checkpoint in checkpoints:
		if '-seq_len='+str(n_window)+'-label_len='+str(label_len)+'-pred_len='+str(n_pred)+'-' in checkpoint:
			desired_checkpoint = checkpoint
			break
	if desired_checkpoint is None:
		raise Exception("Model not found")
	ckpt = torch.load('./checkpoints/'+EXPERIMENTATION_NAME+'/'+dataset+'/'+name+'/'+desired_checkpoint)
	model.load_state_dict(ckpt['state_dict'])
	return model

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
	return data, labels

def load_dataset(name):
	if not name in ["individual_household_electric_power", "new_york_taxi", "residential_power_battery", "tina"]:
		raise Exception("Dataset not found")
	df = None
	if name=="individual_household_electric_power":
		df = pd.read_csv(ROUTE_TO_DATASETS+"individual+household+electric+power+consumption/household_power_consumption.txt", sep=";", low_memory=False)
		df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
		df = df.drop(['Date', 'Time'], axis=1)
		df = df.set_index('datetime').sort_index()
		df = df.replace('?', np.nan)
		df = df.interpolate(method="linear")
		df = df.fillna(method="bfill")
		df = df.fillna(method="ffill")
		for column in df.columns:
			df[column] = df[column].astype("float64")
	elif name=="new_york_taxi":
		df = pd.read_parquet(ROUTE_TO_DATASETS+"New York City Taxi Dataset/yellow_tripdata_clean_grouped.parquet")
	elif name=="residential_power_battery":
		df = pyreadr.read_r(ROUTE_TO_DATASETS+"Residential Power and Battery Data/anonymous_public_load_power_data.rds")[None]
		df['utc'] = pd.to_datetime(df['utc'], format='%Y-%m-%d %H:%M:%S')
		df = df.set_index('utc').sort_index()
		df = df.groupby("utc").sum().drop(columns=["unit", "metric"])
	elif name=="tina":
		df, _ = load_tina_data()
		df = df.sort_index()
		df.index = pd.to_datetime(df.index, unit="s")
		df = df.resample("1Min").mean().bfill()
	return df
		

def init_model(model, in_features, out_features, n_window=96, n_pred=1, n_enc_layers=2, n_dec_layers=2, **kwargs):
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
	if not model in ["transformer_local", "transformer_informer", "transformer_vanilla", "informer_model", "OhShuLih", "KhanZulfiqar", "ZhengZhenyu", "WangKejun",	
					 "KimTaeYoung", "FuJiangmeng", "ShiHaotian",
					 "LihOhShu", "HtetMyetLynn", "YaoQihang"]:
		raise Exception("Model not found")
	
	if model=="transformer_local":
		return TransformerModel(in_features, n_window=n_window, n_pred=n_pred,
						loss = torch.nn.MSELoss(), 
						num_encoder_layers = n_enc_layers,
						num_decoder_layers = n_dec_layers,
						optimizers = torch.optim.Adam,
						attn = LocalAttention,
						attn_params = {"output_attention":False},
						metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	elif model=="transformer_informer":
		return TransformerModel(in_features, n_window=n_window, n_pred=n_pred,
						loss = torch.nn.MSELoss(), 
						num_encoder_layers = n_enc_layers,
						num_decoder_layers = n_dec_layers,
						optimizers = torch.optim.Adam,
						attn = ProbAttention,
						attn_params = {"output_attention":False},
						metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	elif model=="transformer_vanilla":
		return TransformerModel(in_features, n_window=n_window, n_pred=n_pred,
						loss = torch.nn.MSELoss(), 
						num_encoder_layers = n_enc_layers,
						num_decoder_layers = n_dec_layers,
						optimizers = torch.optim.Adam,
						attn = FullAttention,
						attn_params = {"output_attention":False},
						metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	elif model=="informer_model":
		return Informer(enc_in=in_features, dec_in=in_features, c_out=out_features, seq_len=n_window, label_len=0, out_len=n_pred,
						factor=5, d_model=512, n_heads=8, e_layers=2, d_layers=2, d_ff=512, 
						dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
						output_attention = False, distil=True, mix=True,
						device='cuda:0',
						loss = torch.nn.MSELoss(),
						optimizers = torch.optim.Adam,
						metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	elif model=="OhShuLih":
		return tsfedl.OhShuLih(in_features=in_features, 
						  loss = torch.nn.MSELoss(),
						  optimizer = torch.optim.Adam,
						  top_module = TSFEDL_TopModule(in_features=20, out_features=out_features, npred=n_pred),
						  metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	elif model=="KhanZulfiqar":
		return tsfedl.KhanZulfiqar(in_features=in_features, 
						  loss = torch.nn.MSELoss(),
						  optimizer = torch.optim.Adam,
						  top_module = TSFEDL_TopModule(in_features=10, out_features=out_features, npred=n_pred),
						  metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	elif model=="ZhengZhenyu":
		return tsfedl.ZhengZhenyu(in_features=in_features, 
						  loss = torch.nn.MSELoss(),
						  optimizer = torch.optim.Adam,
						  top_module = TSFEDL_TopModule(in_features=256, out_features=out_features, npred=n_pred),
						  metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	elif model=="WangKejun":
		return tsfedl.WangKejun(in_features=in_features, 
						  loss = torch.nn.MSELoss(),
						  optimizer = torch.optim.Adam,
						  top_module = TSFEDL_TopModule(in_features=256, out_features=out_features, npred=n_pred),
						  metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	elif model=="KimTaeYoung":
		return tsfedl.KimTaeYoung(in_features=in_features, 
						  loss = torch.nn.MSELoss(),
						  optimizer = torch.optim.Adam,
						  top_module = TSFEDL_TopModule(in_features=64, out_features=out_features, npred=n_pred),
						  metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	elif model=="FuJiangmeng":
		return tsfedl.FuJiangmeng(in_features=in_features, 
						  loss = torch.nn.MSELoss(),
						  optimizer = torch.optim.Adam,
						  top_module = TSFEDL_TopModule(in_features=256, out_features=out_features, npred=n_pred),
						  metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	elif model=="ShiHaotian":
		return tsfedl.ShiHaotian(in_features=in_features, 
						  loss = torch.nn.MSELoss(),
						  optimizer = torch.optim.Adam,
						  top_module = TSFEDL_TopModule(in_features=32, out_features=out_features, npred=n_pred),
						  metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	elif model=="LihOhShu":
		return tsfedl.LihOhShu(in_features=in_features, 
						  loss = torch.nn.MSELoss(),
						  optimizer = torch.optim.Adam,
						  top_module = TSFEDL_TopModule(in_features=10, out_features=out_features, npred=n_pred),
						  metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	elif model=="HtetMyetLynn":
		return tsfedl.HtetMyetLynn(in_features=in_features, 
						  loss = torch.nn.MSELoss(),
						  optimizer = torch.optim.Adam,
						  top_module = TSFEDL_TopModule(in_features=80, out_features=out_features, npred=n_pred),
						  metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	elif model=="YaoQihang":
		return tsfedl.YaoQihang(in_features=in_features, 
						  loss = torch.nn.MSELoss(),
						  optimizer = torch.optim.Adam,
						  top_module = TSFEDL_TopModule(in_features=32, out_features=out_features, npred=n_pred),
						  metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	return None

if __name__ == '__main__':

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", choices=["train", "test"], default="train")
	parser.add_argument("--model", choices=["transformer_local", "transformer_informer", "transformer_vanilla", "informer_model", 
											"OhShuLih", "KhanZulfiqar", "ZhengZhenyu", "WangKejun",
					 						"KimTaeYoung", "FuJiangmeng", "ShiHaotian",
					 						"LihOhShu", "HtetMyetLynn", "YaoQihang"], default="transformer_local")
	parser.add_argument("--n_epochs", type=int, default=100)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--dataset", choices=["individual_household_electric_power", "new_york_taxi", "residential_power_battery", "tina"], default="individual_household_electric_power")
	parser.add_argument('--workers', type=int, default=0, help='number of cpu threads to use during batch generation')
	parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='device to use for training / testing')
	parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
	parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
	parser.add_argument('--stride', type=int, default=1, help='stride of the input sequence')
	parser.add_argument('--train_percent', type=float, default=0.7, help='percentage of data to use for training')
	parser.add_argument('--val_percent', type=float, default=0.15, help='percentage of data to use for validation')
	parser.add_argument('--enc_layers', type=int, default=2, help='number of encoder layers')
	parser.add_argument('--dec_layers', type=int, default=2, help='number of decoder layers')
	
	args = parser.parse_args()
	if args.mode == "test":
		print("Changing stride to prediction length as we are in test mode.")
		args.stride=args.pred_len
	print("Parameters chosen: ")
	for arg in vars(args):
		print(arg, ":", getattr(args, arg))
	for_transformer = args.model in ["transformer_local", "transformer_informer", "transformer_vanilla"]
	for_informer = args.model=="informer_model"

	# Load dataset
	df = load_dataset(args.dataset)
	print("Dataset loaded")
	print("Dataset shape: ", df.shape)
	print("Head of dataset: ", df.head())
	# Split dataset into train and test
	train_df_total = df.iloc[:int(len(df)*args.train_percent)]
	train_df = train_df_total.iloc[:int(len(train_df_total)*(1-args.val_percent))]
	val_df = train_df_total.iloc[int(len(train_df_total)*(1-args.val_percent)):]
	test_df = df.iloc[int(len(df)*args.train_percent):]
	train_df_timestamps = train_df.index
	val_df_timestamps = val_df.index
	test_df_timestamps = test_df.index
	# Parse timestamps to unix
	# train_df_timestamps = (train_df_timestamps - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
	# val_df_timestamps = (val_df_timestamps - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
	# test_df_timestamps = (test_df_timestamps - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
	# Normalize dataset
	sc = StandardScaler()
	train_df = sc.fit_transform(train_df)
	val_df = sc.transform(val_df)
	test_df = sc.transform(test_df)
	permutation = (0,1) if for_transformer or for_informer else (1,0)
	# Create dataset
	train_dataset = TimeSeriesDataset(torch.from_numpy(train_df), args.seq_len, args.pred_len, timestamps = train_df_timestamps, stride=args.stride, permute=permutation, for_transformer=for_transformer, for_informer=for_informer)
	val_dataset = TimeSeriesDataset(torch.from_numpy(val_df), args.seq_len, args.pred_len, timestamps = val_df_timestamps, stride=args.stride, permute=permutation, for_transformer=for_transformer, for_informer=for_informer)
	test_dataset = TimeSeriesDataset(torch.from_numpy(test_df), args.seq_len, args.pred_len, timestamps = test_df_timestamps, stride=args.stride, permute=permutation, for_transformer=for_transformer, for_informer=for_informer)
	# Create dataloaders
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
	# Initialize model
	model = init_model(args.model, train_df.shape[1], train_df.shape[1], n_window=args.seq_len, n_pred=args.pred_len, n_enc_layers=args.enc_layers, n_dec_layers=args.dec_layers).double()
	if args.mode=="train":
		early_stopping = pl.callbacks.EarlyStopping('val_loss', min_delta=0.0001,
														patience=5, verbose=True, mode='min')
		# Create checkpoint if it does not exist
		if not os.path.exists('./checkpoints/'):
			os.makedirs('./checkpoints/')
		if not os.path.exists('./checkpoints/'+EXPERIMENTATION_NAME+'/'):
			os.makedirs('./checkpoints/'+EXPERIMENTATION_NAME+'/')
		# Create checkpoint for model if it does not exist
		if not os.path.exists('./checkpoints/'+EXPERIMENTATION_NAME+'/'+args.dataset+'/'):
			os.makedirs('./checkpoints/'+EXPERIMENTATION_NAME+'/'+args.dataset+'/')
		if not os.path.exists('./checkpoints/'+EXPERIMENTATION_NAME+'/'+args.dataset+'/'+args.model+'/'):
			os.makedirs('./checkpoints/'+EXPERIMENTATION_NAME+'/'+args.dataset+'/'+args.model+'/')
		# Define checkpoint callback, save only one checkpoint (the best one)
		if args.model in ["transformer_local", "transformer_informer", "transformer_vanilla"]:
			f_name = args.model+'-seq_len='+str(args.seq_len)+'-pred_len='+str(args.pred_len)+'-enc_layers='+str(args.enc_layers)+'-dec_layers='+str(args.dec_layers)+'-{epoch:02d}-{train_loss:.10f}'
		else:
			f_name = args.model+'-seq_len='+str(args.seq_len)+'-pred_len='+str(args.pred_len)+'-{epoch:02d}-{train_loss:.10f}'
		model_checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_loss', dirpath='./checkpoints/'+EXPERIMENTATION_NAME+'/'+args.dataset+'/'+args.model+'/', 
														filename=f_name, 
														save_top_k=1, mode='min')
		
		# Create trainer
		print("Training: ", args.model)
		trainer = pl.Trainer(max_epochs=args.n_epochs, callbacks=[SimpleProgressBar(model_name=args.model), early_stopping, model_checkpoint])
		# Train
		now = time()
		trainer.fit(model, train_loader, val_loader)
		training_time = time()-now
		used_epochs = trainer.current_epoch+1
		times_df = pd.DataFrame({"training_time": [training_time], "used_epochs": [used_epochs], "time_per_epoch": [training_time/used_epochs]})
		if not os.path.exists('./results/'):
				os.makedirs('./results/')
		if not os.path.exists('./results/'+EXPERIMENTATION_NAME+'/'):
			os.makedirs('./results/'+EXPERIMENTATION_NAME+'/')
		if not os.path.exists('./results/'+EXPERIMENTATION_NAME+'/'+args.dataset+'/'):
			os.makedirs('./results/'+EXPERIMENTATION_NAME+'/'+args.dataset+'/')
		if args.model in ["transformer_local", "transformer_informer", "transformer_vanilla"]:
			times_df.to_csv('./results/'+EXPERIMENTATION_NAME+'/'+args.dataset+'/'+args.model+'-seq_len='+str(args.seq_len)+'-pred_len='+str(args.pred_len)+'-enc_layers='+str(args.enc_layers)+'-dec_layers='+str(args.dec_layers)+'-times.csv')
		else:
			times_df.to_csv('./results/'+EXPERIMENTATION_NAME+'/'+args.dataset+'/'+args.model+'-seq_len='+str(args.seq_len)+'-pred_len='+str(args.pred_len)+'-times.csv')
	else:
		# With no gradients to save memory
		with torch.no_grad():
			# Test
			print("Testing: ", args.model)
			model.eval()
			model.to(args.device)
			preds, trues = np.array([]), np.array([])
				
			# Test
			for btch in tqdm(test_loader, desc="Testing"):
				y_hat=None
				# If for_transformer, src and tgt are the input and target sequences
				if for_transformer:
					src,tgt,y = btch
					src = src.to(args.device)
					tgt = tgt.to(args.device)
					y = y.to(args.device)
					y_hat = model(src, tgt)
				elif for_informer:
					x, x_ts, y, y_ts = btch
					x = x.to(args.device)
					y = y.to(args.device)
					x_ts = x_ts.to(args.device)
					y_ts = y_ts.to(args.device)
					y_hat = model.test_step((x, x_ts, y, y_ts), 0)
				# If for_tsfedl, x and y are the input and target sequences
				else:
					x, y = btch
					x = x.to(args.device)
					y = y.to(args.device)
					y_hat = model(x)
				preds = np.append(preds,y_hat.detach().cpu().numpy(), axis=0) if len(preds)>0 else y_hat.detach().cpu().numpy()
				trues = np.append(trues,y.detach().cpu().numpy(), axis=0) if len(trues)>0 else y.detach().cpu().numpy()
				
			preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
			trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

			# Compute metrics
			mae, mse, rmse, mape, mspe = metric(preds, trues)
			print("Model: ", args.model, " - Dataset: ", args.dataset, " - Seq len: ", args.seq_len, " - Pred len: ", args.pred_len)
			print("MAE: ", mae)
			print("MSE: ", mse)
			print("RMSE: ", rmse)
			print("MAPE: ", mape)
			print("MSPE: ", mspe)

			# Save results	
			result_df = pd.DataFrame({"mae": [mae], "mse": [mse], "rmse": [rmse], "mape": [mape], "mspe": [mspe]})
			if not os.path.exists('./results/'):
				os.makedirs('./results/')
			if not os.path.exists('./results/'+EXPERIMENTATION_NAME+'/'):
				os.makedirs('./results/'+EXPERIMENTATION_NAME+'/')
			if not os.path.exists('./results/'+EXPERIMENTATION_NAME+'/'+args.dataset+'/'):
				os.makedirs('./results/'+EXPERIMENTATION_NAME+'/'+args.dataset+'/')
			if args.model in ["transformer_local", "transformer_informer", "transformer_vanilla"]:
				result_df.to_csv('./results/'+EXPERIMENTATION_NAME+'/'+args.dataset+'/'+args.model+'-seq_len='+str(args.seq_len)+'-pred_len='+str(args.pred_len)+'-enc_layers='+str(args.enc_layers)+'-dec_layers='+str(args.dec_layers)+'.csv')
			else:
				result_df.to_csv('./results/'+EXPERIMENTATION_NAME+'/'+args.dataset+'/'+args.model+'-seq_len='+str(args.seq_len)+'-pred_len='+str(args.pred_len)+'.csv')