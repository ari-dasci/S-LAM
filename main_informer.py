import sys
sys.path.append("./utils")
sys.path.append("./models")

import torch
from transformer import TransformerModel
from progress_bar import SimpleProgressBar
import TSFEDL.models_pytorch as tsfedl
from tsfedl_top_module import TSFEDL_TopModule
from informer_datasets import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from metrics_informer import metric

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

EXPERIMENTATION_NAME = "informer_experimentation_attn_visualization"

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

def init_model(model, in_features, out_features, n_window=96, n_pred=1):
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
	if not model in ["transformer_local", "transformer_informer", "transformer_vanilla", "OhShuLih", "KhanZulfiqar", "ZhengZhenyu", "WangKejun",	
					 "KimTaeYoung", "FuJiangmeng", "ShiHaotian",
					 "LihOhShu", "HtetMyetLynn", "YaoQihang"]:
		raise Exception("Model not found")
	
	if model=="transformer_local":
		return TransformerModel(in_features, n_window=n_window, n_pred=n_pred,
						loss = torch.nn.MSELoss(), 
						num_encoder_layers = 2,
						num_decoder_layers = 2,
						optimizers = torch.optim.Adam,
						attn = LocalAttention,
						attn_params = {"output_attention":False},
						metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	elif model=="transformer_informer":
		return TransformerModel(in_features, n_window=n_window, n_pred=n_pred,
						loss = torch.nn.MSELoss(), 
						num_encoder_layers = 2,
						num_decoder_layers = 2,
						optimizers = torch.optim.Adam,
						attn = ProbAttention,
						attn_params = {"output_attention":False},
						metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3)
	elif model=="transformer_vanilla":
		return TransformerModel(in_features, n_window=n_window, n_pred=n_pred,
						loss = torch.nn.MSELoss(), 
						num_encoder_layers = 2,
						num_decoder_layers = 2,
						optimizers = torch.optim.Adam,
						attn = FullAttention,
						attn_params = {"output_attention":False},
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
	# BUG: pred mode not available yet
	parser.add_argument("--mode", choices=["train", "test", "pred"], default="train")
	parser.add_argument("--model", choices=["transformer_local", "transformer_informer", "transformer_vanilla", "OhShuLih", "KhanZulfiqar", "ZhengZhenyu", "WangKejun",
					 						"KimTaeYoung", "FuJiangmeng", "ShiHaotian",
					 						"LihOhShu", "HtetMyetLynn", "YaoQihang"], default="transformer_local")
	parser.add_argument("--n_epochs", type=int, default=100)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--dataset", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "WTH", "ECL"], default="ETTh1")
	parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
	parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
	parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
	parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
	parser.add_argument('--workers', type=int, default=0, help='number of cpu threads to use during batch generation')
	parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='device to use for training / testing')

	# BUG:There are problems with this. Pred len y seq len must be the same otherwise shapes do not match.
	parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
	parser.add_argument('--label_len', type=int, default=0, help='start token length of Informer decoder')
	parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

	nfeatures_dict = {"ETTh1": 7, "ETTh2": 7, "ETTm1": 7, "ETTm2": 7, "WTH": 12, "ECL": 321}
	target_dict = {"ETTh1": "OT", "ETTh2": "OT", "ETTm1": "OT", "ETTm2": "OT", "WTH": "WetBulbCelsius", "ECL": "MT_320"}
	
	args = parser.parse_args()
	print("Parameters chosen: ")
	for arg in vars(args):
		print(arg, ":", getattr(args, arg))

	valid_names_tsfedl = ["OhShuLih", "KhanZulfiqar", "ZhengZhenyu", "WangKejun", 
					"KimTaeYoung", "FuJiangmeng", "ShiHaotian", 
					"LihOhShu", "HtetMyetLynn", "YaoQihang"]
	valid_names = ["transformer"] + valid_names_tsfedl
	if args.model!="all":
		valid_names = [args.model]

	for_transformer = args.model in ["transformer_local", "transformer_informer", "transformer_vanilla"]
	for_tsfedl = args.model in valid_names_tsfedl
	for_test = args.mode=="test"
	
    # Load dataset
	data_dict = {
		'ETTh1': Dataset_ETT_hour,
		'ETTh2': Dataset_ETT_hour,
		'ETTm1': Dataset_ETT_minute,
		'ETTm2': Dataset_ETT_minute,
		'WTH': Dataset_Custom,
		'ECL': Dataset_Custom,
    }
	data_path = {
		'ETTh1': 'ETT',
		'ETTh2': 'ETT',
		'ETTm1': 'ETT',
		'ETTm2': 'ETT',
		'WTH': 'WTH',
		'ECL': 'ECL',
	}

	timeenc = 0 if args.embed == 'timeF' else 1
	if args.mode == "test":
		shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq = args.freq
	elif args.mode == "pred":
		shuffle_flag = False; drop_last = False; batch_size = 1; freq = args.freq
		Data = Dataset_Pred
	else:
		shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq = args.freq

	Data = data_dict[args.dataset]
	dataset = Data(
		root_path = './data/Informer_Datasets/'+data_path[args.dataset]+'/',
		data_path = args.dataset+'.csv',
		flag = args.mode,
		size = [args.seq_len, args.label_len, args.pred_len], # seq_len, label_len, pred_len
		features = args.features,
		target = target_dict[args.dataset],
		inverse = False,
		timeenc = timeenc,
		freq = freq,
		cols = args.cols,
		return_ts = for_test,
		for_transformer = for_transformer,
		for_tsfedl = for_tsfedl,
	)
	dataset_validation = Data(
		root_path = './data/Informer_Datasets/'+data_path[args.dataset]+'/',
		data_path = args.dataset+'.csv',
		flag = "val",
		size = [args.seq_len, args.label_len, args.pred_len], # seq_len, label_len, pred_len
		features = args.features,
		target = target_dict[args.dataset],
		inverse = False,
		timeenc = timeenc,
		freq = freq,
		cols = args.cols,
		return_ts = for_test,
		for_transformer = for_transformer,
		for_tsfedl = for_tsfedl,
	)

	dataloader = DataLoader(
		dataset,
		batch_size = batch_size,
		shuffle = shuffle_flag,
		num_workers = args.workers,
		drop_last = drop_last
	)
	dataloader_validation = DataLoader(
		dataset_validation,
		batch_size = batch_size,
		shuffle = shuffle_flag,
		num_workers = args.workers,
		drop_last = drop_last
	)

	print("N. instances: ", len(dataset.data_x))
	print("N. features: ", nfeatures_dict[args.dataset])

	# Initialize model
	model = None
	if args.mode=="train":
		model = init_model(args.model, nfeatures_dict[args.dataset], nfeatures_dict[args.dataset], args.seq_len, args.pred_len).double() 
	else:
		model = load_model(args.model, args.dataset, nfeatures_dict[args.dataset], nfeatures_dict[args.dataset], args.seq_len, args.pred_len, args.label_len).double()
		
	if args.mode == "train":
		#early_stopping = pl.callbacks.EarlyStopping('val_loss', min_delta=0.0001, 
		#												patience=5, verbose=True, mode='min')
		early_stopping = pl.callbacks.EarlyStopping('train_loss', min_delta=0.0001,
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
		#model_checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_loss', dirpath='./checkpoints/'+EXPERIMENTATION_NAME+'/'+args.dataset+'/'+args.model+'/', 
		#												filename=args.model+'-seq_len='+str(args.seq_len)+'-label_len='+str(args.label_len)+'-pred_len='+str(args.pred_len)+'-{epoch:02d}-{val_loss:.10f}', 
		#												save_top_k=1, mode='min')
		model_checkpoint = pl.callbacks.ModelCheckpoint(monitor='train_loss', dirpath='./checkpoints/'+EXPERIMENTATION_NAME+'/'+args.dataset+'/'+args.model+'/', 
														filename=args.model+'-seq_len='+str(args.seq_len)+'-label_len='+str(args.label_len)+'-pred_len='+str(args.pred_len)+'-{epoch:02d}-{train_loss:.10f}', 
														save_top_k=1, mode='min')
		
		# Create trainer
		print("Training: ", args.model)
		trainer = pl.Trainer(max_epochs=100, callbacks=[SimpleProgressBar(model_name=args.model), early_stopping, model_checkpoint])
		# Train
		now = time()
		#trainer.fit(model, dataloader, dataloader_validation)
		trainer.fit(model, dataloader)
		training_time = time()-now
		used_epochs = trainer.current_epoch+1
		times_df = pd.DataFrame({"training_time": [training_time], "used_epochs": [used_epochs], "time_per_epoch": [training_time/used_epochs]})
		if not os.path.exists('./results/'):
				os.makedirs('./results/')
		if not os.path.exists('./results/'+EXPERIMENTATION_NAME+'/'):
			os.makedirs('./results/'+EXPERIMENTATION_NAME+'/')
		if not os.path.exists('./results/'+EXPERIMENTATION_NAME+'/'+args.dataset+'/'):
			os.makedirs('./results/'+EXPERIMENTATION_NAME+'/'+args.dataset+'/')
		times_df.to_csv('./results/'+EXPERIMENTATION_NAME+'/'+args.dataset+'/'+args.model+'-seq_len='+str(args.seq_len)+'-label_len='+str(args.label_len)+'-pred_len='+str(args.pred_len)+'-times.csv')
	elif args.mode == "test":
		# With no gradients to save memory
		with torch.no_grad():
			# Test
			print("Testing: ", args.model)
			model.eval()
			model.to(args.device)
			preds, trues = [], []
				
			# Test
			for btch in tqdm(dataloader, desc="Testing"):
				# If for_transformer, src and tgt are the input and target sequences
				if for_transformer:
					src,tgt,y,_,_ = btch
					src = src.to(args.device)
					tgt = tgt.to(args.device)
					y = y.to(args.device)
				# If for_tsfedl, x and y are the input and target sequences
				else:
					x, y, _, _ = btch
					x = x.to(args.device)
					y = y.to(args.device)
				y_hat = model(x) if not for_transformer else model(src, tgt)
				preds.append(y_hat.detach().cpu().numpy())
				trues.append(y.detach().cpu().numpy())

			preds = np.array(preds)
			trues = np.array(trues)
			preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
			trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

			# Compute metrics
			mae, mse, rmse, mape, mspe = metric(preds, trues)
			print("Model: ", args.model, " - Dataset: ", args.dataset, " - Seq len: ", args.seq_len, " - Label len: ", args.label_len, " - Pred len: ", args.pred_len)
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
			result_df.to_csv('./results/'+EXPERIMENTATION_NAME+'/'+args.dataset+'/'+args.model+'-seq_len='+str(args.seq_len)+'-label_len='+str(args.label_len)+'-pred_len='+str(args.pred_len)+'.csv')