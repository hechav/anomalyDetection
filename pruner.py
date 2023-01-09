#!/usr/bin/env python3

import re
import fileinput
import sys
import pandas as pd
import os
import numpy as np


import time
from datetime import datetime, timedelta
import json

# ml
import h2o
from h2o.estimators.kmeans import H2OKMeansEstimator
from h2o.estimators import H2OExtendedIsolationForestEstimator, H2OIsolationForestEstimator, H2ORandomForestEstimator

sys.path.insert(0, '/home/sistemas/tj_analytics/mlclient_profiler')

# pruner
from classes.anomaly_pruner import AnomalyPruner


fecha_hoy = datetime.today().strftime("%Y%m%d")

try:
	h2o.shutdown()
except:
	print("No h2o instance running")
# inicializo


h2o.init(
	# bind_to_localhost = False,
	# ip = "192.168.11.12",
	nthreads=-1,
	# de 10g a 15g no cambió mucho
	min_mem_size_GB=10,
	enable_assertions = False )

# missing: reading all host files

files = os.listdir('./temp_chunks')

n_urls = len(files)
print(f'# de hosts:{n_urls}')

feats_grouping = ['clientip','countrycode','content_type','agent','proto','response', 'method']

# listas
ips_tj_risk_max = []
ips_tj_risk_medium = []

# otros datos relevantes
duration = []
combinations = []
sizes = []
profiler_data = []

i = 0

for file in files:
	data_size = [] 
	file_clean = file.replace(".csv","").replace("data_","")
	print(f'\nLoading chunk from: {file_clean}')
	# if os.path.getsize(f'./temp_chunks/{file}')<9*1024**3:
	# 	print('chunk is too big!')
	# else:
	data = pd.read_csv(f'./temp_chunks/{file}')
	data_size.append(data.shape[0])
	data.drop(data[data.clientip=='127.0.0.1'].index, inplace=True)
	data_size.append(data.shape[0])
	sizes.append(data_size)
	print(f'size: {data.shape}')
	n_comb = data.groupby(feats_grouping[1:]).count().shape[0]
	if n_comb<3 or data.shape[0]<10:
		pass
	else:
		# -------------------------------------------------------------------------------------------------
		# -------------------------------------------------------------------------------------------------
		# training y predict
		# -------------------------------------------------------------------------------------------------
		print('loaded')
		pruner = AnomalyPruner(file_clean)
		starttime = time.time()
		pruner.pruner_trn(df=data,n_trees=1000,n_depth=8)
		duration_temp = []
		duration_temp.append(timedelta(seconds=time.time() - starttime).seconds)
		starttime = time.time()
		# ips en percentil .1 de más riesgo
		pruned_data_risk_max = pruner.pruner_predict(percentile=.001)
		ips_file_risk_max = pruner.anomalous_ips
		ips_tj_risk_max.extend( ips_file_risk_max )
		# ips en percentil 1 de más riesgo
		pruned_data_risk_medium = pruner.pruner_predict()
		ips_file_risk_medium = pruner.anomalous_ips
		ips_tj_risk_medium.extend( ips_file_risk_medium )
		duration_temp.append(timedelta(seconds=time.time() - starttime).seconds)
		# ips_tj.to_csv(f'./greylist/TRANSPARENT_ML_LEVEL_0.ABUSE.1D.1D.99.IP.csv', index=False)
		# -------------------------------------------------------------------------------------------------
		# -------------------------------------------------------------------------------------------------
		# info para profiler
		# -------------------------------------------------------------------------------------------------
		print(data.shape)
		data["anomaly"] = pruner.predictions.anomaly==1	
		n_anomalies = data.anomaly.sum()
		n_anomalous_ips = data.loc[data.anomaly==1].groupby('clientip').count().shape[0]
		n_obs = data.shape[0]
		n_ips = data.groupby('clientip').count().shape[0]
		perc_anomalies = 100*n_anomalies/data.shape[0]
		n_countries    = data.groupby('countrycode').count().shape[0]
		n_contents     = data.content_type.str.upper().unique().shape[0]
		n_agents       = data.groupby('agent').count().shape[0]
		n_responses    = data.response.astype('float').unique().shape[0]
		perc_get       = (data.method=='GET').mean()
		perc_video     = data.content_type.str.lower().str.contains('video').mean()
		perc_app       = data.content_type.str.lower().str.contains('application').mean()
		perc_image     = data.content_type.str.lower().str.contains('image').mean()
		perc_text     = data.content_type.str.lower().str.contains('text').mean()
		anomaly_fingerprint = [file_clean, perc_anomalies, n_comb, n_obs, n_ips, n_obs/n_ips, perc_get, perc_video, perc_app, perc_image, perc_text  ]
		profiler_data.append(anomaly_fingerprint)
		# -------------------------------------------------------------------------------------------------
		# -------------------------------------------------------------------------------------------------
		# info sobre anomalías
		# -------------------------------------------------------------------------------------------------	
		# reviso anomalías
		print(f'# de anomalías: {n_anomalies:,}')
		print(f'% de anomalías: {perc_anomalies:,}')
		# guardo toda la data no anómala
		# data.to_csv(f'{path_main}/src/data_{url_clean}{fecha_datos.replace("-","")}.csv', index=False)
		duration.append(duration_temp)
		# obtengo distribución de scores
		distr = pruner.predictions_hc.group_by('mean_length').count().get_frame()
		h2o.export_file(distr,f'./varimp/distr_{file_clean}.csv', force=True)
		print(distr.head(50))
		# obtengo ips anómalas
		ips = pruner.url_data\
		.loc[ pruner.predictions['anomaly']==1 ][feats_grouping]
		scores = pruner.predictions[ pruner.predictions['anomaly']==1 ]['mean_length']
		anomalous_ips = pd.concat([ scores, ips], axis=1 )\
		.groupby(feats_grouping[:6]).agg( {"mean_length":['count','min']} )\
		.dropna()
		anomalous_ips.columns = anomalous_ips.columns.get_level_values(1)
		anomalous_ips.rename(columns={"count":"anomalous_appearances","min":"score_min",}, inplace=True)
		print(f'# de ips anómalas: {anomalous_ips.shape[0]}')
		anomalous_ips_json = anomalous_ips\
		.sort_values(by='score_min', ascending=False)\
		.reset_index()\
		.to_json(orient='index')
		print("Guardo en json")
		with open(f'./greylist/greylist_{file_clean}.json', 'w') as f:
			json.dump(anomalous_ips_json, f)
		print("Obtengo relevancia de variables")
		tst = pruner.predictions.iloc[:,:-2]
		tst_hc = h2o.H2OFrame(tst)
		feature_importance = h2o.as_list( pruner.ml_engine_model.feature_frequencies(pruner.trn_hc), use_pandas=True )\
		.mean()\
		.sort_values(ascending=False)
		# guardo
		feature_importance.to_csv(f'./varimp/varimp_{file_clean}.csv')
		# obtengo fingerprint
		fingerprint = 100*pruner.predictions.groupby('anomaly').mean(numeric_only=True)
		fingerprint.to_csv(f'./varimp/fingerprint_{file_clean}.csv')
		# obtengo más relaciones con la distribución
		# heatmap = 100*pruner.predictions.groupby('quantiles').sum(numeric_only=True)/pruner.predictions.sum()
		# heatmap.to_csv(f'./varimp/heatmap_{file_clean}.csv')
		del data
		print(i)
		i = i+1


# highest risk
ips_df_risk_max = pd.DataFrame(ips_tj_risk_max, columns=['']) 
ips_df_risk_max.to_csv(f'./greylist/TRANSPARENT_ML.ABUSE.1D.1D.99.IP.csv', index=False)
# medium risk
# excluyo los que ya estaban en el top .1
ips_df_risk_medium = pd.DataFrame(ips_tj_risk_medium, columns=['']) 
ips_df_risk_medium.loc[~ips_df_risk_medium.ip.isin(ips_tj_risk_max)]\
.to_csv(f'./greylist/TRANSPARENT_ML.ABUSE.1D.1D.80.IP.csv', index=False)

print(ips_df_risk_medium.loc[ips_df_risk_medium.ip.isin(~ips_tj_risk_max)].shape, ips_df_risk_medium.shape)

duration_df = pd.DataFrame(duration, columns=['trn_seconds','predict_seconds'])
profiler_data_df = pd.DataFrame(profiler_data, columns=['host', 'perc_anomalies', 'n_combinations', 
	'n_obs', 'n_ips', 'avg_logs', 'perc_get', 'perc_video', 'perc_app', 'perc_image', 'perc_text' ])
# sizes_df = pd.DataFrame(sizes, colums=['all','without_ips'])


duration_df.to_csv(f'./varimp/duration.csv', index=False)
# combinations_df.to_csv(f'./varimp/n_combinations.csv', index=False)
profiler_data_df.to_csv(f'./varimp/profiler_data.csv', index=False)
# .rename(columns={"proto":"https"})\

h2o.shutdown()






# de peticiones
# métodos: put/delete - transacciones vs get

# content

# MLCLIENT_ANOMALIES.ABUSE.1D.1D.99.IP.csv
