#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import subprocess
import time
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch

# gráficos
import matplotlib.pyplot as plt
# %matplotlib inline
from matplotlib.patches import *
import matplotlib.ticker as ticker
import seaborn as sns
plt.style.use('seaborn-whitegrid')
from matplotlib.dates import DateFormatter
from IPython.display import Image

# ml
import h2o
from h2o.estimators.kmeans import H2OKMeansEstimator
from h2o.estimators import H2OExtendedIsolationForestEstimator, H2OIsolationForestEstimator, H2ORandomForestEstimator


class AnomalyPruner:
	"""Detecta anomalías en el tráfico de un sitio."""
	def __init__(self, url ):
		assert isinstance(url, str), f"url expected string, got {url} instead"
		self.url = url
	def get_traffic(self,es_connection, day_to_search):
		assert isinstance(day_to_search, str), \
		f" day_to_search expected string, got {day_to_search} instead"
		self.es_connection = es_connection
		self.day_of_data = day_to_search
		"""Obtiene muestreo semi-aleatorio de tráfico."""
		lapses = [str(s)[:16] for s in pd.date_range(start=day_to_search, freq='20min', inclusive='both',periods=n_samples).values]
		n_obs = 10000
		logs = []
		for ll in lapses:
			res = self.es_connection.search(index="delivery-*", 
				size=n_obs, 
				q=f'tcdn.varnish.vhost:{self.url} and @timestamp:"{ll}"')
			logs.extend(res['hits']['hits'])
		self.url_data_json = logs
		df = pd.DataFrame([logs[ xx ]['_source']['tcdn']['varnish'] for  xx  in range(0,len(logs))])\
		.drop_duplicates()
		self.url_data = df
	def pruner_trn(self, df=None, max_cats=4, n_trees=300, n_depth=10, feats_cat= ['proto','agent','countrycode','content_type','response']):
		"""Ejecuta isolation forest sobre el tráfico obtenido. """
		assert isinstance(max_cats, int) and isinstance(n_trees, int) and isinstance(n_depth, int), \
		f" max_cats, n_trees, n_depth expected integer, got {max_cats,n_trees,n_depth} instead"
		assert isinstance(df,pd.DataFrame), 'data is not a pandas dataframe'
		if df is None:
			df = self.url_data.copy()
		else:
			self.url_data = df
		id_feat = "method"
		list_dummies = []
		list_cats = []
		# divido en trn-tst-val
		n_obs = df.shape[0]
		df['random_ix'] = np.random.uniform(0,1,n_obs)
		df['trn_tst_val'] = np.where( df.random_ix>.7, 'tst', 
			np.where( df.random_ix<.45, 'trn','val' )  )
		# --------------------------------------------------------------------------------------------------------------
		# --------------------------------------------------------------------------------------------------------------
		# FEATURES
		# --------------------------------------------------------------------------------------------------------------
		# agent
		top_cats = [str(xx).upper() for xx in df\
					.loc[ df.trn_tst_val!='tst' ]\
					.groupby('agent')[id_feat].count()\
					.sort_values( ascending=False)\
					.iloc[:20]\
					.index.unique()]
		df_temp = pd.get_dummies( np.where(df[ff].str.upper().isin(top_cats), df[ff], 0), prefix='agent' )
		# almaceno dummies
		list_dummies.append(df_temp)
		temp_cats = df_temp.columns.tolist()
		# guardo
		list_cats.extend(temp_cats)
		# resto de variables
		for ff in ['proto','countrycode','content_type','response']:
			# me quedo con cats comunes (considero solo trn)
			top_cats = [str(xx).upper() for xx in df\
						.loc[ df.trn_tst_val!='tst' ]\
						.groupby(ff)[id_feat].count()\
						.sort_values( ascending=False)\
						.iloc[:max_cats]\
						.index.unique()]
			if ff=='response':
				df_temp = pd.get_dummies( np.where(df[ff].astype("float").isin([float(xx) for xx in top_cats]), 
												   df[ff], 0), 
										 prefix=ff )
			elif ff=='proto':
				df_temp = (df.proto=="https").to_frame()*1
			else:
				df_temp = pd.get_dummies( np.where(df[ff].str.upper().isin(top_cats), df[ff], 0), 
										 prefix=ff )
			# almaceno dummies
			list_dummies.append(df_temp)
			temp_cats = df_temp.columns.tolist()
			# guardo
			list_cats.extend(temp_cats)
		# concateno dummies en un solo df
		X_set = pd.concat(list_dummies,axis=1)
		# features que a fuerza deben estar
		X_set['response_403'] = df.response.astype('float')==403
		X_set['agent_presents_as_bot'] = df.agent.str.lower().str.contains('bot')
		# X_set['response_403']
		X_set["trn_tst_val"] = df.trn_tst_val
		X_set['clientip'] = df.clientip
		self.ml_engine_features = list_cats
		# filtro por tipo de set y me quedo con feats relevantes
		trn_temp = X_set\
		.loc[ X_set.trn_tst_val!='tst' ]\
		[list_cats]
		tst_temp = X_set\
		.loc[ X_set.trn_tst_val=='tst' ]\
		[list_cats]
		# entreno
		trn_hc = h2o.H2OFrame( trn_temp )
		tst_hc = h2o.H2OFrame( tst_temp )
		for column in list_cats:
			trn_hc[column] = trn_hc[column].asfactor()
			tst_hc[column] = tst_hc[column].asfactor()
		self.trntst_df = X_set[list_cats+['clientip']]
		self.trn_hc = trn_hc
		self.tst_hc = tst_hc
		# algunos parámetros para almacenado
		# clean_date = self.day_of_data.replace("-","")
		clean_url  = self.url.replace(".","")
		model_name = f'{clean_url}_isolationForest'
		path = f'models/{clean_url}'
		isolation_model = H2OIsolationForestEstimator(model_id=model_name,
													  max_depth=n_depth,
													  ntrees = n_trees,
													  mtries = -1,
													  contamination=-1,
													  sample_rate=.7
													  # score_each_iteration=True
													 )
		isolation_model.train( training_frame=trn_hc, x=list_cats
							  # validation_frame=tst_hc 
							 )
		self.ml_engine_model = isolation_model
		modelfile = isolation_model.download_mojo(path=path, get_genmodel_jar=True)
		self.model_path = modelfile
		print("Model (zip and jar) saved to "+modelfile)
	def pruner_predict(self, df=None, model_path=None, percentile=.05):
		if df is None:
			df = self.trntst_df
		if model_path is None:
			model_path = self.model_path
		isolation_model = h2o.import_mojo(model_path)
		# estimo
		print("\nPredicting about to start...")
		tst_hc = h2o.H2OFrame( df )
		for column in df.columns:
			tst_hc[column] = tst_hc[column].asfactor()
		predictions_hc = isolation_model.predict(tst_hc)
		self.predictions_hc = predictions_hc
		# defino anomalías
		print("\nGetting anomaly threshold...")
		mid = predictions_hc['mean_length'].min()+(predictions_hc['mean_length'].max()-predictions_hc['mean_length'].min())/2
		n_anomalies = int(np.ceil(df.shape[0]*percentile))
		# dist = predictions_hc.group_by('mean_length').count().get_frame()
		# step_max = int((dist[1:,0]-dist[:-1,0]).idxmax()[0,0])
		# anomaly_thresh    = predictions_hc['mean_length'].quantile()[1,1]
		# if (dist[1:,0]-dist[:-1,0]).max()>.25:
		# 	anomaly_thresh = dist[step_max,0]
		# else:
		# 	anomaly_thresh = dist[int(dist.shape[0]/4),0]
		anomaly_thresh = predictions_hc['mean_length'].sort(0)[n_anomalies,0]
		print(f"Threshold Score at:  {anomaly_thresh} mean paths")
		tst_hc["anomaly"] = (predictions_hc["mean_length"] <= anomaly_thresh).ifelse(1,0)
		# almaceno anomalías
		print("Saving...")
		anomalies_df = h2o.as_list(tst_hc["anomaly"],use_pandas=True)
		tst_temp = df.copy()
		# también quiero añadir a los datos iniciales
		tst_temp['anomaly'] = anomalies_df
		tst_temp["predict"] = h2o.as_list( predictions_hc["predict"], use_pandas=True )
		tst_temp["mean_length"] = h2o.as_list( predictions_hc["mean_length"], use_pandas=True )
		tst_temp['quantiles'] =  pd.qcut(tst_temp['mean_length'],q=4, duplicates='drop')
		self.predictions = tst_temp
		self.predictions_pruned = tst_temp.loc[tst_temp.anomaly==1]
		# distribución
		self.score_distribution = predictions_hc.group_by('mean_length').count().get_frame()
		# y también quiero almacenar ips anómalas
		self.anomalous_ips = tst_temp.loc[tst_temp.anomaly==1]["clientip"].drop_duplicates().values.flatten().tolist()
		return self.predictions_pruned
