
import pandas as pd
import tensorflow as tf
import requests
import json
import time
import argparse
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
import os

# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output
# import flask

# import plotly.graph_objs as go
# from numba import jit
# from datetime import date

GCS_BUCKET = 'gs://projectsentient' #CHANGE THIS TO YOUR BUCKET
PROJECT = 'project-sentient-000001' #CHANGE THIS TO YOUR PROJECT ID
REGION = 'us-east1' #OPTIONALLY CHANGE THIS

API_KEY = '7C55TUKVWYOC48V6'
SYMBOL = 'MSFT'
LEN = 5245
SYMBOL = SYMBOL.rstrip()

os.environ['GCS_BUCKET'] = GCS_BUCKET
os.environ['PROJECT'] = PROJECT
os.environ['REGION'] = REGION
DIR='/content/datalab/trainer'


base_url = 'https://www.alphavantage.co/query?'
function_query_string = 'function='
symbol_query_string = 'symbol='
market_query_string = 'market='
api_key_query_string = 'apikey='
datatype_query_string = 'datatype='
interval_query_string = 'interval='
time_period_query_string = 'time_period='
series_type_query_string = 'series_type='

def build_request_string(**request_params):
	request_url = base_url
	for param, value in request_params.items():
		request_url+=(param+'='+str(value)+'&')
	request_url+=('apikey='+API_KEY)
	return request_url

# print(requests.get(build_request_string(function='EMA', symbol='MSFT', interval='daily', time_period='6', series_type='open')).text)

def get_techindicator_data(**request_params):
	ti_dict = {request_params['function']:[]}
	request_url = build_request_string(**request_params)
	request_data = json.loads((requests.get(request_url)).text)
	ti_data = request_data['Technical Analysis: '+request_params['function']]
	for date in ti_data:
		ti_dict[request_params['function']].append(ti_data[date][request_params['function']])
	df_ti = pd.DataFrame(ti_dict)
	return df_ti.tail(LEN).reset_index(drop=True)

def get_time_series_data(**request_params):
	ts_dict = {}
	request_params_for_time_series = {}
	request_params_for_time_series['function'] = request_params['function']
	request_params_for_time_series['symbol'] = request_params['symbol']
	request_params_for_time_series['outputsize'] = 'full'
	request_url = build_request_string(**request_params_for_time_series)
	request_data = json.loads((requests.get(request_url)).text)
	ts_data = request_data['Time Series (Daily)']
	for date in ts_data:
		for key in ts_data[date]:
			if key not in ts_dict:
				ts_dict[key] = []			
			ts_dict[key].append(ts_data[date][key])	
	df_ts = pd.DataFrame(ts_dict)
	return df_ts.tail(LEN).reset_index(drop=True)		



def get_final_df(*frames):
	return pd.concat(frames, axis=1, ignore_index=True)


# request_data = requests.get(build_request_string(function='EMA', symbol='MSFT', interval='daily', time_period='6', series_type='open')).text

mean_based_tech_indicators = ['EMA', 'SMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'T3', 'RSI', 'WILLR', 'ADX', 'ADXR', 'MOM']

def create_data_for_train_test(*indicators, **request_params):
	frames = []
	for indicator in indicators:
		if indicator=='TRIMA' or indicator=='ADX':
			time.sleep(90)
		request_params['function'] = indicator
		print(request_params)
		frame = get_techindicator_data(**request_params)
		frames.append(frame)
	request_params['function'] = 'TIME_SERIES_DAILY_ADJUSTED'	
	frames.append(get_time_series_data(**request_params))	
	return get_final_df(*frames)
	

# dataframe = create_data_for_train_test('EMA', 'SMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'T3', 'RSI', 'WILLR', 'ADX', 'ADXR', 'MOM', function='EMA', symbol='MSFT', interval='daily', time_period='6', series_type='open')	

# dataframe.head(4196).to_csv('fintech_data_train.csv', index=False)
# dataframe.tail(1049).to_csv('fintech_data_test.csv', index=False)

# print(get_time_series_data(function='TIME_SERIES_DAILY_ADJUSTED', symbol='MSFT'))

# print(dict_data['Technical Analysis: EMA'])
# foster scientific innovation to inspire sustainable global change


# print(get_techindicator_data(function='MACD', symbol='MSFT', interval='daily', series_type='open'))











# class StockData():

	# __ti = TechIndicators(key=API_KEY, output_format='pandas')
	# __ts = TimeSeries(key=API_KEY, output_format='pandas')

	# def __init__(self, technical_indicator_type, symbol, interval, time_period = None, series_type = None ):
# 	def __init__(self, **attributes):
# 		for param, value in attributes.items():
# 			if param == 'api_key':
# 				self.api_key = param
# 			elif param == 'output_format':
# 			    self.output_format = param    	
# 		self.technical_indicator_type = technical_indicator_type
# 		self.symbol = symbol
# 		self.interval = interval
# 		self.time_period = time_period
# 		self.series_type = series_type
# 		self.data = self.__get_techindicator_data()[0]
# 		self.meta_data = self.__get_techindicator_data()[1]

# 	def __get_techindicator_data(self):
# 		if self.technical_indicator_type == 'sma':
# 			return StockData.__ti.get_sma(symbol = self.symbol, interval = self.interval, time_period = self.time_period)
# 		elif self.technical_indicator_type == 'ema':
# 			return StockData.__ti.get_ema(symbol = self.symbol, interval = self.interval, time_period = self.time_period)
# 		elif self.technical_indicator_type == 'bbands':
# 			return StockData.__ti.get_bbands(symbol = self.symbol, interval = self.interval, time_period = self.time_period)
# 		elif self.technical_indicator_type == 'macd':
# 			return StockData.__ti.get_macd(symbol = self.symbol, interval = self.interval, series_type = self.series_type)


# stockdata = StockData()

# ts_data, ts_metadata = ts.get_daily(symbol=SYMBOL.rstrip(), outputsize='full')
# ts_data = ts_data.tail(LEN)
# df_sma, meta_data_sma = ti.get_sma(symbol=SYMBOL.rstrip(), interval='daily', time_period=6)
# df_sma = df_sma.tail(LEN)
# df_ema, meta_data_ema = ti.get_ema(symbol=SYMBOL.rstrip(), interval='daily', time_period=6)
# df_wma, meta_data_wma = ti.get_wma(symbol=SYMBOL.rstrip(), interval='daily', time_period=6)
# df_dema, meta_data_dema = ti.get_dema(symbol=SYMBOL.rstrip(), interval='daily', time_period=6)
# # 1998-01-16
# df_tema, meta_data_tema = ti.get_tema(symbol=SYMBOL.rstrip(), interval='daily', time_period=6)
# # 1998-01-26
# df_trima, meta_data_trima = ti.get_trima(symbol=SYMBOL.rstrip(), interval='daily', time_period=6)
# df_kama, meta_data_kama = ti.get_kama(symbol=SYMBOL.rstrip(), interval='daily', time_period=6)
# df_mama, meta_data_mama = ti.get_mama(symbol=SYMBOL.rstrip(), interval='daily')
# # 1998-02-19
# df_t3, meta_data_t3 = ti.get_t3(symbol=SYMBOL.rstrip(), interval='daily', time_period=6)
# df_macd, meta_data_macd = ti.get_macd(symbol=SYMBOL.rstrip(), interval='daily')
# # 1998-02-20
# df_macdext, meta_data_macdext = ti.get_macdext(symbol=SYMBOL.rstrip(), interval='daily')
# df_stoch, meta_data_stoch = ti.get_stoch(symbol=SYMBOL.rstrip(), interval='daily')
# df_stochf, meta_data_stochf = ti.get_stochf(symbol=SYMBOL.rstrip(), interval='daily')
# df_rsi, meta_data_rsi = ti.get_rsi(symbol=SYMBOL.rstrip(), interval='daily', time_period=6)
# df_stochrsi, meta_data_stochrsi = ti.get_stochrsi(symbol=SYMBOL.rstrip(), interval='daily', time_period=6)
# df_willr, meta_data_willr = ti.get_willr(symbol=SYMBOL.rstrip(), interval='daily', time_period=6)
# df_adx, meta_data_adx = ti.get_adx(symbol=SYMBOL.rstrip(), interval='daily', time_period=6)
# df_adxr, meta_data_adxr = ti.get_adxr(symbol=SYMBOL.rstrip(), interval='daily', time_period=6)
# df_apo, meta_data_apo = ti.get_apo(symbol=SYMBOL.rstrip(), interval='daily')
# df_ppo, meta_data_ppo = ti.get_ppo(symbol=SYMBOL.rstrip(), interval='daily')
# df_mom, meta_data_mom = ti.get_mom(symbol=SYMBOL.rstrip(), interval='daily', time_period=6)


# df_dataset = pd.concat([df_sma.tail(LEN), df_ema.tail(LEN), df_wma.tail(LEN), df_dema.tail(LEN), df_tema.tail(LEN), df_trima.tail(LEN), df_kama.tail(LEN), df_mama.tail(LEN), df_t3.tail(LEN), df_macd.tail(LEN), df_macdext.tail(LEN), df_stoch.tail(LEN), df_stochf.tail(LEN), df_rsi.tail(LEN), df_stochrsi.tail(LEN), df_willr.tail(LEN), df_adx.tail(LEN), df_adxr.tail(LEN), df_apo.tail(LEN), df_ppo.tail(LEN), df_mom.tail(LEN)], axis=1, ignore_index=True)
# print(df_dataset)






path_train = DIR + '/fintech_data_train.csv'
path_test = DIR + '/fintech_data_test.csv'

data_train = pd.read_csv(
  filepath_or_buffer= 'https://storage.googleapis.com/projectsentient/fintech_data_train.csv',
  names=['EMA', 'SMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'T3', 'RSI', 'WILLR', 'ADX', 'ADXR', 'MOM',"OPEN", "HIGH", "LOW", "CLOSE", "ADJUSTED_CLOSE", "VOLUME", "DIVIDEND_AMOUNT", "SPLIT_COEFFICIENT"])

data_test = pd.read_csv(
  filepath_or_buffer= 'https://storage.googleapis.com/projectsentient/fintech_data_test.csv',
  names=['EMA', 'SMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'T3', 'RSI', 'WILLR', 'ADX', 'ADXR', 'MOM',"OPEN", "HIGH", "LOW", "CLOSE", "ADJUSTED_CLOSE", "VOLUME", "DIVIDEND_AMOUNT", "SPLIT_COEFFICIENT"])



FEATURES = ['EMA', 'SMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'T3', 'RSI', 'WILLR', 'ADX', 'ADXR', 'MOM', "OPEN", "HIGH", "LOW", "ADJUSTED_CLOSE", "VOLUME", "DIVIDEND_AMOUNT", "SPLIT_COEFFICIENT"]
# LABELS = ["OPEN", "HIGH", "LOW", "CLOSE", "ADJUSTED_CLOSE", "VOLUME", "DIVIDEND_AMOUNT", "SPLIT_COEFFICIENT"]
LABEL = "CLOSE"

feature_cols = [tf.feature_column.numeric_column(k)
                  for k in FEATURES]

def generate_estimator(output_dir):
  return tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                            hidden_units=[10, 10],
                                            model_dir=output_dir)


def generate_input_fn(data_set):
    def input_fn():
      features = {k: tf.constant(data_set[k].values) for k in FEATURES}
      labels = tf.constant(data_set[LABEL].values)
      # labels = {k: tf.constant(data_set[k].values) for k in LABELS}
      return features, labels
    return input_fn



def serving_input_fn():
  #feature_placeholders are what the caller of the predict() method will have to provide
  feature_placeholders = {
      column.name: tf.placeholder(column.dtype, [None])
      for column in feature_cols
  }
  
  #features are what we actually pass to the estimator
  features = {
    # Inputs are rank 1 so that we can provide scalars to the server
    # but Estimator expects rank 2, so we expand dimension
    key: tf.expand_dims(tensor, -1)
    for key, tensor in feature_placeholders.items()
  }
  return tf.estimator.export.ServingInputReceiver(
    features, feature_placeholders
  )

train_spec = tf.estimator.TrainSpec(
                input_fn=generate_input_fn(data_train),
                max_steps=3000)

exporter = tf.estimator.LatestExporter('Servo', serving_input_fn)

eval_spec=tf.estimator.EvalSpec(
            input_fn=generate_input_fn(data_test),
            steps=1,
            exporters=exporter)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # Input Arguments
  parser.add_argument(
      '--output_dir',
      help='GCS location to write checkpoints and export models',
      required=True
  )
  parser.add_argument(
        '--job-dir',
        help='this model ignores this field, but it is required by gcloud',
        default='junk'
    )
  args = parser.parse_args()
  arguments = args.__dict__
  output_dir = arguments.pop('output_dir')

  tf.estimator.train_and_evaluate(generate_estimator(output_dir), train_spec, eval_spec)        
