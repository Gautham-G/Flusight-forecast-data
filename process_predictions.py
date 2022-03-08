"""	
	This is for parsing from CDC's github repo
	Output: score card containing all models
	i.e., 
		model, forecast_week, ahead, location (as region abbreviation), type, quantile, value
	e.g.
		GT-FluFNP, 202205, 1, CA, point, NaN, 843
		GT-FluFNP, 202205, 1, CA, quantile, 0.01, 338
		....
		GT-FluFNP, 202205, 2, CA, point, NaN, 900
		GT-FluFNP, 202205, 2, CA, quantile, 0.01, 438
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
from epiweeks import Week
import pdb

# death_target = ['1 wk ahead inc death' , '2 wk ahead inc death' , '3 wk ahead inc death' , '4 wk ahead inc death']

models = ['GT-FluFNP','Flusight-ensemble']
data_ew = Week.thisweek(system="CDC") - 1  # -1 because we have data for the previous (ending) week
DIR =  './data-forecasts/'

# for each model, get all submissions
for model in models:
	model_dir = DIR + '/' + model + '/' 

	all_items_path = np.array(glob.glob(model_dir + '*.csv'))  # list all csv files' paths
	all_items = [path.replace(model_dir, '') for path in all_items_path]  #list of all csv files' names

	"""
		remove forecasts that were duplicated in a given week (if any)
			forecasts file should be unique for each epiweek
	"""
	subm_dict = {}
	for i, item in enumerate(all_items):
		date = datetime.strptime(item[:10], '%Y-%m-%d')
		epiweek  = date.isocalendar()[1]
		if epiweek in subm_dict.keys():
			if subm_dict[epiweek][0] <= date:
				subm_dict[epiweek] = (date, i)
		else:
			subm_dict[epiweek] = (date, i)

	select = [ value[1] for key, value in subm_dict.items()]
	select_paths = all_items_path[select]


	data_model = []
	for path in select_paths:

		df = pd.read_csv(path)
		
		"""
			create epiweek column
		"""
		date = path.split('/')[-1][:10]
		# epiweek ends on Saturday, but submission is until Monday. 
		# we can subtract 2 days, thus, submission on Monday will be considered in the prev week  
		# this also aligns submission week and data
		date = datetime.strptime(date, '%Y-%m-%d') - timedelta(days=2)
		forecast_week = Week.fromdate(date)
		df['forecast_week'] = forecast_week
		pdb.set_trace()

		data_model.append(df)

	# join all dataframes saved in data_model

	"""
		select, rename and sort columns
	"""
		

	"""
		convert location to region abbreviation
	"""



