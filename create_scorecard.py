"""
    Output: score card containing all models
	i.e., 
		model, location (as region abbreviation), RMSE, CRSP, CS

    if missing value for a model, remove the model
"""
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from epiweeks import Week
import pdb

ground_dir = './data-truth/truth-Incident Hospitalizations.csv'  

df_gt = pd.read_csv(ground_dir)

print(df_gt['date'])

def convert2week(x):
    date = datetime.strptime(x,"%Y-%m-%d")
    return Week.fromdate(date)

week = df_gt['date'].apply(convert2week)
# create a column with the week for which the data was observed
df_gt['week'] = week

# convert location to region abbreviation
pdb.set_trace()

