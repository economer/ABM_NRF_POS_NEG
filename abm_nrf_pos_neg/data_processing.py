import pandas as pd
import numpy as np


N_SAMPLE = 2000
category = 'sex'

# import data
file_path = '/Users/shh/Google_Drive/CALG/ABM/ABM Nov 2023/cchs_final_nrf.csv'
data = pd.read_csv(file_path)
data_cchs = data
data_cchs['one'] = 'one_category'

# Transformations
data_s = data_cchs[['hh_inc_per_capita', 'pos', 'neg', 'nrf', 'age_sex']].copy()
data_s['biweekly_income'] = data_s['hh_inc_per_capita'] / 26
data_s = data_s[(data_s['biweekly_income'] > 0) & (data_s['pos'] > 0) & (data_s['neg'] > 0)]

data_s['inv_bw_inc'] = 1 / data_s['biweekly_income']
data_s['log_bw_inc'] = np.log(data_s['biweekly_income'])
data_s['caseid'] = data_cchs['caseid']


data_s['age_sex'] = data_cchs[category]

#data_s['age_sex'] = data_cchs['sex']
#data_s['age_sex'] = 'one'

data_s['fsddekc'] = data_cchs['fsddekc']
data_s['fsddfi'] = data_cchs['fsddfi']
data_s['fsddcal'] = data_cchs['fsddcal']
data_s['fsddpot'] = data_cchs['fsddpot']
data_s['fsddpro'] = data_cchs['fsddpro']
data_s['fsddiro'] = data_cchs['fsddiro']
data_s['fsddrae'] = data_cchs['fsddrae']
data_s['fsdddmg'] = data_cchs['fsdddmg']
data_s['fsddc'] = data_cchs['fsddc']
data_s['fsddmag'] = data_cchs['fsddmag']

data_s['fsddsod'] = data_cchs['fsddsod']
data_s['fsddfas'] = data_cchs['fsddfas']
data_s['fsddsug'] = data_cchs['fsddsug']


# Sampling
data_new = data_s.sample(n=N_SAMPLE, random_state=42)
#data_new = data_s





