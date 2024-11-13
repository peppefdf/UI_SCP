import pandas as pd

root_dir = "C:/Users/gfotidellaf/repositories/UI_SCP/assets/data/input_data_MCM/"
filename = "workers_eskuzaitzeta_2k.csv"

data = pd.read_csv(root_dir + filename)

data['Hora_Ini'] = '08:10'

#data.to_csv(root_dir + 'workers_eskuzaitzeta_2k_' + 'HoraIni_8_10.csv', index=False)

#shuffle dataframe
data = data.sample(frac = 1)

#data_large_companies = data.sample(n=200, random_state=1)
data_large_company = data.iloc[:200]
data_large_company.to_csv(root_dir + 'workers_eskuzaitzeta_2k_' + 'large_company.csv', index=False)

data_small_companies = data.iloc[200:800]
data_small_companies.to_csv(root_dir + 'workers_eskuzaitzeta_2k_' + 'small_companies.csv', index=False)

data_all_companies = data.iloc[:800]
data_all_companies.to_csv(root_dir + 'workers_eskuzaitzeta_2k_' + 'all_companies.csv', index=False)