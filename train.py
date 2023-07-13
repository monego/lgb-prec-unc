import numpy as np
import pandas as pd
from models.models import (PrecModel, UncModel)

print("Preprocessing data...")

wd = pd.read_feather('Dados_Jan1980_mar2020_interpolado.feather')
wdc = wd.sort_values(by=['lat', 'lon'])

# Add a new column for next month's precipitation
wdc['prec_GPCP_roll'] = np.roll(wdc['prgpcp'], -1)

# Remove data from 2020
wdc = wdc.loc[wdc['year'] < 2020]

print("Training Model 1 (Precipitation)")

Prec = PrecModel(wdc, 2018, 0.25, "prec")

Prec.fit()
# Prec.best_params()
# Prec.plot_learning_curve()
Prec.train_test_error()
Prec.print_errors()
Prec.save_trials()
Prec.calc_variance()
Prec.save_model()
Prec.save_data("output-model-1.xlsx")

print("Training Model 2 (Uncertainty)")

unc_data = pd.read_excel("output-model-1.xlsx")
Unc = UncModel(unc_data, 2018, 0.25, "unc")
Unc.fit()
# Unc.plot_learning_curve()
# Unc.best_params()
Unc.train_test_error()
Unc.print_errors()
Unc.save_trials()
Unc.save_model()
Unc.save_data()
