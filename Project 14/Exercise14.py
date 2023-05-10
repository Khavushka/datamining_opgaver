import numpy as np
import pandas as pd



# Henter dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
breast_cn = pd.read_csv(url, header=None)
