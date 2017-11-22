
import numpy as np
import pandas as pd
import pickle

filename = '/data/cdiscount/test152/result.hdf'
print(filename)
X = pd.read_hdf(filename, "result").values
pk_path = '/data/cdiscount/test152/pk.11'
pk_out = open(pk_path, 'wb')
for ii in range(X.shape[0]):
  score = X[ii]
  pickle.dump(score, pk_out, pickle.HIGHEST_PROTOCOL)
pk_out.close()

