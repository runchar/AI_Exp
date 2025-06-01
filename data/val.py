import pandas as pd
import numpy as np

valdata = np.load('./output/final_y.npy', allow_pickle=True)

print(valdata.shape)
print(valdata)
