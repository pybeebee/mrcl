import os
import seaborn as sns
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

results_dir = "/om/user/gkml/results"
accuracies = []

for f in os.listdir(results_dir):
    if "DS_St" not in f:
        folders.append(f)
        with open(os.path.join(results_dir, f, "metadata.json")) as json_file:
            data_temp = json.load(json_file)
            print(f)
            # print([x[1]['0'][0] for x in data_temp['results']['Final Results']])
            #data[f] = [x[1]['0'][0] for x in data_temp['results']['Final Results']]
        # print(data)e
    # quit()
    # print(f)

