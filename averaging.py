import json

# define a list of keywords
path = '/home/gkml/results/29October2019/omniglot/eval/1/mrcl_omn_r2E_1/meta_data.json'

vals = []

with open(path) as json_file:

    # read json file line by line
    for line in json_file.readlines():
    	if 'Final results' in line:
    		vals.append(str(line))

print(np.array(vals).reshape((len(vals),1)))
