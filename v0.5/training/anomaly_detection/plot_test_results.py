import sys, os
import re
import numpy as np
import plot_utils
import pylab
SCALE = 4.2

model_dir = 'ares_hwa_results_v10'
model_dir = 'ares_hwa_results_xneg'
#model_dir = '/dccstor/transformer/charles/extracted_autoencoder_models/ares_hwa_results_v10'

def unique(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = list(list_set)
    return unique_list
def variable_key_list(params):
    d = {}
    key_list, val_list = [], []

    for param in params:
        for key, val in param.items():
            key_list.append(key)
            val_list.append(val)
            d[key] = val

    plot_key_list = []
    for i, key in enumerate(key_list):
        if (key, val_list[i]) not in d.items(): # means that there vals differed across the key (should be plotted)
            plot_key_list.append(key)

    plot_key_list = unique(plot_key_list)
    return sorted(plot_key_list)

# LOAD PARAMS FROM ALL SAVED CHECKPOINTS
chkpt = []
roc_auc = []
params = []

for dir in os.listdir(model_dir):
    if dir.startswith('chkpt'):
        # print(dir)
        chkpt.append(dir)
        results_path = os.path.join(model_dir, dir)
        try:
            with open(results_path + '/result.csv') as f:
                lines = f.readlines()
                roc_auc.append(float(re.split(',|\n', lines[-2])[1]))
            f.close()
        except IOError:
            print("File does not exist: %s" %(results_path + '/result.csv'))
            break

        d = {}
        try:
            with open(results_path + '/param_dict.txt') as f:
                lines = f.readlines()
                for line in lines:
                    arr = re.split(': |\n', line)
                    key = arr[0]
                    val = arr[1]
                    d[key] = val
                params.append(d)
            f.close()
        except IOError:
            print("File does not exist: %s" % (results_path + '/result.csv'))
            break

key_list = variable_key_list(params)
print(key_list)
key_list.remove('result_dir')

new_params = []
for i, params_dict in enumerate(params):
    d = {}
    for key, val in params_dict.items():
        if key in key_list:
            d[key] = val
    new_params.append(d)
params = new_params

inds = np.flip(np.argsort(np.asarray(roc_auc)))
roc_auc = [roc_auc[i] for i in inds]
chkpts = [chkpt[i] for i in inds]
params = [params[i] for i in inds]

for auc, chkpt, param in zip(roc_auc, chkpts, params):
    #print("auc = %0.2f, chkpt = %s, params = %s" %(auc, chkpt, str(param).replace('\'', '')))
    print("auc = %0.2f, params = %s" %(auc, str(param).replace('\'', '')))


