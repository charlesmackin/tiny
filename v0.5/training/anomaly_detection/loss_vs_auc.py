import sys, os, re
import torch
import numpy as np
import plot_utils
import pylab
SCALE = 1.2

#loss_dir = '/dccstor/transformer/charles/extracted_autoencoder_models/ares_hwa_models_v10'
loss_dir = os.getcwd() + '/ares_hwa_models_v10'
loss_dir = os.getcwd() + '/ares_hwa_models_xneg'
extension = '.pt'

#auc_dir = '/dccstor/transformer/charles/extracted_autoencoder_models/ares_hwa_results_v10'
auc_dir = os.getcwd() + '/ares_hwa_results_v10'
auc_dir = os.getcwd() + '/ares_hwa_results_xneg'

chkpt_name_list = []
loss_list = []
for file in os.listdir(loss_dir):
    if file.endswith(extension):
        chkpt_name_list.append(file.replace('.pt', ''))
        chkpt_path = os.path.join(loss_dir, file)
        chkpt = torch.load(chkpt_path)
        loss_list.append(chkpt["loss_history"][-1])

dir_name_list = []
auc_list = []
for dir in os.listdir(auc_dir):
    dir_name_list.append(dir)
    results_path = os.path.join(auc_dir, dir)
    with open(results_path + '/result.csv') as f:
        lines = f.readlines()
        auc_list.append(float(re.split(',|\n', lines[-2])[1]))
    f.close()

print(chkpt_name_list)
print(dir_name_list)

assert len(chkpt_name_list) == len(dir_name_list), "Error: not same number of elements"
#assert set(chkpt_name_list) == set(dir_name_list), "Error: lists don't have same contents"

chkpt_name_list, loss_list = zip(*sorted(zip(chkpt_name_list, loss_list)))
dir_name_list, auc_list = zip(*sorted(zip(dir_name_list, auc_list)))

losses = np.asarray(loss_list)
aucs = np.asarray(auc_list)

plot_utils.plot_scatter(losses, aucs, XLABEL="Loss [1]", YLABEL="ROC AUC [1]", SAVE_NAME="Loss vs AUC")

pylab.figure(figsize=(SCALE*6.4, SCALE*4.8))
plot_utils.plot_scatter_subplot(losses, aucs, XLABEL="Loss [1]", YLABEL="ROC AUC [1]", **{'SIZE': 2})
pylab.ylim(0.4, 0.9)
pylab.savefig("Loss vs AUC subplot.png")
