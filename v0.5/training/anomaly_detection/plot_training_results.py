import sys, os
import torch
import numpy as np
import plot_utils
import pylab
SCALE = 2.0
#SCALE = 4.2
#SCALE = 1.2
print(SCALE)

model_dir = 'ares_hwa_models_v10'
model_dir = 'ares_hwa_models_complex'
model_dir = 'ares_hwa_models_xneg'
#model_dir = '/dccstor/transformer/charles/extracted_autoencoder_models/ares_hwa_models_v10'
extension = '.pt'

def unique(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = list(list_set)
    return unique_list
def variable_key_list(chkpt_list):
    d = {}
    key_list, val_list = [], []
    for chkpt in chkpt_list:
        for key, val in chkpt["param"].items():
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
params_list = []
chkpt_list = []
for file in os.listdir(model_dir):
    if file.endswith(extension):
        chkpt_path = os.path.join(model_dir, file)
        #print(chkpt_path)
        try:
            chkpt = torch.load(chkpt_path)
            chkpt_list.append(chkpt)
            params_list.append(chkpt["param"])
        except IOError:
            print("File does not exist: %s" % (chkpt_path))

print("Number of chkpts: %d" %len(chkpt_list))
print("Number of params: %d" %len(params_list[0]))

key_list = variable_key_list(chkpt_list)
print("key list = %s" %str(key_list))

# filter checkpoints if necessary
chkpt_list = [chkpt for chkpt in chkpt_list if chkpt["loss_history"][-1] < 20.]

#chkpt_list = [chkpt for chkpt in chkpt_list if chkpt["param"]["noise_std"] > 0.]
end_losses = np.asarray([chkpt["loss_history"][-1] for chkpt in chkpt_list])
epochs = [len(chkpt["loss_history"]) for chkpt in chkpt_list]
inds = np.argsort(end_losses)
inds = inds[0:30] # only take top 20
chkpt_list = [chkpt_list[i] for i in inds]
epochs = [epochs[i] for i in inds]

y_list = [np.asarray(chkpt["loss_history"]) for chkpt in chkpt_list]
y_list_valid = [np.asarray(chkpt["valid_loss_history"]) for chkpt in chkpt_list]
x_list = [np.arange(1, y.size + 1, 1) for y in y_list]

new_legend_list = []
legend_list = [chkpt["param"] for chkpt in chkpt_list]
for i, params_dict in enumerate(legend_list):
    d = {}
    for key, val in params_dict.items():
        if key in key_list:
            d[key] = val
    new_legend_list.append(d)
legend_list = new_legend_list

legend_list = [str(legend).replace('{', '').replace('}', '').replace('\'', '') for legend in legend_list]

print("len(x_list) = %d" %len(x_list))
print("len(y_list) = %d" %len(y_list))

end_loss = np.asarray([loss_history[-1] for loss_history in y_list])
inds = list(np.argsort(end_loss))

end_loss = [end_loss[i] for i in inds]
x_list = [x_list[i] for i in inds]
y_list = [y_list[i] for i in inds]
legend_list = [legend_list[i] for i in inds]

#[print("loss=%0.2f, %s" %(end_loss[i], legend_list[i])) for i in range(len(end_loss))]
[print("loss=%0.2f, epoch=%d, %s" %(end_loss[i], epochs[i], legend_list[i])) for i in range(len(end_loss))]


#print([val, legend for in zip(end_loss, legend_list)])

# PLOT CHECKPOINTS
pylab.figure(figsize=(SCALE * 6.4, SCALE * 4.8))
plot_utils.plot_loglog_overlay_subplot(x_list,
                                       y_list,
                                       XLABEL='Epoch',
                                       YLABEL='Train Loss',
                                       LEGEND_LIST=legend_list
                                       )
#pylab.ylim(9, 100)
#pylab.legend(legend_list)
#pylab.legend(bbox_to_anchor=(0.,0.), loc="lower left")
pylab.savefig(model_dir + "_training_loss.png")
pylab.close()

# pylab.figure(figsize=(SCALE * 6.4, SCALE * 4.8))
# plot_utils.plot_loglog_overlay_subplot(x_list,
#                                        y_list_valid,
#                                        XLABEL='Epoch',
#                                        YLABEL='Valid Loss',
#                                        LEGEND_LIST=legend_list
#                                        )
# #pylab.ylim(1, 100)
# pylab.savefig("training_valid_summary.png")
# pylab.close()
#
#
# train_loss = np.asarray([chkpt["loss_history"][-1] for chkpt in chkpt_list])
# valid_loss = np.asarray([chkpt["valid_loss_history"][-1] for chkpt in chkpt_list])
#
# plot_utils.plot_scatter(train_loss,
#                         valid_loss,
#                         XLABEL='Train Loss',
#                         YLABEL='Valid Loss',
#                         SAVE_NAME='Train vs Valid Loss'
#                         )