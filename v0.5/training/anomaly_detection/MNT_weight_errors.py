import plot_utils
import torch
import ares_torch_model_base as mnt
import numpy as np

rows = 1000
cols = 1000

mnt_std = 0.03

weights = torch.clamp(torch.randn(rows, cols), min=-1., max=1.)
weights = torch.tensor(np.loadtxt('TargetW.txt').flatten() / 100.)

noisy_weights_lognormal = mnt.noise(weights, 'log-normal', mnt_std, 'cpu')
noisy_weights_normal = mnt.noise(weights, 'normal', mnt_std, 'cpu')
noisy_weights_uniform = mnt.noise(weights, 'uniform', mnt_std, 'cpu')
noisy_weights_custom = mnt.noise(weights, 'custom-log-normal', mnt_std, 'cpu')
noisy_weights_additive = mnt.noise(weights, 'additive-noise', mnt_std, 'cpu')

# plot_utils.plot_scatter(weights.detach().cpu().numpy(),
#                         noisy_weights_lognormal.detach().cpu().numpy(),
#                         XLABEL='Weights [1]',
#                         YLABEL='Noisy Weights [1]',
#                         TITLE='MNT: log-normal',
#                         SAVE_NAME='MNT Log-Normal Weight Errors')
#
# plot_utils.plot_scatter(weights.detach().cpu().numpy(),
#                         noisy_weights_normal.detach().cpu().numpy(),
#                         XLABEL='Weights [1]',
#                         YLABEL='Noisy Weights [1]',
#                         TITLE='MNT: normal',
#                         SAVE_NAME='MNT Normal Weight Errors')
#
# plot_utils.plot_scatter(weights.detach().cpu().numpy(),
#                         noisy_weights_uniform.detach().cpu().numpy(),
#                         XLABEL='Weights [1]',
#                         YLABEL='Noisy Weights [1]',
#                         TITLE='MNT: uniform',
#                         SAVE_NAME='MNT Uniform Weight Errors')





# plot_utils.plot_scatter_heatmap(weights.detach().cpu().numpy(),
#                                 noisy_weights_lognormal.detach().cpu().numpy(),
#                                 #bins=(200,200),
#                                 XLABEL='Weights [1]',
#                                 YLABEL='Noisy Weights [1]',
#                                 TITLE='MNT: log-normal',
#                                 SAVE_NAME='MNT Log-Normal Weight Errors heatmap')
#
# plot_utils.plot_scatter_heatmap(weights.detach().cpu().numpy(),
#                                 noisy_weights_normal.detach().cpu().numpy(),
#                                 #bins=(200,200),
#                                 XLABEL='Weights [1]',
#                                 YLABEL='Noisy Weights [1]',
#                                 TITLE='MNT: normal',
#                                 SAVE_NAME='MNT Normal Weight Errors heatmap')
#
# plot_utils.plot_scatter_heatmap(weights.detach().cpu().numpy(),
#                                 noisy_weights_uniform.detach().cpu().numpy(),
#                                 #bins=(200,200),
#                                 XLABEL='Weights [1]',
#                                 YLABEL='Noisy Weights [1]',
#                                 TITLE='MNT: uniform',
#                                 SAVE_NAME='MNT Uniform Weight Errors heatmap')
#
# plot_utils.plot_scatter_heatmap(weights.detach().cpu().numpy(),
#                                 noisy_weights_custom.detach().cpu().numpy(),
#                                 #bins=(200,200),
#                                 XLABEL='Weights [1]',
#                                 YLABEL='Noisy Weights [1]',
#                                 TITLE='MNT: custom-log-normal',
#                                 SAVE_NAME='MNT Custom Log-Normal Weight Errors heatmap')

plot_utils.plot_scatter_heatmap(weights.detach().cpu().numpy(),
                                noisy_weights_additive.detach().cpu().numpy(),
                                #bins=(200,200),
                                XLABEL='Weights [1]',
                                YLABEL='Noisy Weights [1]',
                                #TITLE='Additive Weight Errors',
                                SAVE_NAME='Additive Weight Errors Model Heatmap')
