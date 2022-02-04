import os, sys
import numpy as np
import plot_utils

dir2 = "ares_hwa_layers4_h128_c24_result"
dir1 = "result"
dir_out = "ROC_comparison"
os.makedirs(dir_out, exist_ok=True)

a, b = [], []
for i in range(4):
    a.append(np.loadtxt(dir1 + '/ROC_data_id_0' + str(i+1) + '.txt', delimiter=','))
    b.append(np.loadtxt(dir2 + '/ROC_data_id_0' + str(i+1) + '.txt', delimiter=','))

    x_list = [a[i][:,0], b[i][:,0], np.arange(0,1,0.01)]
    y_list = [a[i][:,1], b[i][:,1], np.arange(0,1,0.01)]

    auc_fp = np.abs(np.trapz(a[i][:,1], x=a[i][:,0]))  # abs because order
    auc_hw = np.abs(np.trapz(b[i][:,1], x=b[i][:,0]))  # abs because order

    legend_list = ["FP ROC (AUC = %0.2f)" %auc_fp, "HW ROC (AUC = %0.2f)" %auc_hw, "Reference"]
    plot_utils.plot_1d_overlay(x_list,
                               y_list,
                               LEGEND_LIST=legend_list,
                               XLABEL='False Positive Rate',
                               YLABEL='True Positive Rate',
                               TITLE='ToyCar ID = %d' %(i+1),
                               SAVE_NAME= dir_out + '/ROC_' + str(i),
                               markers=False
                               )
