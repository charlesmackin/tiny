import os, sys
import numpy as np
import plot_utils

weights_target = np.loadtxt('TargetW.txt') / 100.
weights_prog = np.loadtxt('ActualW.txt') / 100.

plot_utils.plot_scatter_heatmap(weights_target, weights_prog, XLABEL='Target Weights [1]', YLABEL='Actual Weights [1]', SAVE_NAME='Weight Programming Heatmap')

p = np.polyfit(weights_target.flatten(), weights_prog.flatten(), 1)
fit = np.poly1d(p)
x = weights_target
y = fit(x)

errors = weights_prog - y

print("max weight error = %0.2f" %np.amax(np.abs(errors.flatten())))
print("mean weight error = %f" %np.mean(errors))
print("std weight error = %f" %np.std(errors))

plot_utils.plot_scatter_heatmap(weights_target, errors, XLABEL='Target Weights [1]', YLABEL='Weight Errors [1]', SAVE_NAME='Weight Programming Errors Heatmap')

plot_utils.plot_pdf(errors, bins=20, XLABEL='Weight Errors [1]', YLABEL='PDF [1]', SAVE_NAME='Weight Programming Errors PDF')
#errors = np.random.randn(10000)
plot_utils.plot_qq(errors, XLABEL='Theoretical Quantiles [1]', YLABEL='Sample Quantiles [1]', SAVE_NAME='qq plot')
