import pylab
import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cycler import cycler

matplotlib.rcParams.update({'font.size': 22})
matplotlib.use('Agg')


import statsmodels.api as sm
from scipy.stats import norm

#from mpl_toolkits import mplot3d
#from mpl_toolkits.mplot3d import Axes3D

plt.rc('axes', labelsize=32)    # fontsize of the h and y labels
plt.rc('axes', labelsize=28)    # fontsize of the h and y labels
plt.rc('legend',fontsize=20)

SCALE = 1.2

def get_ecdf(x):
    x_sorted = np.sort(x.flatten())
    ecdf = np.arange(len(x_sorted)) / len(x_sorted)
    return x_sorted, ecdf
def plot_qq(x, XLABEL='', YLABEL='', TITLE='', SAVE_NAME='', **kwargs):
    x = x.flatten()
    mu = np.mean(x)
    std = np.std(x)
    pylab.figure(figsize=(SCALE*6.4, SCALE*4.8))
    #sm.qqplot((x-mu)/std, fit=True, line='45')
    sm.qqplot(x, fit=True, line='45')
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.tight_layout()
    pylab.savefig(SAVE_NAME+".png")
    pylab.close()
def plot_scatter(x, y, XLABEL='', YLABEL='', TITLE='', SAVE_NAME='', **kwargs):
    x = np.float32(x)
    y = np.float32(y)
    poly = np.polyfit(x.flatten(), y.flatten(), 1)
    pylab.figure(figsize=(SCALE*6.4, SCALE*4.8))
    pylab.plot(x.flatten(), y.flatten(), '.')
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    #pylab.title(TITLE + ", poly = " + str(poly))
    pylab.tight_layout()
    pylab.savefig(SAVE_NAME+".png")
    pylab.close()
def plot_scatter_overlay(x_list, y_list, XLABEL='', YLABEL='', TITLE='', LEGEND_LIST='', SAVE_NAME='', **kwargs):
    marker = itertools.cycle(('>', '<', '+', '.', 'o', '*'))    # RESET markers
    pylab.figure(figsize=(SCALE*6.4, SCALE*4.8))
    for i in range(len(x_list)):
        poly = np.polyfit(x_list[i].flatten(), y_list[i].flatten(), 1)
        pylab.scatter(x_list[i].flatten(), y_list[i].flatten(), mark=next(marker), label=LEGEND_LIST[i])
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.legend(LEGEND_LIST)
    pylab.title(TITLE)
    pylab.tight_layout()
    pylab.savefig(SAVE_NAME+".png")
    pylab.close()
def plot_scatter_overlay_subplot(x_list, y_list, XLABEL='', YLABEL='', TITLE='', LEGEND_LIST='', **kwargs):
    marker = itertools.cycle(('>', '<', '+', '.', 'o', '*'))    # RESET markers
    for i in range(len(x_list)):
        poly = np.polyfit(x_list[i].flatten(), y_list[i].flatten(), 1)
        pylab.scatter(x_list[i].flatten(), y_list[i].flatten(), marker=next(marker), label=LEGEND_LIST[i])
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.legend(LEGEND_LIST)
    pylab.title(TITLE)
    pylab.tight_layout()
def plot_scatter_subplot(x, y, XLABEL='', YLABEL='', TITLE='', **kwargs):
    color = kwargs.get('COLOR', 'blue')
    size = kwargs.get('SIZE', 1)
    poly = np.polyfit(x.flatten(), y.flatten(), 1)
    #pylab.plot(h.flatten(), y.flatten(), color=color, marker='.', ms=1)
    pylab.scatter(x.flatten(), y.flatten(), s=size, c=color)
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.tight_layout()
def plot_scatter_heatmap_subplot(x, y, XLABEL='', YLABEL='', TITLE='', CLABEL='PDF', bins=(100,100), R2=False, **kwargs):
    poly = np.polyfit(x.flatten(), y.flatten(), 1)
    heatmap, xedges, yedges = np.histogram2d(x.flatten(), y.flatten(), bins=bins, density=True)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    heatmap = pylab.imshow(heatmap.T, extent=extent, origin='lower', interpolation='nearest', cmap=pylab.cm.rainbow, norm=matplotlib.colors.LogNorm())
    cbar = pylab.colorbar(heatmap)
    cbar.set_label(CLABEL)
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    if R2 and TITLE == '':
        TITLE = r"$R^2$" + " = %0.3f" %np.corrcoef(x,y)[0,1]**2
    elif R2 and TITLE != '':
        TITLE = TITLE + r"$, \ R^2$" + " = %0.3f" %np.corrcoef(x,y)[0,1]**2
    pylab.title(TITLE)
    pylab.tight_layout()
def heatmap(image, XLABEL='', YLABEL='', CLABEL='', TITLE='', SAVE_NAME=''):
    pylab.figure(figsize=(SCALE * 6.4, SCALE * 4.8))
    heatmap = pylab.imshow(image.T, origin='lower', interpolation='nearest',
                           cmap=pylab.cm.rainbow, norm=matplotlib.colors.LogNorm())
    cbar = pylab.colorbar(heatmap)
    cbar.set_label(CLABEL)
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.tight_layout()
    pylab.savefig(SAVE_NAME + '.png')
    pylab.close()
def plot_scatter_heatmap_overlay_subplot(x_list, y_list, XLABEL='', YLABEL='', TITLE='', CLABEL='PDF', LEGEND_LIST=None, COLORS=None, bins=(100,100), R2=False, **kwargs):

    heatmap, xedges, yedges = np.histogram2d(np.concatenate(x_list).flatten(), np.concatenate(y_list).flatten(), bins=bins, density=True)

    for i, x in enumerate(x_list):
        poly = np.polyfit(x_list[i].flatten(), y_list[i].flatten(), 1)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        heatmap, xedges, yedges = np.histogram2d(x_list[i], y_list[i], bins=[xedges, yedges], density=True)
        heatmap = pylab.imshow(heatmap.T, extent=extent, origin='lower', interpolation='nearest', cmap=pylab.cm.rainbow, norm=matplotlib.colors.LogNorm())
    cbar = pylab.colorbar(heatmap)
    cbar.set_label(CLABEL)
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    if R2 and TITLE == '':
        TITLE = r"$R^2$" + " = %0.3f" %np.corrcoef(x,y)[0,1]**2
    elif R2 and TITLE != '':
        TITLE = TITLE + r"$, \ R^2$" + " = %0.3f" %np.corrcoef(x,y)[0,1]**2
    pylab.title(TITLE)
    pylab.tight_layout()
def plot_scatter_heatmap(x, y, XLABEL='', YLABEL='', TITLE='', CLABEL='PDF', SAVE_NAME='', bins=(100,100), aspect='auto', R2=False, DISP_FIT=False, **kwargs):
    pylab.figure(figsize=(SCALE*6.4, SCALE*4.8))
    poly = np.polyfit(x.flatten(), y.flatten(), 1)
    heatmap, xedges, yedges = np.histogram2d(x.flatten(), y.flatten(), bins=bins, density=True)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    if 'square' in aspect.lower():
        aspect = (xedges[-1]-xedges[0])/(yedges[-1]-yedges[0])
    heatmap = pylab.imshow(heatmap.T, extent=extent, aspect=aspect, origin='lower', interpolation='nearest', cmap=pylab.cm.rainbow, norm=matplotlib.colors.LogNorm())
    cbar = pylab.colorbar(heatmap)
    cbar.set_label(CLABEL)
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    if R2 and TITLE == '':
        TITLE = r"$R^2$" + " = %0.3f" %np.corrcoef(x,y)[0,1]**2
    elif R2 and TITLE != '':
        TITLE = TITLE + r"$, \ R^2$" + " = %0.3f" %np.corrcoef(x,y)[0,1]**2
    elif DISP_FIT:
        TITLE = "poly = " + str(poly) + "\n" + r"$R^2$" + " = %0.3f" %np.corrcoef(x,y)[0,1]**2
    print(np.corrcoef(x,y)[0,1]**2)
    pylab.title(TITLE)
    pylab.tight_layout()
    pylab.savefig(SAVE_NAME+".png")
    pylab.close()
def plot_scatter_heatmap_subplot(x, y, XLABEL='', YLABEL='', TITLE='', CLABEL='PDF', bins=(100,100), aspect='auto', R2=False, **kwargs):
    #pylab.figure(figsize=(SCALE*6.4, SCALE*4.8))
    poly = np.polyfit(x.flatten(), y.flatten(), 1)
    heatmap, xedges, yedges = np.histogram2d(x.flatten(), y.flatten(), bins=bins, density=True)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    LABEL_FONTSIZE = kwargs.get('xylabel_fontsize', None)
    if 'square' in aspect.lower():
        aspect = (xedges[-1]-xedges[0])/(yedges[-1]-yedges[0])
    heatmap = pylab.imshow(heatmap.T, extent=extent, aspect=aspect, origin='lower', interpolation='nearest', cmap=pylab.cm.rainbow, norm=matplotlib.colors.LogNorm())
    cbar = pylab.colorbar(heatmap)
    cbar.set_label(CLABEL)
    pylab.yticks(np.arange(-1, 1.1, step=0.5))
    if LABEL_FONTSIZE is not None:
        pylab.xlabel(XLABEL, fontsize=LABEL_FONTSIZE)
        pylab.ylabel(YLABEL, fontsize=LABEL_FONTSIZE)
    else:
        pylab.xlabel(XLABEL)
        pylab.ylabel(YLABEL)
    if R2 and TITLE == '':
        TITLE = r"$R^2$" + " = %0.3f" %np.corrcoef(x,y)[0,1]**2
    elif R2 and TITLE != '':
        TITLE = TITLE + r"$, \ R^2$" + " = %0.3f" %np.corrcoef(x,y)[0,1]**2
    pylab.ylim(-1, 1)
    pylab.title(TITLE)
    pylab.tight_layout()
    #pylab.savefig(SAVE_NAME+".png")
    #pylab.close()
def plot_scatter_3d(x, y, z, XLABEL='', YLABEL='', ZLABEL='', TITLE='', SAVE_NAME='', **kwargs):
    #ax = plt.axes(111, projection='3d')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_aspect('equal')  # equal aspect ratio (cube)
    #ax.set_box_aspect((1, 1, 1))    # Data for a three-dimensional line
    # zline = np.linspace(0, 15, 1000)
    # xline = np.sin(zline)
    # yline = np.cos(zline)
    # ax.plot3D(xline, yline, zline, 'gray')
    #ax.scatter3D(h, y, z, c=zdata, cmap='Greens')
    ax.scatter3D(x, y, z)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_zlabel(ZLABEL)   # pylab.zlabel(ZLABEL) doesn't work
    #ax.view_init(60, 35)
    plt.title(TITLE)
    plt.tight_layout()
    plt.savefig(SAVE_NAME+".png")
    plt.close()
def plot_1d(x, y, XLABEL='', YLABEL='', TITLE='', SAVE_NAME='', **kwargs):
    pylab.figure(figsize=(SCALE*6.4, SCALE*4.8))
    pylab.plot(x.flatten(), y.flatten(), '-b')
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.tight_layout()
    pylab.savefig(SAVE_NAME+".png")
    pylab.close()
def plot_1d_error_bars(x, y, s, XLABEL='', YLABEL='', TITLE='', SAVE_NAME='', **kwargs):
    pylab.figure(figsize=(SCALE*6.4, SCALE*4.8))
    pylab.plot(x, y)
    pylab.fill_between(x, y-s, y+s, alpha=0.2)
    #pylab.fill_between(h, y-s, y+s, color='blue', alpha=0.2)
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.tight_layout()
    pylab.savefig(SAVE_NAME+".png")
    pylab.close()
def plot_1d_error_bars_subplot(x, y, s, XLABEL='', YLABEL='', TITLE='', **kwargs):
    COLOR = kwargs.get('COLOR')
    if COLOR:
        pylab.plot(x, y, '-', c=COLOR)
        pylab.fill_between(x, y-s, y+s, color=COLOR, edgecolor=COLOR, linewidth=2, alpha=0.2)
    else:
        pylab.plot(x, y, '-')
        pylab.fill_between(x, y-s, y+s, edgecolor=pylab.gca().lines[-1].get_color(), linewidth=2, alpha=0.2)
    #pylab.fill_between(h, y-s, y+s, color='blue', alpha=0.2)
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.tight_layout()
def plot_1d_semilogx_error_bars(x, y, s, XLABEL='', YLABEL='', TITLE='', SAVE_NAME='', **kwargs):
    pylab.figure(figsize=(SCALE*6.4, SCALE*4.8))
    pylab.semilogx(x, y)
    pylab.fill_between(x, y-s, y+s, alpha=0.2)
    #pylab.fill_between(h, y-s, y+s, color='blue', alpha=0.2)
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.tight_layout()
    pylab.savefig(SAVE_NAME+".png")
    pylab.close()
def plot_1d_semilogx_error_bars_subplot(x, y, s, XLABEL='', YLABEL='', TITLE='', **kwargs):
    pylab.semilogx(x, y)
    pylab.fill_between(x, y - s, y + s, alpha=0.2)
    # pylab.fill_between(h, y-s, y+s, color='blue', alpha=0.2)
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.tight_layout()
    pylab.close()
def plot_1d_semilogy_error_bars_subplot(x, y, s, XLABEL='', YLABEL='', TITLE='', **kwargs):
    #pylab.semilogy(h, y)
    print(x)
    print(y)
    print(np.log10(x))
    print(np.log10(y))
    pylab.plot(x, y)
    #pylab.fill_between(h, y - s, y + s, alpha=0.2)
    pylab.yscale("log")
    # pylab.fill_between(h, y-s, y+s, color='blue', alpha=0.2)
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.tight_layout()
    pylab.close()
def plot_1d_loglog_error_bars_subplot(x, y, s, XLABEL='', YLABEL='', TITLE='', **kwargs):
    pylab.loglog(x, y)
    pylab.fill_between(x, y-s, y+s, alpha=0.2)
    #pylab.fill_between(h, y-s, y+s, color='blue', alpha=0.2)
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.tight_layout()
    pylab.close()
def plot_1d_subplot(x, y, XLABEL='', YLABEL='', TITLE='', **kwargs):
    pylab.plot(x.flatten(), y.flatten(), '-b')
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.tight_layout()
def plot_1d_overlay(x_list, y_list, XLABEL='', YLABEL='', TITLE='', LEGEND_LIST='', SAVE_NAME='', markers=True, **kwargs):
    marker = itertools.cycle(('>', '<', '+', '.', 'o', '*')) if markers else itertools.cycle((' '))  # RESET markers
    #pylab.figure(figsize=(12,12))
    pylab.figure(figsize=(SCALE*6.4, SCALE*4.8))
    for i in range(len(x_list)):
        pylab.plot(x_list[i].flatten(), y_list[i].flatten(), marker=next(marker))
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.legend(LEGEND_LIST)
    pylab.tight_layout()
    pylab.savefig(SAVE_NAME+".png")
    pylab.close()
def plot_1d_overlay_subplot(x_list, y_list, XLABEL='', YLABEL='', TITLE='', LEGEND_LIST='', **kwargs):
    marker = itertools.cycle(('>', '<', '+', '.', 'o', '*'))    # RESET markers
    COLOR_LIST = kwargs.get('COLOR_LIST')
    COLOR_LEGEND = kwargs.get('COLOR_LEGEND')
    MARKER_SCALE = kwargs.get('MARKER_SCALE')
    FONTSIZE = kwargs.get('FONTSIZE')
    LINEWIDTH = kwargs.get('LINEWIDTH')
    LEGEND_FONTSIZE = kwargs.get('LEGEND_FONTSIZE', 12)
    MARKER_OVERRIDE = kwargs.get('MARKER_OVERRIDE')
    ALPHA = kwargs.get('ALPHA', 1.0)
    corr = -0.05

    MKS = kwargs.get('MARKER_SIZE', 6)
    if COLOR_LIST is not None:
        for i in range(len(y_list)):
            mark = next(marker) if MARKER_OVERRIDE is None else None
            if x_list:
                pylab.plot(x_list[i].flatten(), y_list[i].flatten(), marker=mark, color=COLOR_LIST[i], markersize=MKS, linewidth=LINEWIDTH, alpha=ALPHA)
                if MARKER_OVERRIDE is not None:
                    for j, x in enumerate(x_list[i]):
                        ax = pylab.gca()
                        ax.annotate(MARKER_OVERRIDE, xy=(x_list[i][j]+corr, y_list[i][j]+corr), color=COLOR_LIST[i])
            else:
                pylab.plot(y_list[i].flatten(), marker=mark, color=COLOR_LIST[i], markersize=MKS, linewidth=LINEWIDTH, alpha=ALPHA)
                if MARKER_OVERRIDE is not None:
                    for j, x in enumerate(x_list[i]):
                        ax = pylab.gca()
                        ax.annotate(MARKER_OVERRIDE, xy=(x_list[i][j]+corr, y_list[i][j]+corr), color=COLOR_LIST[i])
    else:
        for i in range(len(y_list)):
            mark = next(marker) if MARKER_OVERRIDE is None else None
            if x_list:
                pylab.plot(x_list[i].flatten(), y_list[i].flatten(), marker=mark, markersize=MKS, linewidth=LINEWIDTH, alpha=ALPHA)
                if MARKER_OVERRIDE is not None:
                    for j, x in enumerate(x_list[i]):
                        ax = pylab.gca()
                        ax.annotate(MARKER_OVERRIDE, xy=(x_list[i][j]+corr, y_list[i][j]+corr))
            else:
                pylab.plot(y_list[i].flatten(), marker=mark, markersize=MKS, linewidth=LINEWIDTH, alpha=ALPHA)
                if MARKER_OVERRIDE is not None:
                    for j, x in enumerate(x_list[i]):
                        ax = pylab.gca()
                        ax.annotate(MARKER_OVERRIDE, xy=(x_list[i][j]+corr, y_list[i][j]+corr))
    pylab.xlabel(XLABEL, fontsize=FONTSIZE)
    pylab.ylabel(YLABEL, fontsize=FONTSIZE)
    pylab.title(TITLE, fontsize=FONTSIZE)
    if MARKER_SCALE is not None:
        l = pylab.legend(LEGEND_LIST, markerscale=MARKER_SCALE, fontsize=LEGEND_FONTSIZE)
    else:
        l = pylab.legend(LEGEND_LIST, fontsize=LEGEND_FONTSIZE)
    if COLOR_LEGEND is not None:
        for i, text in enumerate(l.get_texts()):
            text.set_color(COLOR_LEGEND[i])
    pylab.tight_layout()
def plot_semilogx(x, y, XLABEL='', YLABEL='', TITLE='', SAVE_NAME='', **kwargs):
    pylab.figure(figsize=(SCALE*6.4, SCALE*4.8))
    pylab.semilogx(x.flatten(), y.flatten(), '-b')
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.tight_layout()
    pylab.savefig(SAVE_NAME+".png")
    pylab.close()
def plot_semilogx_subplot(x, y, XLABEL='', YLABEL='', TITLE='', **kwargs):
    pylab.semilogx(x.flatten(), y.flatten(), '-b')
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.tight_layout()
def plot_semilogx_overlay_subplot(x_list, y_list, XLABEL='', YLABEL='', TITLE='', LEGEND_LIST=[], **kwargs):
    marker = itertools.cycle(('>', '<', '+', '.', 'o', '*'))    # RESET markers
    for i in range(len(x_list)):
        pylab.semilogx(x_list[i].flatten(), y_list[i].flatten(), marker=next(marker))
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.legend(LEGEND_LIST)
    pylab.tight_layout()
def plot_semilogx_overlay(x_list, y_list, XLABEL='', YLABEL='', TITLE='', LEGEND_LIST=[], SAVE_NAME='', **kwargs):
    pylab.figure(figsize=(SCALE*6.4, SCALE*4.8))
    marker = itertools.cycle(('>', '<', '+', '.', 'o', '*'))  # RESET markers
    for i in range(len(x_list)):
        pylab.semilogx(x_list[i].flatten(), y_list[i].flatten(), marker=next(marker))
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.legend(LEGEND_LIST)
    pylab.tight_layout()
    pylab.savefig(SAVE_NAME + ".png")
    pylab.close()
def plot_error_bars_overlay(x_list, y_list, s_list, XLABEL='', YLABEL='', TITLE='', LEGEND_LIST=[], SAVE_NAME='', **kwargs):
    #pylab.figure(figsize=(SCALE*6.4, SCALE*4.8))
    pylab.figure(figsize=(SCALE*8.6, SCALE*4.8))
    marker = itertools.cycle(('>', '<', '+', '.', 'o', '*'))  # RESET markers
    for i in range(len(x_list)):
        pylab.plot(x_list[i].flatten(), y_list[i].flatten(), marker=next(marker))
        pylab.fill_between(x_list[i], y_list[i] - s_list[i], y_list[i] + s_list[i], alpha=0.2)
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.legend(LEGEND_LIST)
    pylab.tight_layout()
    pylab.savefig(SAVE_NAME + ".png")
    pylab.close()
def plot_error_bar_comparison(y_list, s_list, XLABEL='', YLABEL='', TITLE='', LEGEND_LIST=[], SAVE_NAME='', **kwargs):
    #pylab.figure(figsize=(SCALE*6.4, SCALE*4.8))
    pylab.figure(figsize=(SCALE*8.6, SCALE*4.8))
    marker = itertools.cycle(('>', '<', '+', '.', 'o', '*'))  # RESET markers
    w = 0.5
    for i in range(len(y_list)):
        print(i)
        mu = np.array([y_list[i], y_list[i]])
        s = np.array([s_list[i], s_list[i]])
        pylab.plot(np.array([i-w, i+w]), mu)
        pylab.fill_between(i, mu-s, mu+s, alpha=0.2)
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.legend(LEGEND_LIST)
    pylab.tight_layout()
    pylab.savefig(SAVE_NAME + ".png")
    pylab.close()
def plot_violin_plot(y_list, XLABEL='', YLABEL='', TITLE='', LEGEND_LIST=[], SAVE_NAME='', **kwargs):
    # Create the boxplot
    positions = np.arange(len(y_list))
    alpha = 0.3
    width = 0.15
    width = 0.4
    SCALE = 1.4
    fig = pylab.figure(figsize=(SCALE * 6.4, SCALE * 4.8))
    ax = fig.add_subplot(111)
    #vp = ax.violinplot(y_list, positions=np.arange(len(y_list)), widths=width, showmeans=True, showextrema=False)
    #vp = ax.violinplot(y_list, positions=positions, widths=width, showmeans=True, showextrema=True)
    vp = ax.violinplot(y_list, positions=positions, widths=width, showmedians=False, showmeans=False, showextrema=True)

    # colors = ['g', 'b', 'r', 'k']
    # for i, pc in enumerate(vp['bodies']):
    #     pc.set_facecolor(colors[i])
    #     pc.set_edgecolor(colors[i])
    #     pc.set_alpha(alpha)

    #vp['cmeans'].set_color('black')
    # vp['cextrema'].set_color('black')

    if len(XLABEL) == positions.size:
        #plt.xticks(positions, XLABEL, rotation='vertical')
        plt.xticks(positions, XLABEL)
        #plt.margins(0.2)
        #plt.subplots_adjust(bottom=0.15)
    else:
        pylab.xlabel(XLABEL)


    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.tight_layout()
    pylab.savefig(SAVE_NAME + ".png")
    pylab.close()
def plot_violin_subplot(y_list, XLABEL='', YLABEL='', TITLE='', LEGEND_LIST=[], SAVE_NAME='', **kwargs):
    # Create the boxplot
    positions = np.arange(len(y_list))
    alpha = 0.3
    width = 0.15
    width = 0.4
    SCALE = 1.4
    fig = pylab.figure(figsize=(SCALE * 6.4, SCALE * 4.8))
    ax = fig.add_subplot(111)
    #vp = ax.violinplot(y_list, positions=np.arange(len(y_list)), widths=width, showmeans=True, showextrema=False)
    #vp = ax.violinplot(y_list, positions=positions, widths=width, showmeans=True, showextrema=True)
    vp = ax.violinplot(y_list, positions=positions, widths=width, showmedians=False, showmeans=False, showextrema=True)

    # colors = ['g', 'b', 'r', 'k']
    # for i, pc in enumerate(vp['bodies']):
    #     pc.set_facecolor(colors[i])
    #     pc.set_edgecolor(colors[i])
    #     pc.set_alpha(alpha)

    #vp['cmeans'].set_color('black')
    # vp['cextrema'].set_color('black')

    if len(XLABEL) == positions.size:
        #plt.xticks(positions, XLABEL, rotation='vertical')
        plt.xticks(positions, XLABEL)
        #plt.margins(0.2)
        #plt.subplots_adjust(bottom=0.15)
    else:
        pylab.xlabel(XLABEL)

    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.tight_layout()
def plot_semilogx_error_bars_overlay(x_list, y_list, s_list, XLABEL='', YLABEL='', TITLE='', LEGEND_LIST=[], SAVE_NAME='', **kwargs):
    pylab.figure(figsize=(SCALE*6.4, SCALE*4.8))
    #pylab.figure(figsize=(SCALE*8.6, SCALE*4.8))
    marker = itertools.cycle(('>', '<', '+', '.', 'o', '*'))  # RESET markers
    for i in range(len(x_list)):
        pylab.semilogx(x_list[i].flatten(), y_list[i].flatten(), marker=next(marker))
        pylab.fill_between(x_list[i], y_list[i] - s_list[i], y_list[i] + s_list[i], alpha=0.2)
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.legend(LEGEND_LIST)
    pylab.tight_layout()
    pylab.savefig(SAVE_NAME + ".png")
    pylab.close()
def plot_semilogx_error_bars_overlay_subplot(x_list, y_list, s_list, XLABEL='', YLABEL='', TITLE='', LEGEND_LIST=[], **kwargs):
    #pylab.figure(figsize=(SCALE*6.4, SCALE*4.8))
    marker = itertools.cycle(('>', '<', '+', '.', 'o', '*'))  # RESET markers
    for i in range(len(x_list)):
        pylab.semilogx(x_list[i].flatten(), y_list[i].flatten(), marker=next(marker))
        pylab.fill_between(x_list[i], y_list[i] - s_list[i], y_list[i] + s_list[i], alpha=0.2)
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.legend(LEGEND_LIST)
    pylab.tight_layout()
def plot_semilogy(x, y, XLABEL='', YLABEL='', TITLE='', SAVE_NAME='', **kwargs):
    pylab.figure(figsize=(SCALE*6.4, SCALE*4.8))
    pylab.semilogy(x.flatten(), y.flatten(), '-b')
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.tight_layout()
    pylab.savefig(SAVE_NAME+".png")
    pylab.close()
def plot_semilogy_subplot(x, y, XLABEL='', YLABEL='', TITLE='', **kwargs):
    pylab.semilogy(x.flatten(), y.flatten(), '-b')
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.tight_layout()
def plot_semilogy_overlay(x_list, y_list, XLABEL='', YLABEL='', TITLE='', LEGEND_LIST=[], SAVE_NAME='', **kwargs):
    pylab.figure(figsize=(SCALE*6.4, SCALE*4.8))
    marker = itertools.cycle(('>', '<', '+', '.', 'o', '*'))  # RESET markers
    for i in range(len(x_list)):
        if x_list[i] == None:
            pylab.semilogy(y_list[i].flatten(), marker=next(marker))
        else:
            pylab.semilogy(x_list[i].flatten(), y_list[i].flatten(), marker=next(marker))
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.legend(LEGEND_LIST)
    pylab.tight_layout()
    pylab.savefig(SAVE_NAME + ".png")
    pylab.close()
def plot_loglog(x, y, XLABEL='', YLABEL='', TITLE='', SAVE_NAME='', **kwargs):
    pylab.figure(figsize=(SCALE*6.4, SCALE*4.8))
    pylab.loglog(x.flatten(), y.flatten(), '-b')
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.tight_layout()
    pylab.savefig(SAVE_NAME+".png")
    pylab.close()
def plot_loglog_overlay(x_list, y_list, XLABEL='', YLABEL='', TITLE='', LEGEND_LIST=[], SAVE_NAME='', **kwargs):
    pylab.figure(figsize=(SCALE*6.4, SCALE*4.8))
    marker = itertools.cycle(('>', '<', '+', '.', 'o', '*'))  # RESET markers
    for i in range(len(x_list)):
        pylab.loglog(x_list[i].flatten(), y_list[i].flatten(), marker=next(marker))
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.legend(LEGEND_LIST)
    pylab.tight_layout()
    pylab.savefig(SAVE_NAME + ".png")
    pylab.close()
def plot_loglog_overlay_subplot(x_list, y_list, XLABEL='', YLABEL='', TITLE='', LEGEND_LIST=[], SAVE_NAME='', **kwargs):
    marker = itertools.cycle(('>', '<', '+', '.', 'o', '*'))  # RESET markers
    for i in range(len(x_list)):
        pylab.loglog(x_list[i].flatten(), y_list[i].flatten(), marker=next(marker))
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.legend(LEGEND_LIST, fontsize=12)
    pylab.tight_layout()
def plot_pdf(x, bins=None, XLABEL='', YLABEL='', TITLE='', SAVE_NAME='', alpha=0.75, density=True, weights=None, **kwargs):
    #bins = int(np.sqrt(len(h.flatten().tolist()))) if bins == None else bins
    pylab.figure(figsize=(SCALE*6.4, SCALE*4.8))
    pylab.hist(x.flatten(), bins=bins, density=density, weights=weights, facecolor='g', alpha=alpha)
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.tight_layout()
    pylab.savefig(SAVE_NAME+".png")
    pylab.close()
def plot_pdf_subplot(x, bins=None, XLABEL='', YLABEL='', TITLE='', alpha=0.75, density=True, weights=None, **kwargs):
    #bins = int(np.sqrt(len(h.flatten().tolist()))) if bins == None else bins
    pylab.hist(x.flatten(), bins=bins, density=density, weights=weights, facecolor='g', alpha=alpha)
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.tight_layout()
def plot_pdf_overlay(x_list, bins=None, XLABEL='', YLABEL='', TITLE='', LEGEND_LIST=[], SAVE_NAME='', alpha=0.6, density=True, weights=None, **kwargs):
    #bins = int(np.sqrt(len(x_list[0].flatten().tolist()))) if bins == None else int(bins)
    pylab.figure(figsize=(SCALE*6.4, SCALE*4.8))
    for i in range(len(x_list)):
        pylab.hist(x_list[i].flatten(), bins=bins, density=density, alpha=alpha, weights=weights, label=LEGEND_LIST[i])
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.legend(LEGEND_LIST)
    pylab.tight_layout()
    pylab.savefig(SAVE_NAME+".png")
    pylab.close()
def plot_pdf_overlay_subplot(x_list, bins=None, XLABEL='', YLABEL='', TITLE='', LEGEND_LIST=[], alpha=0.6, density=True, weights=None, **kwargs):
    #bins = int(np.sqrt(len(x_list[0].flatten().tolist()))) if bins == None else int(bins)
    for i in range(len(x_list)):
        pylab.hist(x_list[i].flatten(), bins=bins, density=density, alpha=alpha, weights=weights, label=LEGEND_LIST[i])
    pylab.xlabel(XLABEL)
    pylab.ylabel(YLABEL)
    pylab.title(TITLE)
    pylab.legend(LEGEND_LIST)
    pylab.tight_layout()