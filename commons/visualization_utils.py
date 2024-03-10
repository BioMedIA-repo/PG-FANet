import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib import colors
from sklearn.manifold import TSNE
from commons.utils import *
from PIL import Image
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from matplotlib.pyplot import MultipleLocator
import matplotlib as mpl


def colorize_mask(mask):
    palette = [128, 128, 128, 128, 0, 0, 192, 192, 128, 128, 64, 128, 0, 0, 192, 128, 128, 0, 192, 128, 128, 64, 64,
               128,
               64, 0, 128, 64, 64, 0, 0, 128, 192, 0, 0, 0]
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def plot_mask(axis_in, img_in, mask_in, title_in):
    mask_colors = ['k', 'r', 'b', 'g', 'y', 'c', 'm']
    img = img_in.copy()
    axis_in.clear()
    if mask_in.shape[2] > 1:
        mask_max = np.argmax(mask_in, axis=2)
        for mask_idx in range(1, mask_in.shape[2]):
            img[mask_max == mask_idx, :] = np.round(
                np.asarray(colors.colorConverter.to_rgb(mask_colors[mask_idx])) * 255)
    else:
        img[mask_in[:, :, 0] > 0.5, :] = np.round(np.asarray(colors.colorConverter.to_rgb('y')) * 255)

    axis_in.imshow(img)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    font_weight = 'bold'
    font_color = 'k'

    fontdict = {'family': 'serif',
                'color': font_color,
                'weight': font_weight,
                'size': 14,
                }

    # place a text box in upper left in axes coords
    axis_in.text(0.35, 0.95, title_in, transform=axis_in.transAxes, fontdict=fontdict, verticalalignment='top',
                 bbox=props)


def t_sne_1(latent_vecs, fname='tsne.png'):
    latent_vecs_reduced = TSNE(n_components=2, random_state=0).fit_transform(latent_vecs)
    # latent_vecs_reduced = PCA(n_components=2).fit_transform(latent_vecs)
    plt.scatter(latent_vecs_reduced[:, 0], latent_vecs_reduced[:, 1],
                cmap='jet')
    plt.colorbar()
    plt.savefig(fname, transparent=True)
    plt.show()


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.pyplot import cm
import matplotlib.mlab as mlab
from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import time


def t_sne(latent_vecs, final_label, config, label_names=['Benign', 'Malignant'],
          num_classes=2):
    fname = "tsne_" + str(time.time()) + '.pdf'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = cm.Spectral(np.linspace(0, 1, num_classes))
    embeddings = TSNE(n_components=2, random_state=0).fit_transform(latent_vecs)
    xx = embeddings[:, 0]
    yy = embeddings[:, 1]

    # plot the images
    if False:
        for i, (x, y) in enumerate(zip(xx, yy)):
            im = OffsetImage(X_sample[i], zoom=0.1, cmap='gray')
            ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
            ax.add_artist(ab)
        ax.update_datalim(np.column_stack([xx, yy]))
        ax.autoscale()

    # plot the 2D data points
    for i in range(num_classes):
        ax.scatter(xx[final_label == i], yy[final_label == i], color=colors[i], label=label_names[i], s=10)

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.set_xticks([])
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.set_yticks([])
    plt.axis('tight')
    # plt.xticks([])
    # plt.yticks([])
    plt.axis('off')
    plt.legend(loc='best', scatterpoints=1, fontsize=5)
    plt.savefig(config.tmp_dir + '/' + fname, format='pdf', dpi=600)
    # plt.show()


def visualize(data, filename):
    assert (len(data.shape) == 3)  # height*width*channels
    if data.shape[2] == 1:  # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))  # the image is already 0-255
    else:
        img = Image.fromarray((data * 255).astype(np.uint8))  # the image is between 0-1
    img.save(filename + '.png')
    return img


def plt_props():
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.style'] = 'normal'
    plt.rcParams['font.variant'] = 'normal'
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['figure.titlesize'] = 12
    plt.rcParams['figure.figsize'] = 6, 4
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 7



def plot_sns_line(data_frames, y_aix, y_label, title, data_set, dashes):
    sns.set(font='serif')
    sns.set_style("whitegrid")
    final_sns = pd.concat(data_frames, axis=0)
    fig, ax = plt.subplots(dpi=500, figsize=(8, 6))
    plt_props()
    ax = sns.lineplot(x="per", y="data", hue="Method", style='Method',
                      markers=False, dashes=dashes,
                      data=final_sns)
    # ax.lines[2].set_linestyle("--")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.f'))
    y_major_locator = MultipleLocator(5)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.set_xlim(9.8, 50.4)
    ax.set_ylim(y_aix[0], y_aix[1])
    x_major_locator = MultipleLocator(10)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.grid(visible=False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:], ncol=2, loc='lower right')
    # leg = ax.legend(handles=handles[1:], labels=labels[1:], ncol=2, loc='lower right')
    # leg_lines = leg.get_lines()
    # leg_lines[2].set_linestyle("--")
    fig.tight_layout()
    plt.xlabel('% of Labeled Data')
    plt.ylabel(y_label)
    plt.title(title + ' performance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(join('../log/', title + "_line_seg_" + data_set + ".jpg"))
    plt.close()


def plot_error_bar(res_stat, y_aix, y_label, title, m, data_set, nn=3):
    sns.set(font='serif')
    sns.set_style("whitegrid")
    if nn == 3:
        isic_ratios = [100. * item for item in [0.05, 0.10, 0.15]]
    if nn == 6:
        isic_ratios = [100. * item for item in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
    if nn == 9:
        isic_ratios = [100. * item for item in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]]
    fig, ax = plt.subplots(dpi=500, figsize=(8, 6))
    plt_props()
    for key in m.keys():
        print(key)
        if key == 'Full':
            ax.errorbar(isic_ratios, np.max(np.array(res_stat[key]) * 100, axis=0), label=key, marker=m[key],
                        yerr=0, capsize=2, linestyle='--')
            # ax.errorbar(isic_ratios, np.mean(np.array(res_stat[key]) * 100, axis=0), label=key, marker=m[key],
            #             yerr=np.std(np.array(res_stat[key]) * 100, axis=0), capsize=2, linestyle='--')
        else:
            ax.errorbar(isic_ratios, np.mean(np.array(res_stat[key]) * 100, axis=0), label=key, marker=m[key],
                        yerr=np.std(np.array(res_stat[key]) * 100, axis=0), capsize=2)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.f'))
    y_major_locator = MultipleLocator(5)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.set_xlim(isic_ratios[0]-0.5, isic_ratios[-1]+0.5)
    ax.set_ylim(y_aix[0], y_aix[1])
    x_major_locator = MultipleLocator(10)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.legend(ncol=2, loc='lower right')
    ax.grid(visible=False)
    plt.xlabel('% of Labeled Data')
    plt.ylabel(y_label)
    plt.title(title + ' performance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(join('../log/', title + "_error_seg_" + data_set + ".jpg"))
    plt.close()
