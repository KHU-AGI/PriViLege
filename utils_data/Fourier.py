import math

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch

def fourier(x):  # 2D Fourier transform
    f = torch.fft.fft2(x)
    f = f.abs() + 1e-6
    f = f.log()
    return f


def shift(x):  # shift Fourier transformed feature map
    b, c, h, w = x.shape
    return torch.roll(x, shifts=(int(h/2), int(w/2)), dims=(2, 3))


def make_segments(x, y):  # make segment for `plot_segment`
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def plot_segment(ax, xs, ys, cmap_name="plasma"):  # plot with cmap segments
    z = np.linspace(0.0, 1.0, len(ys))
    z = np.asarray(z)
    
    cmap = cm.get_cmap(cmap_name)
    norm = plt.Normalize(0.0, 1.0)
    segments = make_segments(xs, ys)
    lc = LineCollection(segments, array=z, cmap=cmap_name, norm=norm,
                        linewidth=2.5, alpha=1.0)
    ax.add_collection(lc)

    colors = [cmap(x) for x in xs]
    ax.scatter(xs, ys, color=colors, marker="s", zorder=100)
    

# # Fourier transform feature maps
# fourier_latents = []
# for latent in latents:  # `latents` is a list of hidden feature maps in latent spaces
#     latent = latent.cpu()
    
#     if len(latent.shape) == 3:  # for ViT
#         b, n, c = latent.shape
#         h, w = int(math.sqrt(n)), int(math.sqrt(n))
#         latent = rearrange(latent, "b (h w) c -> b c h w", h=h, w=w)
#     elif len(latent.shape) == 4:  # for CNN
#         b, c, h, w = latent.shape
#     else:
#         raise Exception("shape: %s" % str(latent.shape))
#     latent = fourier(latent)
#     latent = shift(latent).mean(dim=(0, 1))
#     latent = latent.diag()[int(h/2):]  # only use the half-diagonal components
#     latent = latent - latent[0]  # visualize 'relative' log amplitudes 
#                                  # (i.e., low-freq amp - high freq amp)
#     fourier_latents.append(latent)
    

# # A. Plot Fig 2a: "Relative log amplitudes of Fourier transformed feature maps"
# fig, ax1 = plt.subplots(1, 1, figsize=(3.3, 4), dpi=150)
# for i, latent in enumerate(reversed(fourier_latents[:-1])):
#     freq = np.linspace(0, 1, len(latent))
#     ax1.plot(freq, latent, color=cm.plasma_r(i / len(fourier_latents)))
    
# ax1.set_xlim(left=0, right=1)

# ax1.set_xlabel("Frequency")
# ax1.set_ylabel("$\Delta$ Log amplitude")

# from matplotlib.ticker import FormatStrFormatter
# ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1fπ'))


# # B. Plot Fig 8: "Relative log amplitudes of high-frequency feature maps"
# if name == "resnet50":  # for ResNet-50
#     pools = [4, 8, 14]
#     msas = []
#     marker = "D"
# elif name == "vit_tiny_patch16_224":  # for ViT-Ti
#     pools = []
#     msas = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23,]
#     marker = "o"
# else:
#     import warnings
#     warnings.warn("The configuration for %s are not implemented." % name, Warning)
#     pools, msas = [], []
#     marker = "s"

# depths = range(len(fourier_latents))

# # Normalize
# depth = len(depths) - 1
# depths = (np.array(depths)) / depth
# pools = (np.array(pools)) / depth
# msas = (np.array(msas)) / depth

# fig, ax2 = plt.subplots(1, 1, figsize=(6.5, 4), dpi=120)
# plot_segment(ax2, depths, [latent[-1] for latent in fourier_latents])  # high-frequency component

# for pool in pools:
#     ax2.axvspan(pool - 1.0 / depth, pool + 0.0 / depth, color="tab:blue", alpha=0.15, lw=0)
# for msa in msas:
#     ax2.axvspan(msa - 1.0 / depth, msa + 0.0 / depth, color="tab:gray", alpha=0.15, lw=0)
    
# ax2.set_xlabel("Normalized depth")
# ax2.set_ylabel("$\Delta$ Log amplitude")
# ax2.set_xlim(0.0, 1.0)

# from matplotlib.ticker import FormatStrFormatter
# ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# plt.show()