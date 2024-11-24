import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Embedded data
data_frechet = {
    "shifted_data_proportion": [0, 0.25, 0.50, 0.75, 1.00],
    "frechet_distance": [0.1259, 0.1284, 0.1492, 0.1846, 0.2491]
}

data_tpr_1 = {
    "shifted_data_proportion": [0, 0.25, 0.50, 0.75, 1.00],
    "secmi": [0.0155, 0.0280, 0.0692, 0.1140, 0.2888],
    "pia": [0.0232, 0.0296, 0.0652, 0.1004, 0.3120]
}

data_tpr_01 = {
    "shifted_data_proportion": [0, 0.25, 0.50, 0.75, 1.00],
    "secmi": [0, 0, 0.0040, 0.0092, 0.1364],
    "pia": [0, 0, 0.0044, 0.0088, 0.1776]
}

data_auc = {
    "shifted_data_proportion": [0, 0.25, 0.50, 0.75, 1.00],
    "secmi": [0.6600, 0.6706, 0.6828, 0.6919, 0.6991],
    "pia": [0.6689, 0.6894, 0.7225, 0.7400, 0.7617]
}

# Create figure with GridSpec
fig = plt.figure(figsize=(35, 4.5))
gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1])
gs.update(wspace=0.3, left=0.06, right=0.99, top=0.92, bottom=0.15)

# Parameters for increased size and visibility
line_width = 4
marker_size = 12
title_fontsize = 20
label_fontsize = 18
tick_labelsize = 16
x_ticks = [0, 0.5, 1.0]

# Define colors
secmi_color = '#90EE90'  # swallow color (light blue)
pia_color = 'gray'
frechet_color = '#FFCCCB'

# Plot for Fréchet Distance
ax0 = plt.subplot(gs[0])
ax0.plot(
    data_frechet["shifted_data_proportion"], data_frechet["frechet_distance"], 
    marker='o', markersize=marker_size, linewidth=line_width, 
    label="Fréchet Distance", color=frechet_color
)
ax0.set_title("Fréchet Distance", fontsize=title_fontsize)
ax0.set_xlabel("Shifted Data Proportion", fontsize=label_fontsize)
ax0.set_ylabel("Fréchet Distance", fontsize=label_fontsize)
ax0.tick_params(labelsize=tick_labelsize, width=2, length=8)
ax0.set_xticks(x_ticks)
ax0.yaxis.set_major_locator(plt.MaxNLocator(4))
ax0.grid(visible=False)
ax0.legend(fontsize=label_fontsize)

# Plot for TPR @ 1% FPR
ax1 = plt.subplot(gs[1])
ax1.plot(
    data_tpr_1["shifted_data_proportion"], data_tpr_1["secmi"], 
    marker='o', markersize=marker_size, linewidth=line_width, 
    label="SecMI", color=secmi_color
)
ax1.plot(
    data_tpr_1["shifted_data_proportion"], data_tpr_1["pia"], 
    marker='o', markersize=marker_size, linewidth=line_width, 
    label="PIA", color=pia_color
)
ax1.set_title("TPR @ 1% FPR", fontsize=title_fontsize)
ax1.set_xlabel("Shifted Data Proportion", fontsize=label_fontsize)
ax1.set_ylabel("TPR @ 1% FPR", fontsize=label_fontsize)
ax1.tick_params(labelsize=tick_labelsize, width=2, length=8)
ax1.set_xticks(x_ticks)
ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
ax1.grid(visible=False)
ax1.legend(fontsize=label_fontsize)

# Plot for TPR @ 0.1% FPR
ax2 = plt.subplot(gs[2])
ax2.plot(
    data_tpr_01["shifted_data_proportion"], data_tpr_01["secmi"], 
    marker='o', markersize=marker_size, linewidth=line_width, 
    label="SecMI", color=secmi_color
)
ax2.plot(
    data_tpr_01["shifted_data_proportion"], data_tpr_01["pia"], 
    marker='o', markersize=marker_size, linewidth=line_width, 
    label="PIA", color=pia_color
)
ax2.set_title("TPR @ 0.1% FPR", fontsize=title_fontsize)
ax2.set_xlabel("Shifted Data Proportion", fontsize=label_fontsize)
ax2.set_ylabel("TPR @ 0.1% FPR", fontsize=label_fontsize)
ax2.tick_params(labelsize=tick_labelsize, width=2, length=8)
ax2.set_xticks(x_ticks)
ax2.yaxis.set_major_locator(plt.MaxNLocator(4))
ax2.grid(visible=False)
ax2.legend(fontsize=label_fontsize)

# Plot for AUC
ax3 = plt.subplot(gs[3])
ax3.plot(
    data_auc["shifted_data_proportion"], data_auc["secmi"], 
    marker='o', markersize=marker_size, linewidth=line_width, 
    label="SecMI", color=secmi_color
)
ax3.plot(
    data_auc["shifted_data_proportion"], data_auc["pia"], 
    marker='o', markersize=marker_size, linewidth=line_width, 
    label="PIA", color=pia_color
)
ax3.set_title("AUC", fontsize=title_fontsize)
ax3.set_xlabel("Shifted Data Proportion", fontsize=label_fontsize)
ax3.set_ylabel("AUC", fontsize=label_fontsize)
ax3.tick_params(labelsize=tick_labelsize, width=2, length=8)
ax3.set_xticks(x_ticks)
ax3.yaxis.set_major_locator(plt.MaxNLocator(4))
ax3.grid(visible=False)
ax3.legend(fontsize=label_fontsize)

plt.show()
plt.savefig('./img.png')