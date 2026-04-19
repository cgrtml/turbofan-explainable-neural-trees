"""Generate updated robustness figures with LSTM comparison."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Results from b2u3eplw5 run
noise_sigmas = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]
noise_nt   = [17.13, 17.44, 19.22, 26.16, 32.99, 38.16]
noise_lstm = [16.10, 15.91, 16.46, 19.77, 24.27, 28.69]
noise_rf   = [18.07, 18.63, 20.20, 25.57, 30.48, 35.10]

miss_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
miss_nt   = [17.13, 17.37, 20.86, 19.31, 25.50, 42.45]
miss_lstm = [16.10, 21.92, 23.72, 34.94, 25.14, 61.53]
miss_rf   = [18.07, 17.96, 19.51, 28.55, 27.76, 26.89]

os.makedirs('figures', exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
})

STYLES = {
    'NT (ours)':      ('black',   'o', '-'),
    'LSTM':           ('dimgray', 's', '--'),
    'Random Forest':  ('gray',    '^', ':'),
}

# Figure: noise robustness
fig, ax = plt.subplots(figsize=(3.5, 2.6))
ax.plot(noise_sigmas, noise_nt,   color='black',   marker='o', ls='-',  label='NT (ours)')
ax.plot(noise_sigmas, noise_lstm, color='dimgray', marker='s', ls='--', label='LSTM')
ax.plot(noise_sigmas, noise_rf,   color='gray',    marker='^', ls=':',  label='Random Forest')
ax.set_xlabel('Gaussian noise std $\\sigma$')
ax.set_ylabel('RMSE (cycles)')
ax.set_title('Noise Robustness — FD001')
ax.legend(loc='upper left')
ax.set_xlim(-0.01, 0.21)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.annotate('LSTM: temporal\naveraging helps', xy=(0.20, 28.69), fontsize=7,
            xytext=(0.13, 22), arrowprops=dict(arrowstyle='->', color='dimgray', lw=0.8))
plt.tight_layout()
plt.savefig('figures/fig3_noise_robustness.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig3_noise_robustness.png")

# Figure: missing sensor robustness
fig, ax = plt.subplots(figsize=(3.5, 2.6))
ax.plot([r*100 for r in miss_ratios], miss_nt,   color='black',   marker='o', ls='-',  label='NT (ours)')
ax.plot([r*100 for r in miss_ratios], miss_lstm, color='dimgray', marker='s', ls='--', label='LSTM')
ax.plot([r*100 for r in miss_ratios], miss_rf,   color='gray',    marker='^', ls=':',  label='Random Forest')
ax.set_xlabel('Missing sensor channels (%)')
ax.set_ylabel('RMSE (cycles)')
ax.set_title('Missing Sensor Robustness — FD001')
ax.legend(loc='upper left')
ax.set_xlim(-2, 52)
ax.grid(True, alpha=0.3, linewidth=0.5)

# Annotate the 30% crossover point
ax.annotate('LSTM: +18.8\nNT: +2.2\n(at 30%)', xy=(30, 34.94), fontsize=7,
            xytext=(35, 45), arrowprops=dict(arrowstyle='->', color='dimgray', lw=0.8))

plt.tight_layout()
plt.savefig('figures/fig4_missing_sensor_robustness.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig4_missing_sensor_robustness.png")
