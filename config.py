"""
Configuration file for the correlation analysis pipeline.
"""

# Data paths
DATA_PATH = "/Users/groupies/Documents/NOVAProjects/Multichannel_ECoG_GAN-main/Version2/global_norm_3s_512hz.pkl"
OUTPUT_DIR = "/Users/groupies/Documents/NOVAProjects/Multichannel_ECoG_GAN-main/Spatial_Matrix"

# Signal processing parameters
SAMPLE_RATE = 512  # Hz
ALPHA_BAND = (8.0, 12.0)  # Hz
FILTER_ORDER = 4

# Electrode configuration
ELECTRODE_GRID = [
    ["Gr16", "Gr11", "Gr6", "Gr1"],
    ["Gr17", "Gr12", "Gr7", "Gr2"],
    ["Gr18", "Gr13", "Gr8", "Gr3"],
    ["Gr19", "Gr14", "Gr9", "Gr4"],
    ["Gr20", "Gr15", "Gr10", "Gr5"]
]

CHANNEL_NAME_ORDER = [f'Gr{i}' for i in range(1, 21)]
CUSTOM_LABELS = [f'C{i}{j}' for i in range(1, 6) for j in range(1, 5)]

# Spacing parameters
ROW_SPACING = 1.0
COL_SPACING = 2.0

# Analysis parameters
MAX_WINDOWS = 20
SUBJECT_INDICES = [0, 1, 2]  # Analyze first 3 subjects
SAMPLE_WINDOWS = [0, 5, 10]  # Windows for PSD analysis

# Visualization parameters
FIGURE_SIZE_LARGE = (22, 9)
FIGURE_SIZE_MEDIUM = (12, 10)
FIGURE_SIZE_SMALL = (10, 7)
FIGURE_SIZE_PSD = (15, 12)

# Plotting parameters
HEATMAP_CMAP = 'coolwarm'
HEATMAP_VMIN = -1
HEATMAP_VMAX = 1
CORRELATION_Y_LIM = (-1.1, 1.1)

# File naming patterns
FILE_PATTERNS = {
    'correlation_vs_distance': 'corr_vs_dist_subj_{subject}_window_{window}.png',
    'dual_correlation_heatmaps': 'dual_corr_heatmaps_subj_{subject}_window_{window}.png',
    'raw_correlation': 'raw_corr_subj_{subject}_window_{window}.png',
    'alpha_correlation': 'alpha_corr_subj_{subject}_window_{window}.png',
    'cross_window_correlation': 'cross_window_corr_window_{window}.png',
    'alpha_power_correlation': 'alpha_power_corr_subj_{subject}.png',
    'psd': 'psd_subj_{subject}_window_{window}.png'
}
