import os
import numpy as np

data_path = '../data'
spectroscopy_path = os.path.join(data_path, 'part1_spectroscopy')
loading_path = os.path.join(data_path, 'part3_loading')
temperature_path = os.path.join(data_path, 'part4_release_recapture')

PLOT_MARKERSIZE = 2
RB87_FREQ_SEP_THEORY_F1_F2 = 6.834682610904290  # (90) [GHz]
MOT_RADIUS = 1.5*1e-3  # [m]
RB85_MASS = 84.911789738*1.66054e-27  # [kg]
P_POWERMETER = 54  # [mW]
GAMMA = 2 * np.pi * 6.07 * 1e6  # [Hz]
INTENSITY_SAT = 4.1  # [mW/cm^2]
INTENSITY = 2 * P_POWERMETER / (np.pi * 0.2 ** 2)  # [mW/cm^2]
LASER_LENGTH = 780.241*1e-9  # [Hz]
SOLID_ANGLE = np.pi * 25.4 ** 2 / (4 * np.pi * 150 ** 2)
QE_REL_ERROR = 0.029
G_REL_ERROR = 0.05
DETUNING_ERROR = 0.06
DETUNING_DICT = 'part3_detuning.txt'
DURATION_DICT = 'part4_duration.txt'
BLUE = 'C0'
RED = 'r'
GREEN = 'g'
T = 0.96
G = 4.75 * 1e+6
QE = 0.52
S = 1e6 / (1e6 + 50)
