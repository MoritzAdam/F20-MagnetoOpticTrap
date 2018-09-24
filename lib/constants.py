import os
import numpy as np
import scipy.constants as sc

data_path = '../data'
spectroscopy_path = os.path.join(data_path, 'part1_spectroscopy')
loading_path = os.path.join(data_path, 'part3_loading')
temperature_path = os.path.join(data_path, 'part4_release_recapture')

evaluation_path = '../eval'
save_loading_path = os.path.join(evaluation_path, 'loading.xlsx')
save_rr_path = os.path.join(evaluation_path, 'release_recapture.xlsx')
save_finestructure_path = os.path.join(evaluation_path, 'finestructure_fit.xlsx')
save_multiplet_path = os.path.join(evaluation_path, 'multiplet_separation.xlsx')
save_hyperfinestructure_path = os.path.join(evaluation_path, 'hyperfinestructure_fit.xlsx')
save_hyperfinestructure_zoom_path = os.path.join(evaluation_path, 'hyperfinestructure_zoom_fit.xlsx')
save_crossings = os.path.join(data_path, 'crossings.xlsx')

PLOT_MARKERSIZE = 2
PLOT_TITLE_SIZE = 14
PLOT_AX_LABEL_SIZE = 14
PLOT_AX_TICK_SIZE = 12
PLOT_ANNOTATION_TEXT_SIZE = 12
MOT_RADIUS = 1.5 * 1e-3  # [m]
RB85_MASS = 84.911789738 * 1.66054e-27  # [kg]
RB87_MASS = 86.909180527 * 1.66054e-27  # [kg]
P_POWERMETER = 54  # [mW]
GAMMA = 2 * np.pi * 6.07 * 1e6  # [Hz]
INTENSITY_SAT = 4.1  # [mW/cm^2]
INTENSITY = 2 * P_POWERMETER / (np.pi * 0.2 ** 2)  # [mW/cm^2]
LASER_LENGTH = 780.241 * 1e-9  # [nm]
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

# constants for spectroscopy
MASK_NAME = 'Masked - '
RB87_FREQ_SEP_THEORY_F1_F2 = 6.834682610904290  # (90) [GHz]
K_BOLTZMANN = sc.k
C = sc.c
FIT_PLOT_BARRIER = 100
RB87_NU0 = 384230.4844685  # (62) GHz
RB85_NU0 = 384230.406373  # (14) GHz

GLOBAL_ZOOM_FOR_HYPERFINE = [(460, -400),
                             (450, -370),
                             (400, -380),
                             (290, -390)]

START_TOKEN = 'START_TOKEN'
STOP_TOKEN = 'STOP_TOKEN'
# RB_FINESTRUCTURE_THEORY = {'85f2': }
