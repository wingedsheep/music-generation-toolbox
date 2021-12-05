import numpy as np

DRUM_INSTRUMENT = 128

DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32 + 1, dtype=np.int)
DEFAULT_FRACTION = 16
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

# parameters for outputItem
DEFAULT_RESOLUTION = 480