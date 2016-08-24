CURRENT_GENERATION = 0
MAX_GENERATIONS = 10
POPULATION = 20
SELECT = 5
INPUT_DATA = None
MIN_FILTER_SIZE = 3
MAX_FILTER_SIZE = 16
MAX_FILTERS = 64
MIN_FILTERS = 16
LEARNING_RATE = 0.05
NUM_CLASSES = 2
MIN_DENSE_LAYER_SIZE = 64
MAX_DENSE_LAYER_SIZE = 512
IMAGE_SHAPE = [32, 32, 3] # H, W, C

DATASET = None


DIR = 'trounament'

#actions
APPEND_CONVOLUTIONAL_LAYER = 0
APPEND_DENSE_LAYER = 1
REMOVE_CONVOLUTIONAL_LAYER = 2
REMOVE_DENSE_LAYER = 3
KEEP = 4
INSERT_CONVOLUTIONAL_LAYER = 5
INSERT_DENSE_LAYER = 6
CROSS_MODELS = 7




# CRAZINESS how much randomness is allowed to happen, reusing convs from earlier layers

# ZOMBIE : bring back models that were mutated away from earier generations