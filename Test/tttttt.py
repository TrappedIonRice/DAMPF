import utils
import quimb.tensor as qtn
import numpy as np
import Pure_QT_config as config

submpo = utils.construct_interaction_gates(config.nsites, config.nosc, config.localDim, )