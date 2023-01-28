r"""Path constants shared by all files.

Usage:
    import fine_tune

    fine_tune.path.DATA
    fine_tune.path.DOC
    fine_tune.path.FINE_TUNE_DATA
    fine_tune.path.FINE_TUNE_EXPERIMENT
    fine_tune.path.LOG
    fine_tune.path.PROJECT_ROOT
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

# Project folder absolute path.

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.abspath(__file__),
    os.pardir,
    os.pardir
))

# Data folder absolute path.

DATA = os.path.join(
    PROJECT_ROOT,
    'data'
)

# Create data folder if not exists.

if not os.path.exists(DATA):
    os.makedirs(DATA)

# Document folder absolute path.

DOC = os.path.join(
    PROJECT_ROOT,
    'doc'
)

# Fine tune data folder absolute path.

FINE_TUNE_DATA = os.path.join(
    DATA,
    'fine_tune'
)

# Create fine tune data folder if not exists.

if not os.path.exists(FINE_TUNE_DATA):
    os.makedirs(FINE_TUNE_DATA)

# Fine tune experiment folder absolute path.

FINE_TUNE_EXPERIMENT = os.path.join(
    DATA,
    'fine_tune_experiment'
)

# Create fine tune experiment folder if not exists.

if not os.path.exists(FINE_TUNE_EXPERIMENT):
    os.makedirs(FINE_TUNE_EXPERIMENT)

# Fine tune experiment log folder absolute path.

LOG = os.path.join(
    FINE_TUNE_EXPERIMENT,
    'log'
)

# Create fine tune experiment log folder if not exists.

if not os.path.exists(LOG):
    os.makedirs(LOG)
