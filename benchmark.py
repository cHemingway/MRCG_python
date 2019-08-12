#!/usr/bin/env python
# Benchmark for MRCG, useful for profiling etc

# Chris Hemingway 2019, MIT License
# See LICENSE file for details

import timeit

# Quite slow to execute so use a low number of cycles
NUMBER = 10

DATA_SECONDS = 4
SAMPLE_RATE = 16000

# Setup code, generates 4 seconds of noise values -1 to 1
setup = \
    "import sys,os; sys.path.append(os.getcwd());"\
    "import numpy as np;"\
    "from MRCG import mrcg_extract;"\
    "data = (2 * np.random.rand({})) - 1".format(DATA_SECONDS*SAMPLE_RATE)

print("Benchmarking {} cycles".format(NUMBER))
time = timeit.timeit("mrcg_extract(data)", setup=setup, number=NUMBER) 
per_cycle = time/NUMBER
print("Took {:.3f} seconds for {:.3f} of audio".format(per_cycle, DATA_SECONDS))
print("{:.2f}x faster than realtime".format(DATA_SECONDS/per_cycle))
                