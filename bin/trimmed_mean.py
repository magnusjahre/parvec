#!/usr/bin/python
from scipy import stats
import sys

#print 'Number of arguments:', len(sys.argv), 'arguments.'
#print 'Argument List:', str(sys.argv[1:])

try:
    x = [float(arg) for arg in sys.argv[1:]]
    total = stats.trim_mean(x, 0.3)
    print total
except ValueError:
    print 'Please supply integer arguments'
