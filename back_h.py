#!/usr/bin/env python
# python

import numpy as np


def repRelHeight(sw, tsz, blh):

    z = -blh / 5 * np.log(1 - (tsz * sw) / (0.15 * blh))

    return z
