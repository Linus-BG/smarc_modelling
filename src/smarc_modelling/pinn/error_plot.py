#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from smarc_modelling.vehicles.SAM import SAM
from smarc_modelling.vehicles.SAM_PINN import SAM_PINN
import pandas as pd

dt = 0.01
sam = SAM(dt)
sam_pinn = SAM_PINN(dt)

