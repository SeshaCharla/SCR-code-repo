import numpy as np
import read_data as rd

# Nominal values for the normalization
nom = {x1: 15, x2:1.5, u1:450, u2:0.2, F:200, T:250, y1:15}

class normalized_data(rd.data):
    def __init__(self, ):
        # Load the data
        self.data = rd.data(tt, age, num)
        # Normalize the data
        self.normalize_data()
