import numpy as np

y = np.array([[1, -4, 2], [3, 1, 2], [2, 1, -1]])
print(y)

"""
class Makeplace:
    def __init__(self, name, *args):
        self.name = name
        if not args:
            raise ValueError("error occured")
"""

mt2 = np.linalg.inv(y)
print(mt2)

import numpy as np
from fractions import Fraction

# Define the array
y = np.array([[1, 3, 2], [-4, 1, 1], [2, 2, -1]])

# Convert each element to a Fraction
mt2 = np.array([[Fraction(value).limit_denominator() for value in row] for row in mt2])

# Print each row as fractions
print(mt2)
