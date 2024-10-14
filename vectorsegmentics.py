import numpy as np
import pandas as pd
import re

file_path = ""
txtfile = pd.read_table(file_path)
print(txtfile.head(4))


def bbow(document):
    document = document.replace(",", "")
    document = document.raplace(".", "")
    return document
