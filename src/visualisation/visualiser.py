from abc import ABC
from typing import List, Tuple

import pandas as pd


class Visualiser(ABC):
    def __init__(self, data: pd.DataFrame):
        self.data = data


    def visualise(self):
        pass



