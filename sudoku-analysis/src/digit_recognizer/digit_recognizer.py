from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from numpy import ndarray

class DigitRecognizer(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def check_precision(self)->int:
        pass

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def is_trained(self) ->bool:
        pass

    @abstractmethod
    def classify(self, image : ndarray) -> int:
        pass

    @abstractmethod
    def load_model(self) -> None:
        pass