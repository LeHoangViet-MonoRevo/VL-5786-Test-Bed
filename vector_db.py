from abc import ABC, abstractmethod


class VectorDataBaseInteraction(ABC):
    @abstractmethod
    def search_vector(self):
        pass

    @abstractmethod
    def insert_vector(self):
        pass
