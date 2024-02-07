
import abc
import os


DATA_DIR = "./data"


class Dataset(abc.ABC):

    def __init__(self, data_dir: str = DATA_DIR):

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        self.data_dir = data_dir

        return

    @abc.abstractmethod
    def download_record(self, id: str, replace: bool = False):

        raise NotImplementedError

    @abc.abstractmethod
    def load_record(self, id: str):

        raise NotImplementedError
