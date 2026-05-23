from guidedLP.network import net_utils as nu
import polars as pl

class Net:

    def __init__(self,file_path=None) -> None:
        
        self.file_path = file_path

        self.directed = True
        self.com_detect_resolutions = [1.0]
        self.com_detect_its = 10
        self.affinity_epochs = 3
        self.affinity_its = 60

    def configure(self):

        pass

    def create_unipartite(self):

        pass

    def create_bipartite(self):

        pass

    def load_csv_file(self,file_path=None):

        if file_path is not None:
            self.file_path = file_path
        self.data = pl.read_csv(self.file_path)





