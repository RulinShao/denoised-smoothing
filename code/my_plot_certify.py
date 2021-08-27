import numpy as np
import pandas as pd


class Accuracy(object):
    def at_radii(self, radii: np.ndarray):
        raise NotImplementedError()


class ApproximateAccuracy(Accuracy):
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        return (df["correct"] & (df["radius"] >= radius)).mean()

    def get_abstention_rate(self) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return 1.*(df["predict"]==-1).sum()/len(df["predict"])*100



if __name__ == '__main__':
    filename = 'certi_deno/vit_small/sigma_25'
    ck_radii = np.array([0.1 * r for r in range(10)])

    file = ApproximateAccuracy(filename)
    print('Loaded results from {}'.format(filename))
    abstention_rate = file.get_abstention_rate()
    print('abstention rate is {}'.format(abstention_rate))
    radii = file.at_radii(ck_radii)
    print('certified radii w.r.t. {} is'.format(ck_radii))
    print(radii)
