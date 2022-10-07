import numpy as np
import pandas as pd
import os


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
    
    def get_sanity_accuracy(self):
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return (df["correct"]).mean()


def print_radii(filename):
    # ck_radii = np.array([0.1 * r for r in range(10)])
    ck_radii = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])

    file = ApproximateAccuracy(filename)
    print('Loaded results from {}'.format(filename))
    abstention_rate = file.get_abstention_rate()
    print('abstention rate is {}'.format(abstention_rate))
    radii = file.at_radii(ck_radii)
    print('certified radii w.r.t. {} is'.format(ck_radii))
    print(list(radii))

def plot_curve():
    files = os.listdir('certi_deno/cifar10/')
    filename_list = ['certi_deno/cifar10/' + file + '/sigma_25' for file in files]
    names = files
    ck_radii = np.array([0.01 * r for r in range(100)])
    radii_list = []
    # # check sanity accuracy
    # sanity_acc_list = []
    # abstention_rate_list = []

    for filename in filename_list:
        file = ApproximateAccuracy(filename)
        radii = file.at_radii(ck_radii)
        radii_list.append(radii)
        sanity_acc_list.append(file.get_sanity_accuracy())
        abstention_rate_list.append(file.get_abstention_rate())
    # print(sanity_acc_list)
    # print(abstention_rate_list)

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    for name, radii in zip(names, radii_list):
        plt.plot(ck_radii, radii, linestyle="--", label=name, linewidth=2)
        # plt.plot(ck_radii, vit_small, label="ViT-S/16", marker='x', color="c", linewidth=3, markersize=8, linestyle="--")
        # plt.plot(ck_radii, res18, label="ResNet18", marker='.', color="coral", linewidth=3, markersize=8)
    ax.set_xlabel('Radius', fontsize=20)
    ax.set_ylabel('Certified Accuracy', fontsize=20)
    plt.legend(fontsize=8)
    plt.savefig('certified_radii_cifar10.png')
    plt.show()





if __name__ == '__main__':
    filename = 'certi_deno/clip_vit16_feat_denoising/softmax_alpha_30/sigma_25'
    # filename = '/home/ubuntu/denoised-smoothing/code/certi_deno/clip_vit16_dual/sigma_25'
    print_radii(filename)
    # plot_curve()
