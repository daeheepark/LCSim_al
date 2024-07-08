import argparse
import os
import pickle
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
from dataset.womd import WaymoMotionDataset
from sklearn.decomposition import PCA
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataset.utils import TargetBuilder

def parse_args():
    parser = argparse.ArgumentParser("pca-para-gen")
    parser.add_argument(
        "--dataset", dest="dataset", type=str, default="", help="dataset path"
    )
    parser.add_argument(
        "--root", dest="root", type=str, required=True, help="dataset root"
    )
    parser.add_argument(
        "--num_components",
        dest="num_components",
        type=int,
        default=10,
        help="number of components",
    )
    parser.add_argument(
        "--only_fut", action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset == "":
        dataset = WaymoMotionDataset(
            root=args.root,
            split="training",
            data_len=50000,
            transform=TargetBuilder(only_fut=args.only_fut)
        )
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        pca_fit_data = torch.tensor([])
        bar = tqdm(total=len(loader), desc="pca data collection", dynamic_ncols=True)
        for data in loader:
            target = data["agent"]["target"]
            if args.only_fut:
                mask = data["agent"]["valid"][:, 10:].squeeze(-1).all(dim=-1)
                target = target[mask][..., :2].reshape(-1, 160)
            else:
                mask = data["agent"]["valid"].squeeze(-1).all(dim=-1)
                target = target[mask][..., :2].reshape(-1, 182)
            
            pca_fit_data = torch.cat([pca_fit_data, target], dim=0)
            bar.update(1)
        bar.close()
        # save pca data
        with open("gen/pca_fit_data.pkl", "wb") as f:
            pickle.dump(pca_fit_data, f)
    else:
        with open(args.dataset, "rb") as f:
            pca_fit_data = pickle.load(f)
    # pca
    print("pca...")
    pca = PCA(n_components=args.num_components, whiten=True)
    pca.fit(pca_fit_data)
    print(f"explained_variance_ratio_: {pca.explained_variance_ratio_}")
    print(f"explained_variance_: {pca.explained_variance_}")
    # save pca
    with open(f"pca_onlyfut_{args.only_fut}.pkl", "wb") as f:
        pickle.dump(pca, f)


if __name__ == "__main__":
    main()
