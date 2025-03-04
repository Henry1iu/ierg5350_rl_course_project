import os
import pickle

import h5py
import numpy as np
import ipdb
from tqdm import tqdm

from torch.utils.data import Dataset
from multimodal.dataloaders.ToTensor import MyToTensor
from torchvision import transforms


class MultimodalManipulationDataset(Dataset):
    """Multimodal Manipulation dataset."""

    def __init__(
        self,
        filename_list,
        transform=None,
        episode_length=50,
        training_type="selfsupervised",
        n_time_steps=1,
        action_dim=4,
        pairing_tolerance=0.06
    ):
        """
        Args:
            hdf5_file (handle): h5py handle of the hdf5 file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_path = filename_list
        self.transform = transform
        self.episode_length = episode_length
        self.training_type = training_type
        self.n_time_steps = n_time_steps
        self.dataset = {}
        self.action_dim = action_dim
        self.pairing_tolerance = pairing_tolerance

        self._config_checks()
        self._init_paired_filenames()

    def __len__(self):
        return len(self.dataset_path) * (self.episode_length - self.n_time_steps)

    def __getitem__(self, idx):

        list_index = idx // (self.episode_length - self.n_time_steps)
        dataset_index = idx % (self.episode_length - self.n_time_steps)
        filename = self.dataset_path[list_index][:-8]

        file_number, filename = self._parse_filename(filename)

        unpaired_filename, unpaired_idx = self.paired_filenames[(list_index, dataset_index)]

        if dataset_index >= self.episode_length - self.n_time_steps - 1:
            dataset_index = np.random.randint(
                self.episode_length - self.n_time_steps - 1
            )

        sample = self._get_single(
            self.dataset_path[list_index],
            list_index,
            unpaired_filename,
            dataset_index,
            unpaired_idx,
        )
        return sample

    def _get_single(
        self, dataset_name, list_index, unpaired_filename, dataset_index, unpaired_idx
    ):

        dataset = h5py.File(dataset_name, "r", swmr=True, libver="latest")
        unpaired_dataset = h5py.File(unpaired_filename, "r", swmr=True, libver="latest")

        if self.training_type == "selfsupervised":

            image = dataset["image"][dataset_index]
            depth = dataset["depth_data"][dataset_index]
            proprio = dataset["proprio"][dataset_index][:8]
            force = dataset["ee_forces_continuous"][dataset_index]

            if image.shape[0] == 3:
                image = np.transpose(image, (2, 1, 0))

            if depth.ndim == 2:
                depth = depth.reshape((128, 128, 1))

            flow = np.array(dataset["optical_flow"][dataset_index])
            flow_mask = np.expand_dims(
                np.where(
                    flow.sum(axis=2) == 0,
                    np.zeros_like(flow.sum(axis=2)),
                    np.ones_like(flow.sum(axis=2)),
                ),
                2,
            )

            unpaired_image = image
            unpaired_depth = depth
            unpaired_proprio = unpaired_dataset["proprio"][unpaired_idx][:8]
            unpaired_force = unpaired_dataset["ee_forces_continuous"][unpaired_idx]

            sample = {
                "image": image,
                "depth": depth,
                "flow": flow,
                "flow_mask": flow_mask,
                "action": dataset["action"][dataset_index + 1],
                "force": force,
                "proprio": proprio,
                "ee_yaw_next": dataset["proprio"][dataset_index + 1][:self.action_dim],
                "contact_next": np.array(
                    [dataset["contact"][dataset_index + 1].sum() > 0]
                ).astype(np.float),
                "unpaired_image": unpaired_image,
                "unpaired_force": unpaired_force,
                "unpaired_proprio": unpaired_proprio,
                "unpaired_depth": unpaired_depth,
            }

        dataset.close()
        unpaired_dataset.close()

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _init_paired_filenames(self):
        """
        Precalculates the paired filenames.
        Imposes a distance tolerance between paired images
        """
        tolerance = self.pairing_tolerance

        all_combos = set()

        self.paired_filenames = {}
        for list_index in tqdm(range(len(self.dataset_path)), desc="pairing_files"):
            filename = self.dataset_path[list_index]
            file_number, _ = self._parse_filename(filename[:-8])

            dataset = h5py.File(filename, "r", swmr=True, libver="latest")

            for idx in range(self.episode_length - self.n_time_steps):

                proprio_dist = None
                while proprio_dist is None or proprio_dist < tolerance:
                    # Get a random idx, file that is not the same as current
                    unpaired_dataset_idx = np.random.randint(self.__len__())
                    unpaired_filename, unpaired_idx, _ = self._idx_to_filename_idx(unpaired_dataset_idx)

                    while unpaired_filename == filename:
                        unpaired_dataset_idx = np.random.randint(self.__len__())
                        unpaired_filename, unpaired_idx, _ = self._idx_to_filename_idx(unpaired_dataset_idx)

                    with h5py.File(unpaired_filename, "r", swmr=True, libver="latest") as unpaired_dataset:
                        proprio_dist = np.linalg.norm(dataset['proprio'][idx][:3] - unpaired_dataset['proprio'][unpaired_idx][:3])

                self.paired_filenames[(list_index, idx)] = (unpaired_filename, unpaired_idx)
                all_combos.add((unpaired_filename, unpaired_idx))

            dataset.close()

    def _idx_to_filename_idx(self, idx):
        """
        Utility function for finding info about a dataset index

        Args:
            idx (int): Dataset index

        Returns:
            filename (string): Filename associated with dataset index
            dataset_index (int): Index of data within that file
            list_index (int): Index of data in filename list
        """
        list_index = idx // (self.episode_length - self.n_time_steps)
        dataset_index = idx % (self.episode_length - self.n_time_steps)
        filename = self.dataset_path[list_index]
        return filename, dataset_index, list_index

    def _parse_filename(self, filename):
        """ Parses the filename to get the file number and filename"""
        if filename[-2] == "_":
            file_number = int(filename[-1])
            filename = filename[:-1]
        else:
            file_number = int(filename[-2:])
            filename = filename[:-2]

        return file_number, filename

    def _config_checks(self):
        if self.training_type != "selfsupervised":
            raise ValueError(
                "Training type not supported: {}".format(self.training_type)
            )


class MyMultimodalManipulationDataset(Dataset):
    def __init__(self, root_path, device="cpu"):
        self._root_path = root_path

        self._files = [file for file in os.listdir(self._root_path) if os.path.splitext(file)[1] in ".pkl"]

        # load all the pickle data into memory
        self._data = []
        for f in self._files:
            self._data.extend(self._pickle_load(f))

        self._device = device
        self._tf = transforms.Compose([MyToTensor(device=self._device)])

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._tf(self._data[idx])

    def _pickle_load(self, file):
        sub_dataset = []
        with open(os.path.join(self._root_path, file), 'rb') as f:
            while True:
                try:
                    data = pickle.load(f)
                except EOFError:
                    break
                sub_dataset.append(data)
        return sub_dataset


if __name__ == "__main__":
    path = "/home/jb/projects/Code/IERG5350/project/ierg5350_rl_course_project/multimodal/dataset/simulation_data/eval"
    dataset = MyMultimodalManipulationDataset(path)

    for i in range(5000):
        sample = dataset[i]
        # print("Sampel type: {}".format(type(sample)))
        # print("keys in sample {}: {}".format(i, sample.keys()))
        #
        # # print data info
        # print("Sample {}[{}]: {}, {}, {}".format(i, "color_prev", type(sample["color_prev"]), sample["color_prev"].shape, sample["color_prev"].dtype))
        # print("Sample {}[{}]: {}, {}, {}".format(i, "depth_prev", type(sample["depth_prev"]), sample["depth_prev"].shape, sample["depth_prev"].dtype))
        # print("Sample {}[{}]: {}, {}, {}".format(i, "ft_prev", type(sample["ft_prev"]), sample["ft_prev"].shape, sample["ft_prev"].dtype))
        # print("Sample {}[{}]: {}, {}, {}".format(i, "action", type(sample["action"]), sample["action"].shape, sample["action"].dtype))
        # print("Sample {}[{}]: {}, {}, {}".format(i, "color", type(sample["color"]), sample["color"].shape, sample["color"].dtype))
        # print("Sample {}[{}]: {}, {}, {}".format(i, "depth", type(sample["depth"]), sample["depth"].shape, sample["depth"].dtype))
        # print("Sample {}[{}]: {}, {}, {}".format(i, "contact", type(sample["contact"]), sample["contact"], sample["contact"].dtype))
