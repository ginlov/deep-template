from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __getitem__(self, item):
        pass

    def __len__(self):
        pass
