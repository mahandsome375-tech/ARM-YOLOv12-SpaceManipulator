from torch.utils.data import DataLoader
from ultralytics.data.dataset import YOLODataset


data_cfg = {
    "train": r"dataset\images\train",
    "names": {0: "obj"}
}



dataset = YOLODataset(img_path=data_cfg["train"], data=data_cfg, task="detect")


dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=YOLODataset.collate_fn)


batch = next(iter(dataloader))
for k in ["depth", "phi_sin", "phi_cos", "tau", "logvar", "valid_mask"]:
    x = batch[k]
    print(k, x.shape, x.dtype)
