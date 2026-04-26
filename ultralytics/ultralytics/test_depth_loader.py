from ultralytics.data.dataset import YOLODataset


data = {"names": {0: "person"}}


dataset = YOLODataset(
    img_path=r"dataset\images\train",
    data=data,
    task="detect"
)


batch = YOLODataset.collate_fn([dataset[0], dataset[1]])


print("✅ img shape:", batch["img"].shape)
print("✅ depth shape:", batch["depth"].shape)

