from PIL import Image
from torch.utils.data import Dataset
import os
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

class CovidDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file, sep=';', header=None, names=['image_name', 'class'])
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform
        self.targets = self.annotations['class'].tolist()

        self.folder_mapping = {
            0: 'COVID', 1: 'Lung_Opacity', 2: 'Normal', 3: 'Viral Pneumonia'
        }

        print(f"Root directory (absolute): {self.root_dir}")

        for index, row in self.annotations.iterrows():
            img_name, class_label = row['image_name'], row['class']
            print(f"Row {index}: image_name={img_name}, class_label={class_label}")
            try:
                class_label = int(class_label)
            except ValueError as e:
                print(f"ValueError: {e}. class_label: {class_label}")
                continue
            class_folder = self.folder_mapping.get(class_label, None)
            if class_folder is None:
                print(f"Class label {class_label} not found in folder mapping.")
                continue
            img_path = os.path.join(self.root_dir, class_folder, img_name)
            print(f"Generated image path: {img_path}")
            if not os.path.isfile(img_path):
                print(f"File not found: {img_path}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = self.annotations.iloc[index]['image_name']
        class_label = self.annotations.iloc[index]['class']

        class_label = int(class_label)

        class_folder = self.folder_mapping[class_label]

        img_path = os.path.join(self.root_dir, class_folder, img_name)

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        y_label = torch.tensor(class_label, dtype=torch.long)

        return image, y_label, img_name

    @classmethod
    def create_dataloaders(
            cls,
            csv_file,
            root_dir,
            batch_size=64,
            train_ratio=0.70,
            val_ratio=0.15,
            transform=None
    ):

        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])

        dataset = cls(csv_file=csv_file, root_dir=root_dir, transform=transform)

        n = len(dataset)
        train_size = int(train_ratio * n)
        val_size = int(val_ratio * n)
        test_size = n - train_size - val_size

        train_set, val_set, test_set = torch.utils.data.random_split(
            dataset,
            [train_size, val_size, test_size]
        )

        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_set, batch_size=batch_size)
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size)

        return train_loader, val_loader, test_loader




#
# if __name__ == "__main__":
#     script_dir = os.path.dirname(os.path.abspath(__file__))  # src/data/
#     project_root = os.path.dirname(os.path.dirname(script_dir))
#
#     csv_path = os.path.join(project_root, 'data', 'raw', 'COVID-19', 'covid_dataset.csv')
#     root_dir = os.path.join(project_root, 'data', 'raw', 'COVID-19')
#
#     print(f"CSV path: {csv_path}")
#     print(f"Root dir: {root_dir}")
#     print(f"CSV exists: {os.path.exists(csv_path)}")
#
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])
#
#     dataset = CovidDataset(
#         csv_file=csv_path,
#         root_dir=root_dir,
#         transform=transform
#     )
#
#     dataloader = DataLoader(
#         dataset,
#         batch_size=32,
#         shuffle=True,
#         num_workers=0
#     )
#
#     # Test
#     print("\n" + "=" * 50)
#     print("Testing DataLoader")
#     print("=" * 50)
#
#     images, labels, img_names = next(iter(dataloader))
#     print(f"Batch images shape: {images.shape}")
#     print(f"Batch labels shape: {labels.shape}")
#     print(f"Labels: {labels[:5]}")
#     print(f"Image names: {img_names[:3]}")