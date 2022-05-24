import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import segmentation_models_pytorch as smp

epochs = 25
train_batch_size = 32
val_batch_size = 32
device = "cuda:0" if torch.cuda.is_available() else "cpu"
bands = ["B02", "B03", "B04", "B08"]
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam
model = smp.DeepLabV3Plus(
    encoder_name="resnet101",
    in_channels=4,
    classes=2,
)
scaler = torch.cuda.amp.GradScaler()
learning_rate = 3e-4
train_transforms = A.Compose(
        [
            A.Rotate(limit=60, p=0.6),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2(),
        ],
    )
val_transforms = A.Compose(
    [
        ToTensorV2(),
    ]
)
