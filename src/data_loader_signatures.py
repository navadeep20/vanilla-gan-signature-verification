from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_signature_dataloader(
    batch_size=32,
    data_dir="../data/signatures/train"
):
    """
    Returns a DataLoader for signature images.

    Args:
        batch_size (int): Number of images per batch
        data_dir (str): Path to signature dataset directory

    Default:
        data_dir -> real signatures (baseline)
    """

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader
