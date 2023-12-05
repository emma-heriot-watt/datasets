import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Union

import faiss
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, CLIPVisionModel

from emma_datasets.common import get_progress


class CLIProcessor:
    """CLIP processor class.

    Used to select diverse images by performing kmeans on their CLIP embeddings.
    """

    def __init__(
        self, image_encodings_path: Path = Path("/home/ubuntu/data/clip_features/")  # noqa: WPS404
    ) -> None:
        self.image_encodings_path = image_encodings_path

    def __call__(self, image_names: list[str], centroids: int = 16) -> tuple[list[str], list[int]]:
        """Select the centroid-most diverse images."""
        # If the number of images is less than the number of centroids, return all the images.
        if len(image_names) <= centroids:
            return image_names, list(range(len(image_names)))

        image_encodings = []
        for image_name in image_names:
            image_encoding_basename = f"{os.path.splitext(image_name)[0]}.pt"
            image_encoding_path = self.image_encodings_path.joinpath(image_encoding_basename)
            image_encoding = torch.load(image_encoding_path)
            image_encodings.append(image_encoding)

        tensor_image_encodings = torch.stack(image_encodings)
        means = tensor_image_encodings.mean(dim=1, keepdim=True)
        stds = tensor_image_encodings.std(dim=1, keepdim=True)
        normalized_data = (tensor_image_encodings - means) / stds

        kmeans = faiss.Kmeans(
            normalized_data.shape[1],
            centroids,
            verbose=False,
            min_points_per_centroid=1,
        )
        kmeans.train(normalized_data)

        index = faiss.IndexFlatL2(normalized_data.shape[1])
        index.add(normalized_data)
        _, indices = index.search(kmeans.centroids, 1)

        indices = indices.squeeze()
        return [image_names[idx] for idx in indices.squeeze()], indices.tolist()


class CLIPDataset(Dataset[tuple[torch.Tensor, str]]):
    """Placeholder dataset to get CLIP embeddings."""

    def __init__(
        self,
        root_vision_path: Path,
        metadata_files: list[Path],
        output_dir: Path,
        processor: AutoProcessor,
    ) -> None:
        self.root_vision_path = root_vision_path
        self.metadata_files = metadata_files
        self.output_dir = output_dir
        self.processor = processor

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.metadata_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        """Get an item from dataset."""
        metadata_json_path = self.metadata_files[idx]
        full_image_name = metadata_json_path.parent.joinpath(
            f"{metadata_json_path.stem.split('_')[0]}_color.png",
        )
        image_name = str(full_image_name.relative_to(self.root_vision_path))

        image = Image.open(full_image_name)
        output_pt = os.path.splitext(str(image_name).replace(os.sep, "__"))[0]
        output_pt = os.path.join(self.output_dir, f"{output_pt}.pt")
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs, output_pt


class CLIPFeatureExtractor:
    """CLIP feature extractor."""

    def __init__(
        self,
        root_vision_path: Path,
        metadata_file: Path,
        output_dir: Path,
        dataset_version: str,
        batch_size: int,
        num_workers: int,
        model_name: str,
        limit_examples: Optional[int],
    ):
        output_dir.mkdir(parents=True, exist_ok=True)

        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.model = self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

        self.metadata_files = load_all_metadata_files(
            root_vision_path, metadata_file, limit_examples, dataset_version
        )
        self.bsz = batch_size
        self.dataset = CLIPDataset(
            root_vision_path, self.metadata_files, output_dir, self.processor
        )
        self.dataloader = DataLoader(
            self.dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )

    def run(self) -> None:
        """Run the feature extractor."""
        progress = get_progress()
        task_id = progress.add_task(
            "Encoding images with CLIP",
            visible=True,
            start=True,
            total=int(len(self.dataset) // self.bsz),
            comment="",
        )
        with progress:
            for batch in self.dataloader:
                inputs, output_pt_files = batch
                inputs["pixel_values"] = inputs["pixel_values"].squeeze(1)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                batch_encodings = outputs.pooler_output
                for image_encoding, output_pt_file in zip(batch_encodings, output_pt_files):
                    torch.save(image_encoding, output_pt_file)
            progress.advance(task_id)


def get_metadata_version(root_file_path: Union[str, Path]) -> str:
    """Get the version from a metadata filepath."""
    return str(root_file_path).split("object_detection_data_")[1][:2]


def load_all_metadata_files(
    root_vision_path: Path,
    metadata_file: Path,
    limit_examples: Optional[int] = None,
    dataset_version: Optional[str] = None,
) -> list[Path]:
    """Reads all the available image annotation files."""
    with open(metadata_file) as f:
        annotation_files = f.readlines()
    annotation_files = sorted([line.strip() for line in annotation_files])
    metadata_files_temp = sorted(
        [root_vision_path.joinpath(line.strip()) for line in annotation_files]
    )
    if dataset_version is not None:
        metadata_files_temp = [
            metadata_file
            for metadata_file in metadata_files_temp
            if get_metadata_version(metadata_file) == dataset_version
        ]

    if limit_examples is not None:
        metadata_files_temp = metadata_files_temp[:limit_examples]

    metadata_files = []

    progress = get_progress()
    task_id = progress.add_task(
        f"Loading metadata from file {metadata_file}",
        visible=True,
        start=True,
        total=len(metadata_files_temp),
        comment="",
    )
    with progress:
        for meta_path in metadata_files_temp:
            img_num = meta_path.name.split("_")[0]
            subroot_dir = meta_path.parent
            image_path = subroot_dir.joinpath(f"{img_num}_color.png")
            image_seg_path = subroot_dir.joinpath(f"{img_num}_seg.png")
            if image_path.exists() and image_seg_path.exists():
                metadata_files.append(Path(meta_path))
            progress.advance(task_id)

    return metadata_files


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--root_vision_path",
        type=Path,
        help="Path to the root directory containing the vision datasets",
        default=Path("/home/ubuntu/data/object_detection"),
    )

    parser.add_argument(
        "--input_metadata_txt_path",
        type=Path,
        help="Path to the root directory containing the vision datasets",
        default=Path(
            "/home/ubuntu/data/datav2_collapsev4_isvalidv4_rgv1.12_classfiltered_train_09_09_2022/metadata_train.txt"
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to the output directory containing the clip features",
        default=Path("/home/ubuntu/data/clip_features"),
    )

    parser.add_argument(
        "--limit_examples",
        type=int,
        help="Limit of examples",
    )
    parser.add_argument(
        "--dataset_version",
        type=str,
        help="Use only examples from a specific dataset version",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,  # noqa: WPS432
        help="Number of workers",
    )

    args = parser.parse_args()

    feature_extractor = CLIPFeatureExtractor(
        root_vision_path=args.root_vision_path,
        metadata_file=args.input_metadata_txt_path,
        output_dir=args.output_dir,
        dataset_version=args.dataset_version,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_name="openai/clip-vit-large-patch14",
        limit_examples=args.limit_examples,
    )

    feature_extractor.run()
