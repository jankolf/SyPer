from enum import Enum
from pathlib import Path
import queue as Queue
import threading

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

from .nir import NIRTrain, NIRTest
from .synthetic import NoIDSyntheticDataset

class ProtocolType(Enum):
    CLOSED_WORLD            = "closed_world"
    OPEN_WORLD_CLOSED_VAL   = "open_world"
    OPEN_WORLD_OPEN_VAL     = "open_world_valopen"


class DatasetType(Enum):
    TRAIN = "train"
    VAL   = "val"
    TEST  = "test"




class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self




class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                 non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch




class PeriocularDataset(Dataset):

    def __init__(   self, 
                    file_content,
                    img_size : int,
                    flip_L_to_R : bool,
                    dataset_root : str
                ):
        super(PeriocularDataset, self).__init__()

        self.transform = transforms.Compose(
            [
             transforms.Resize((img_size,img_size)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        self.flip_L_to_R = flip_L_to_R

        self._dataset_root  = dataset_root
        self._location_imgs = Path(dataset_root, 
                                   "periocularCropped").resolve()

        self._image_names, self._image_labels = self.extract_data(file_content)

        self._labels_encoded = self.encode_labels(self._image_labels)
        self._num_classes = self.calculate_num_classes(self._image_labels)


    @property
    def image_names(self):
        return self._image_names


    @property
    def num_classes(self):
        return self._num_classes


    def __len__(self):
        return len(self._image_names)


    def __getitem__(self, item_idx):
        image = self.load_image(
                        self._image_names[item_idx]
                )
        class_label = self._labels_encoded[item_idx].item()
        class_label = torch.tensor(class_label, dtype=torch.long)
        return image, class_label


    def extract_data(self, lines):
        
        image_names     = []
        image_labels    = []

        flip_match = {}

        for line in lines:
            splits = line.strip().split(" ")
            if len(splits) < 2:
                continue

            img_name = splits[0]
            img_label = int(splits[1])

            if self.flip_L_to_R:
                key = img_name.split("S")[0]
                if key in flip_match.keys():
                    img_label = flip_match[key]
                else:
                    flip_match[key] = img_label

            image_names.append(img_name)
            image_labels.append(img_label)
                                

        image_labels = torch.tensor(image_labels, 
                                    dtype=torch.long)

        return image_names, image_labels


    def encode_labels(self, original_labels):
        image_labels_encoded = torch.zeros(original_labels.shape, dtype=torch.int32)
        unique = torch.unique(original_labels)

        for idx in range(len(unique)):
            label = unique[idx]
            mask = original_labels == label
            image_labels_encoded[mask] = idx

        return image_labels_encoded


    def calculate_num_classes(self, original_labels):
        return len(torch.unique(original_labels))


    def load_image(self, img_path):
        img = Image.open(self._location_imgs / img_path)
        img = self.transform(img)

        if self.flip_L_to_R and "L.jpg" in img_path:
            img = TF.hflip(img)

        return img




class PeriocularTrain(PeriocularDataset):
    
    def __init__(   self,
                    file_content,
                    img_size : int = 224,
                    flip_L_to_R : bool = False,
                    dataset_root : str = "/data/jkolf/datasets/UFPR-Periocular/"
                ):
        super(PeriocularTrain, self).__init__(
                            file_content=file_content,
                            img_size=img_size,
                            flip_L_to_R=flip_L_to_R,
                            dataset_root=dataset_root
                            )




class PeriocularValidation(PeriocularDataset):

    def __init__(   self,
                    file_content,
                    amount_genuine : int = 10,
                    amount_imposter : int = 50,
                    img_size : int = 224,
                    flip_L_to_R : bool = False,
                    dataset_root : str = "/data/jkolf/datasets/UFPR-Periocular/"
                ):
        super(PeriocularValidation, self).__init__(
                            file_content=file_content,
                            img_size=img_size,
                            flip_L_to_R=flip_L_to_R,
                            dataset_root=dataset_root
                            )
        
        ret = self.generate_pairs(self._image_labels, amount_genuine, amount_imposter)
        self._probes        = ret[0]
        self._references    = ret[1]


    def generate_pairs(self,
                       labels,
                       amount_genuine  : int = 10,
                       amount_imposter : int = 50
                       ):

        N_comp = amount_genuine + amount_imposter
        N_samples = len(labels)

        rows = torch.arange(0, N_samples, dtype=torch.int32)

        probes      = rows.repeat_interleave(N_comp).reshape((N_samples, N_comp))
        references  = torch.full(probes.shape, -1, dtype=torch.int32)
        
        for idx in range(N_samples):
            print(f"Generating validation pair {idx+1}/{N_samples}", end="\r", flush=True)
            mask = labels == labels[idx]
            # Get genuine indices and remove current idx
            genuine_rows = rows[mask]
            genuine_rows = genuine_rows[genuine_rows != idx]
            # Get imposter indices
            imposter_rows = rows[~mask]
        
            # Calculate available 
            N_gen = min(amount_genuine,  len(genuine_rows))
            N_imp = min(amount_imposter, len(imposter_rows))

            # Calculate genuine choice
            selection_gen = torch.randperm(len(genuine_rows ))[:N_gen]
            selection_imp = torch.randperm(len(imposter_rows))[:N_imp]

            selection_gen = genuine_rows[selection_gen]
            selection_imp = imposter_rows[selection_imp]

            # Save the selection
            references[idx, :N_gen] = selection_gen
            references[idx, amount_genuine:amount_genuine+N_imp] = selection_imp
        print("\n")

        references = references.flatten()
        # Create mask with valid entries
        mask = references != -1

        probes = probes.flatten()[mask]
        references = references[mask]
        
        if probes.shape != references.shape:
            raise ValueError("Probe and Reference arrays do not have the same shape.")

        return (probes, references)


    def __len__(self):
        return len(self._probes)

    
    def __getitem__(self, item_idx):

        index_probe = self._probes[item_idx].item()
        index_reference = self._references[item_idx].item()

        if index_probe == index_reference:
            raise ValueError("Generated Genuine/Imposter Pair contains the same data!")

        name_probe = self._image_names[index_probe]
        name_reference = self._image_names[index_reference]

        label_probe = self._image_labels[index_probe].item()
        label_reference = self._image_labels[index_reference].item()
        
        label = int(label_probe == label_reference)

        return (name_probe, name_reference, label)


    def get_base_data(self, base_index):
        name    = self._image_names[base_index]
        label   = self._image_labels[base_index]
        return (name, label)




class PeriocularTest(PeriocularDataset):

    def __init__(self,
                 protocol_type : ProtocolType,
                 fold : int,
                 img_size : int = 224,
                 flip_L_to_R : bool = False,
                 dataset_root : str = "/data/jkolf/datasets/UFPR-Periocular/"
                ):
        test_images = get_file_content(dataset_root,
                                       protocol_type,
                                       DatasetType.TEST,
                                       fold
                                       )
        super(PeriocularTest, self).__init__(
                                    file_content=test_images,
                                    img_size=img_size,
                                    flip_L_to_R=flip_L_to_R,
                                    dataset_root=dataset_root
                                    )
        self._protocol_type = protocol_type
        self._fold = fold

        self._probes, self._references, self._labels = self._read_pair_files()

    
    def __len__(self):
        return len(self._probes)

    
    def __getitem__(self, item_idx):
        P = self._probes[item_idx]
        R = self._references[item_idx]
        L = self._labels[item_idx]
        return (P, R, L)


    def _read_pair_files(self):

        pairs_folder = Path(
                        self._dataset_root,
                        "experimentalProtocol",
                        self._protocol_type.value,
                        "test_pairs",
                        f"fold{self._fold}"
                        )
        pairs_files = [pf for pf in pairs_folder.iterdir() if pf.suffix==".txt"]

        probes = []
        references = []
        labels = []

        for pf in pairs_files:
            with open(pf, "r") as file:
                for line in file.readlines():
                    if len(line) < 2:
                        continue

                    data = line.strip().split(" ")
                    P = data[0].strip()
                    R = data[1].strip()
                    L = int(data[2])

                    if self.flip_L_to_R:
                        C1 = P.split("S")[0]
                        C2 = R.split("S")[0]
                        
                        if C1 == C2 and L == 0:
                            continue
                        
                    probes.append(data[0])
                    references.append(data[1])
                    labels.append(int(data[2]))

        return probes, references, labels




def get_file_content(directory_root : str, protocol_type : ProtocolType, dataset_type : DatasetType, fold : int):
    """
    get_file_content

    Loads all lines of a periocular dataset .txt file.

    Parameters
    ----------
    directory_root : str
        Path to the dataset.
    protocol_type : ProtocolType
        Protocol type of the file (closed_world, open_world, open_world_valopen)
    dataset_type : DatasetType
        Dataset type (train, val, test)
    fold : int
        The fold to load (1,2,3)

    Returns
    -------
    lines : List[str]
        The content of the file
    """
    file_path = Path(
                    directory_root, 
                    "experimentalProtocol",
                    protocol_type.value,
                    f"{dataset_type.value}_fold{fold}.txt")
    with open(file_path, "r") as f:
        lines = f.readlines()
    return lines
