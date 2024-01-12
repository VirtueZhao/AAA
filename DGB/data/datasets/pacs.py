import sys
import os.path as osp
from .build_dataset import DATASET_REGISTRY
from .base_dataset import Datum, BaseDataset


@DATASET_REGISTRY.register()
class PACS(BaseDataset):
    """PACS.
    Statistics:
        - 4 domains: Photo (1,670), Art (2,048), Cartoon (2,344), Sketch (3,929).
        - 7 categories: dog, elephant, giraffe, guitar, horse, house and person.

    Reference:
        - Li et al. Deeper, broader and artier domain generalization. ICCV 2017.
    """

    def __init__(self, cfg):
        self._dataset_dir = "pacs"
        self._domains = ["art_painting", "cartoon", "photo", "sketch"]
        self._data_url = "https://drive.google.com/uc?id=1wN5jJiG3makr8D2iDX5CI7oFXGua8nYB"
        dataset_path = osp.abspath(osp.expanduser(cfg.DATASET.PATH))
        self._dataset_dir = osp.join(dataset_path, self._dataset_dir)
        self._image_dir = osp.join(self._dataset_dir, "images")
        self._split_dir = osp.join(self._dataset_dir, "splits")
        self._error_img_paths = ["sketch/dog/n02103406_4068-1.png"]
        self.domain_info = {}

        if not osp.exists(self._dataset_dir):
            dst = osp.join(dataset_path, "pacs.zip")
            self.download_data_from_gdrive(dst)

        self.check_input_domains(cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAIN)

        train_data = self.read_data(cfg.DATASET.SOURCE_DOMAINS)
        test_data = self.read_data(cfg.DATASET.TARGET_DOMAIN)

        super().__init__(dataset_dir=self._dataset_dir, domains=self._domains, data_url=self._data_url, train_data=train_data, test_data=test_data)

    def read_data(self, input_domains):
        def _load_img_paths(directory):
            img_paths = []
            with open(directory, "r") as file:
                lines = file.readlines()
                for line in lines:
                    img_path, class_label = line.split(" ")
                    if img_path in self._error_img_paths:
                        continue
                    img_path = osp.join(self._image_dir, img_path)
                    class_label = int(class_label) - 1
                    img_paths.append((img_path, class_label))

            return img_paths

        img_datums = []

        for domain_label, domain_name in enumerate(input_domains):
            data_dir = osp.join(self._split_dir, domain_name + "_test_kfold.txt")
            img_path_class_label_list = _load_img_paths(data_dir)
            self.domain_info[domain_name] = len(img_path_class_label_list)

            for img_path, class_label in img_path_class_label_list:
                if sys.platform == "linux":
                    class_name = str(img_path.split("/")[-2].lower())
                else:
                    class_name = str(img_path.split("\\")[-2].lower())

                img_datum = Datum(
                    img_path=img_path,
                    class_label=class_label,
                    domain_label=domain_label,
                    class_name=class_name
                )
                img_datums.append(img_datum)

        return img_datums



