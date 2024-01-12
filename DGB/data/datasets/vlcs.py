import sys
import glob
import os.path as osp
from .build_dataset import DATASET_REGISTRY
from .base_dataset import Datum, BaseDataset
from DGB.utils import list_non_hidden_directory


@DATASET_REGISTRY.register()
class VLCS(BaseDataset):
    """VLCS.

    Statistics:
        - 4 domains: CALTECH, LABELME, PASCAL, SUN.
        - 5 categories: bird, car, chair, dog, and person.

    Reference:
        - Torralba and Efros. Unbiased look at dataset bias. CVPR 2011.
    """

    def __init__(self, cfg):
        self._dataset_dir = "vlcs"
        self._domains = ["caltech", "labelme", "pascal", "sun"]
        self._data_url = "https://drive.google.com/uc?id=1vN7bWbEpKD-m0Q5M0QEmZDgHv3L1EYC5"
        dataset_path = osp.abspath(osp.expanduser(cfg.DATASET.PATH))
        self._dataset_dir = osp.join(dataset_path, self._dataset_dir)
        self.domain_info = {}

        if not osp.exists(self._dataset_dir):
            dst = osp.join(dataset_path, "vlcs.zip")
            self.download_data_from_gdrive(dst)

        self.check_input_domains(cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAIN)

        train_data = self.read_data(cfg.DATASET.SOURCE_DOMAINS)
        test_data = self.read_data(cfg.DATASET.TARGET_DOMAIN)

        super().__init__(dataset_dir=self._dataset_dir, domains=self._domains, data_url=self._data_url, train_data=train_data, test_data=test_data)

    def read_data(self, input_domains):

        def _load_img_paths(directory):
            folders = list_non_hidden_directory(directory, sort=True)
            images_ = []

            for class_label, folder in enumerate(folders):
                img_paths = glob.glob(osp.join(directory, folder, "*.jpg"))

                for img_path in img_paths:
                    images_.append((img_path, class_label))

            return images_

        img_datums = []

        for domain_label, domain_name in enumerate(input_domains):
            data_dir = osp.join(self._dataset_dir, domain_name, "full")
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
