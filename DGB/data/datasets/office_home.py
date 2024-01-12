import sys
import glob
import os.path as osp
from .build_dataset import DATASET_REGISTRY
from .base_dataset import Datum, BaseDataset
from DGB.utils import list_non_hidden_directory


@DATASET_REGISTRY.register()
class OfficeHome(BaseDataset):
    """Office-Home.

    Statistics:
        - Around 15,500 images.
        - 65 classes related to office and home objects.
        - 4 domains: Art, Clipart, Product, Real World.
        - URL: http://hemanthdv.org/OfficeHome-Dataset/.

    Reference:
        - Venkateswara et al. Deep Hashing Network for Unsupervised
        Domain Adaptation. CVPR 2017.
    """

    def __init__(self, cfg):
        self._dataset_dir = "office_home"
        self._domains = ["art", "clipart", "product", "real_world"]
        self._data_url = "https://drive.google.com/uc?id=19NGHnQNJst8XlOeq5ThFS3U6kUnKMA3g"
        dataset_path = osp.abspath(osp.expanduser(cfg.DATASET.PATH))
        self._dataset_dir = osp.join(dataset_path, self._dataset_dir)
        self.domain_info = {}

        if not osp.exists(self._dataset_dir):
            dst = osp.join(dataset_path, "office_home.zip")
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
            train_dir = osp.join(self._dataset_dir, domain_name, "train")
            img_path_class_label_list = _load_img_paths(train_dir)
            val_dir = osp.join(self._dataset_dir, domain_name, "val")
            img_path_class_label_list += _load_img_paths(val_dir)
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
