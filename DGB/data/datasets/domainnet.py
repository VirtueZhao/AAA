import sys
import os.path as osp
from .build_dataset import DATASET_REGISTRY
from .base_dataset import Datum, BaseDataset


@DATASET_REGISTRY.register()
class DomainNet(BaseDataset):
    """DomainNet.

    Statistics:
        - 6 distinct domains: Clipart, Infograph, Painting, Quickdraw,
        Real, Sketch.
        - Around 0.6M images.
        - 345 categories.
        - URL: http://ai.bu.edu/M3SDA/.

    Special note: the t-shirt class (327) is missing in painting_train.txt.

    Reference:
        - Peng et al. Moment Matching for Multi-Source Domain
        Adaptation. ICCV 2019.
    """

    def __init__(self, cfg):
        self._dataset_dir = "domainnet"
        self._domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

        dataset_path = osp.abspath(osp.expanduser(cfg.DATASET.PATH))
        self._dataset_dir = osp.join(dataset_path, self._dataset_dir)
        self.domain_info = {}

        self.check_input_domains(cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAIN)

        train_data = self.read_data(cfg.DATASET.SOURCE_DOMAINS)
        test_data = self.read_data(cfg.DATASET.TARGET_DOMAIN)
        
        super().__init__(dataset_dir=self._dataset_dir, domains=self._domains, train_data=train_data, test_data=test_data)

    def read_data(self, input_domains):
        def _load_img_paths(directory):
            img_paths = []
            with open(directory, "r") as file:
                lines = file.readlines()
                for line in lines:
                    img_path, class_label = line.split(" ")
                    img_path = osp.join(self._dataset_dir, img_path)
                    class_label = int(class_label)
                    img_paths.append((img_path, class_label))

            return img_paths

        img_datums = []

        for domain_label, domain_name in enumerate(input_domains):
            data_dir = osp.join(self._dataset_dir, domain_name + "_train.txt")
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
