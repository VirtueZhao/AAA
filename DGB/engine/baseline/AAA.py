import copy

import torch

from tqdm import tqdm
from tabulate import tabulate

# from collections import OrderedDict
from DGB.model import build_network
from torch.nn import functional as F
from DGB.engine.trainer import GenericNet
from DGB.engine import TRAINER_REGISTRY, GenericTrainer
from DGB.optim import build_optimizer, build_lr_scheduler
from DGB.utils import (
    count_num_parameters,
    # evaluator,
    measure_diversity,
    compute_impact_factor,
    # TripletLoss,
)

# from DGB.evaluation import build_evaluator


@TRAINER_REGISTRY.register()
class AAA(GenericTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.lmda_domain = 0.3
        self.lmda_class = 0.3
        self.alpha = 0.5

        self.best_mAP = 0
        self.best_cmc_scores = None

        self.best_accuracy = 0

    def build_model(self):
        print("Build Feature Extractor")
        self.feature_extractor = GenericNet(self.cfg, self.num_classes)
        self.feature_extractor.to(self.device)
        self.optimizer_feature_extractor = build_optimizer(
            self.feature_extractor, self.cfg.OPTIM
        )
        self.scheduler_feature_extractor = build_lr_scheduler(
            self.optimizer_feature_extractor, self.cfg.OPTIM
        )
        self.model_registration(
            "feature_extractor",
            self.feature_extractor,
            self.optimizer_feature_extractor,
            self.scheduler_feature_extractor,
        )

        print("Build Domain Generator")
        self.domain_generator = build_network("fcn_3x32_gctx")
        self.domain_generator.to(self.device)
        self.optimizer_domain_generator = build_optimizer(
            self.domain_generator, self.cfg.OPTIM
        )
        self.scheduler_domain_generator = build_lr_scheduler(
            self.optimizer_domain_generator, self.cfg.OPTIM
        )
        self.model_registration(
            "domain_generator",
            self.domain_generator,
            self.optimizer_domain_generator,
            self.scheduler_domain_generator,
        )

        print("Build Domain Discriminator")
        self.domain_discriminator = GenericNet(self.cfg, self.num_source_domains)
        self.domain_discriminator.to(self.device)
        self.optimizer_domain_discriminator = build_optimizer(
            self.domain_discriminator, self.cfg.OPTIM
        )
        self.scheduler_domain_discriminator = build_lr_scheduler(
            self.optimizer_domain_discriminator, self.cfg.OPTIM
        )
        self.model_registration(
            "domain_discriminator",
            self.domain_discriminator,
            self.optimizer_domain_discriminator,
            self.scheduler_domain_discriminator,
        )

        print("Build Class Generator")
        self.class_generator = build_network("fcn_3x32_gctx")
        self.class_generator.to(self.device)
        self.optimizer_class_generator = build_optimizer(
            self.class_generator, self.cfg.OPTIM
        )
        self.scheduler_class_generator = build_lr_scheduler(
            self.optimizer_class_generator, self.cfg.OPTIM
        )
        self.model_registration(
            "class_generator",
            self.class_generator,
            self.optimizer_class_generator,
            self.scheduler_class_generator,
        )

        print("Build Class Discriminator")
        self.class_discriminator = GenericNet(self.cfg, self.num_classes)
        self.class_discriminator.to(self.device)
        self.optimizer_class_discriminator = build_optimizer(
            self.class_discriminator, self.cfg.OPTIM
        )
        self.scheduler_class_discriminator = build_lr_scheduler(
            self.optimizer_class_discriminator, self.cfg.OPTIM
        )
        self.model_registration(
            "class_discriminator",
            self.class_discriminator,
            self.optimizer_class_discriminator,
            self.scheduler_class_discriminator,
        )

        model_parameters_table = [
            ["Model", "# Parameters"],
            ["Feature Extractor", f"{count_num_parameters(self.feature_extractor):,}"],
            # ["Domain Generator", f"{count_num_parameters(self.domain_generator):,}"],
            # [
            #     "Domain Discriminator",
            #     f"{count_num_parameters(self.domain_discriminator):,}",
            # ],
            ["Class Generator", f"{count_num_parameters(self.class_generator):,}"],
            [
                "Class Discriminator",
                f"{count_num_parameters(self.class_discriminator):,}",
            ],
        ]

        print(tabulate(model_parameters_table))

    def forward_backward(self, batch_data):
        input_data, class_label, domain_label = self.parse_batch_train(batch_data)

        if self.current_epoch + 1 <= -1:
            # print("Warm-up Epoch")
            output, representations = self.feature_extractor(
                input_data, return_feature=True
            )
            c_loss = F.cross_entropy(output, class_label)
            # triplet_loss = TripletLoss(margin=1.2)
            # t_loss = triplet_loss(representations, class_label)
            # loss_feature_extractor = c_loss + t_loss
            loss_feature_extractor = c_loss

            self.model_backward_and_update(loss_feature_extractor)
        else:
            temp_data = copy.deepcopy(input_data)
            temp_model = copy.deepcopy(self.feature_extractor)
            _, embeddings = temp_model(temp_data, return_feature=True)

            embeddings_min = torch.min(embeddings)
            embeddings_max = torch.max(embeddings)
            normalized_embeddings = (embeddings - embeddings_min) / (
                embeddings_max - embeddings_min
            )

            diversity = measure_diversity(
                normalized_embeddings.cpu(), diversity_type="gini"
            )
            # print(type(diversity))
            # print(diversity.shape)
            # print("Current Diversity: {}".format(diversity.mean()))
            self.lmda_domain = self.lmda_class = compute_impact_factor(
                diversity, lower_bound=0, upper_bound=0.3
            )
            # print("Current Lambda: {}".format(self.lmda_domain))

            # Update D-GAN
            # Update Domain Generator
            temp_input = copy.deepcopy(input_data)
            input_data_domain_augmented = self.domain_generator(
                temp_input, lmda=self.lmda_domain
            )
            loss_domain_generator = 0
            temp_feature_extractor = copy.deepcopy(self.feature_extractor)
            loss_domain_generator += F.cross_entropy(
                temp_feature_extractor(input_data_domain_augmented), class_label
            )
            loss_domain_generator -= F.cross_entropy(
                self.domain_discriminator(input_data_domain_augmented), domain_label
            )
            self.model_backward_and_update(loss_domain_generator, "domain_generator")

            # Update Domain Discriminator
            input_data_domain_augmented = self.domain_generator(
                temp_input, lmda=self.lmda_domain
            )
            loss_domain_discriminator = 0
            loss_domain_discriminator += F.cross_entropy(
                self.domain_discriminator(temp_input), domain_label
            )
            loss_domain_discriminator += F.cross_entropy(
                self.domain_discriminator(input_data_domain_augmented), domain_label
            )
            self.model_backward_and_update(
                loss_domain_discriminator, "domain_discriminator"
            )

            # Update C-GAN
            # Update Class Generator
            temp_input = copy.deepcopy(input_data)
            input_data_class_augmented = self.class_generator(
                temp_input, lmda=self.lmda_class
            )
            loss_class_generator = 0
            temp_feature_extractor = copy.deepcopy(self.feature_extractor)
            loss_class_generator += F.cross_entropy(
                temp_feature_extractor(input_data_class_augmented), class_label
            )
            loss_class_generator -= F.cross_entropy(
                self.class_discriminator(input_data_class_augmented), class_label
            )
            self.model_backward_and_update(loss_class_generator, "class_generator")

            # Update Class Discriminator
            input_data_class_augmented = self.class_generator(
                input_data, lmda=self.lmda_class
            )
            loss_class_discriminator = 0
            loss_class_discriminator += F.cross_entropy(
                self.class_discriminator(input_data), class_label
            )
            loss_class_discriminator += F.cross_entropy(
                self.class_discriminator(input_data_class_augmented), class_label
            )
            self.model_backward_and_update(
                loss_class_discriminator, "class_discriminator"
            )

            # Update Feature Extractor
            # Generate Domain Perturbation and Class Perturbation
            with torch.no_grad():
                temp_input = copy.deepcopy(input_data)
                _, domain_perturbation = self.domain_generator(
                    temp_input, lmda=self.lmda_domain, return_p=True
                )
                _, class_perturbation = self.class_generator(
                    temp_input, lmda=self.lmda_class, return_p=True
                )
                # self.lmda_domain = self.lmda_class = 0
                input_data_domain_class_augmented = (
                    temp_input
                    + self.lmda_domain * domain_perturbation
                    + self.lmda_class * class_perturbation
                )

            pred, representations = self.feature_extractor(
                input_data, return_feature=True
            )
            pred_augmented, representations_augmented = self.feature_extractor(
                input_data_domain_class_augmented, return_feature=True
            )

            # triplet_loss = TripletLoss(margin=1.2)
            loss_c1 = F.cross_entropy(pred, class_label)
            loss_c2 = F.cross_entropy(pred_augmented, class_label)
            # loss_t1 = triplet_loss(representations, class_label)
            # loss_t2 = triplet_loss(representations_augmented, class_label)
            loss_feature_extractor = (1.0 - self.alpha) * (loss_c1) + self.alpha * (
                loss_c2
            )
            self.model_backward_and_update(loss_feature_extractor, "feature_extractor")

        loss_summary = {
            "loss_feature_extractor": loss_feature_extractor.item(),
            # "loss_domain_generator": loss_domain_generator.item(),
            # "loss_domain_discriminator": loss_domain_discriminator.item(),
            # "loss_class_generator": loss_class_generator.item(),
            # "loss_class_discriminator": loss_class_discriminator.item()
        }

        if self.batch_index + 1 == self.num_batches:
            self.update_lr()

        return loss_summary, representations, representations_augmented, domain_label

    def parse_batch_train(self, batch_data):
        input_data = batch_data["img"].to(self.device)
        class_label = batch_data["class_label"].to(self.device)
        domain_label = batch_data["domain_label"].to(self.device)
        return input_data, class_label, domain_label

    def model_inference(self, input_data):
        return self.feature_extractor(input_data)

    def after_epoch(self):
        print("After Epoch Testing")
        temp_feature_extractor = copy.deepcopy(self.feature_extractor)
        temp_feature_extractor.eval()

        self.evaluator.reset()

        with torch.no_grad():
            for batch_index, batch_data in enumerate(tqdm(self.test_data_loader)):
                input_data, class_label = self.parse_batch_test(batch_data)
                output = temp_feature_extractor(input_data)
                self.evaluator.process(output, class_label)
        evaluation_results = self.evaluator.evaluate()
        if self.best_accuracy < evaluation_results["accuracy"]:
            self.best_accuracy = evaluation_results["accuracy"]
        print("Best Accuracy: {}".format(self.best_accuracy))
