import math
import torch
from tabulate import tabulate
from torch.nn import functional as F
from DGB.model.ops import SelfPacedLearning, SymmetricSelfPacedLearning
from DGB.model import build_network
from DGB.engine.trainer import GenericNet
from DGB.utils import count_num_parameters
from DGB.engine import TRAINER_REGISTRY, GenericTrainer
from DGB.optim import build_optimizer, build_lr_scheduler


@TRAINER_REGISTRY.register()
class DDAIG(GenericTrainer):
    """Deep Domain-Adversarial Image Generation.

    https://arxiv.org/abs/2003.06054.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lmda = cfg.TRAINER.DDAIG.LMDA
        self.clamp = cfg.TRAINER.DDAIG.CLAMP
        self.clamp_min = cfg.TRAINER.DDAIG.CLAMP_MIN
        self.clamp_max = cfg.TRAINER.DDAIG.CLAMP_MAX
        self.warmup = cfg.TRAINER.DDAIG.WARMUP
        self.alpha = cfg.TRAINER.DDAIG.ALPHA

    def build_model(self):
        print("Building Label Classifier")
        self.label_classifier = GenericNet(self.cfg, self.num_classes)
        self.label_classifier.to(self.device)
        self.optimizer_label = build_optimizer(self.label_classifier, self.cfg.OPTIM)
        self.scheduler_label = build_lr_scheduler(self.optimizer_label, self.cfg.OPTIM)
        self.model_registration("label_classifier", self.label_classifier, self.optimizer_label, self.scheduler_label)

        print("Building Domain Classifier")
        self.domain_classifier = GenericNet(self.cfg, self.num_source_domains)
        self.domain_classifier.to(self.device)
        self.optimizer_domain = build_optimizer(self.domain_classifier, self.cfg.OPTIM)
        self.scheduler_domain = build_lr_scheduler(self.optimizer_domain, self.cfg.OPTIM)
        self.model_registration("domain_classifier", self.domain_classifier, self.optimizer_domain, self.scheduler_domain)

        print("Building Domain Transformation Net")
        self.domain_transformation_net = build_network(self.cfg.TRAINER.DDAIG.G_ARCH)
        self.domain_transformation_net.to(self.device)
        self.optimizer_domain_transformation = build_optimizer(self.domain_transformation_net, self.cfg.OPTIM)
        self.scheduler_domain_transformation = build_lr_scheduler(self.optimizer_domain_transformation, self.cfg.OPTIM)
        self.model_registration("domain_transformation_net", self.domain_transformation_net, self.optimizer_domain_transformation, self.scheduler_domain_transformation)

        model_parameters_table = [
            ["Model", "# Parameters"],
            ["Label Classifier", f"{count_num_parameters(self.label_classifier):,}"],
            ["Domain Classifier", f"{count_num_parameters(self.domain_classifier):,}"],
            ["Domain Transformation Net", f"{count_num_parameters(self.domain_transformation_net):,}"]
        ]
        print(tabulate(model_parameters_table))

    def forward_backward(self, batch_data):
        input_data_original, class_label, domain_label = self.parse_batch_train(batch_data)

        #############
        # Update Domain Transformation Net
        #############
        input_data_augmented = self.domain_transformation_net(input_data_original, lmda=self.lmda)
        #
        domain_transformation_loss = 0
        # Minimize Label Classifier Loss
        domain_transformation_loss += F.cross_entropy(self.label_classifier(input_data_augmented), class_label)
        # # Maximize Domain Classifier Loss
        domain_transformation_loss -= F.cross_entropy(self.domain_classifier(input_data_augmented), domain_label)
        self.model_backward_and_update(domain_transformation_loss, "domain_transformation_net")

        # Perturb Data with Updated Domain Transformation Net
        with torch.no_grad():
            input_data_augmented = self.domain_transformation_net(input_data_original, lmda=self.lmda)

        #############
        # Update Label Classifier
        #############
        if self.cfg.TRAIN.OP == "None":
            label_loss_original = F.cross_entropy(self.label_classifier(input_data_original), class_label)
            label_loss_augmented = F.cross_entropy(self.label_classifier(input_data_augmented), class_label)
            label_loss = (1.0 - self.alpha) * label_loss_original + self.alpha * label_loss_augmented
        else:
            # Activate Sample Selection
            label_loss_original = F.cross_entropy(self.label_classifier(input_data_original), class_label, reduction="none")
            label_loss_augmented = F.cross_entropy(self.label_classifier(input_data_augmented), class_label, reduction="none")
            label_loss = (1.0 - self.alpha) * label_loss_original + self.alpha * label_loss_augmented
            if self.cfg.TRAIN.OP == "SPL":
                # Self-Paced Learning
                current_lmda = math.ceil(((self.current_epoch + 1) / self.cfg.OPTIM.MAX_EPOCH) * 10) / 10
                self_paced_learning = SelfPacedLearning(lmda=current_lmda)
                label_loss = self_paced_learning(label_loss)
            elif self.cfg.TRAIN.OP == "SSPL":
                # Symmetric Self-Paced Learning
                symmetric_self_paced_learning = SymmetricSelfPacedLearning(self.cfg, self.current_epoch)
                label_loss = symmetric_self_paced_learning(label_loss)
            else:
                raise NotImplementedError

        self.model_backward_and_update(label_loss, "label_classifier")

        #############
        # Update Domain Classifier
        #############
        domain_loss_original = F.cross_entropy(self.domain_classifier(input_data_original), domain_label)
        self.model_backward_and_update(domain_loss_original, "domain_classifier")

        loss_summary = {
            # "domain_transformation_loss": domain_transformation_loss.item(),
            "label_loss_original": label_loss.item(),
            "domain_loss_original": domain_loss_original.item()
        }

        if self.batch_index + 1 == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_data):
        input_data = batch_data["img"].to(self.device)
        class_label = batch_data["class_label"].to(self.device)
        domain_label = batch_data["domain_label"].to(self.device)
        return input_data, class_label, domain_label

    def model_inference(self, input_data):
        return self.label_classifier(input_data)
