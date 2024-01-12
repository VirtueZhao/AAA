import math
import torch
from tabulate import tabulate
from torch.nn import functional as F
from DGB.model.ops import SelfPacedLearning, SymmetricSelfPacedLearning
from DGB.engine.trainer import GenericNet
from DGB.utils import count_num_parameters
from DGB.engine import TRAINER_REGISTRY, GenericTrainer
from DGB.optim import build_optimizer, build_lr_scheduler

@TRAINER_REGISTRY.register()
class CrossGrad(GenericTrainer):
    """Cross-gradient training.

    https://arxiv.org/abs/1804.10745.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.eps_l = cfg.TRAINER.CROSSGRAD.EPS_L
        self.eps_d = cfg.TRAINER.CROSSGRAD.EPS_D
        self.alpha_l = cfg.TRAINER.CROSSGRAD.ALPHA_L
        self.alpha_d = cfg.TRAINER.CROSSGRAD.ALPHA_D

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

        model_parameters_table = [
            ["Model", "# Parameters"],
            ["Label Classifier", f"{count_num_parameters(self.label_classifier):,}"],
            ["Domain Classifier", f"{count_num_parameters(self.domain_classifier):,}"]
        ]
        print(tabulate(model_parameters_table))

    def forward_backward(self, batch_data):
        input_data_original, class_label, domain_label = self.parse_batch_train(batch_data)

        # Compute Domain Perturbation
        loss_domain = F.cross_entropy(self.domain_classifier(input_data_original), domain_label)
        loss_domain.backward()
        grad_domain = torch.clamp(input_data_original.grad.data, min=-0.1, max=0.1)
        input_data_domain_perturb = input_data_original.data + self.eps_l * grad_domain

        # Compute Label Perturbation
        input_data_original.grad.data.zero_()
        loss_label = F.cross_entropy(self.label_classifier(input_data_original), class_label)
        loss_label.backward()
        grad_label = torch.clamp(input_data_original.grad.data, min=-0.1, max=0.1)
        input_data_label_perturb = input_data_original.data + self.eps_d * grad_label

        input_data_original = input_data_original.detach()

        # Update Label Classifier
        if self.cfg.TRAIN.OP == "None":
            label_loss_original = F.cross_entropy(self.label_classifier(input_data_original), class_label)
            label_loss_domain_perturb = F.cross_entropy(self.label_classifier(input_data_domain_perturb), class_label)
            label_loss = (1 - self.alpha_l) * label_loss_original + self.alpha_l * label_loss_domain_perturb
        else:
            # Activate Sample Selection
            label_loss_original = F.cross_entropy(self.label_classifier(input_data_original), class_label, reduction="none")
            label_loss_domain_perturb = F.cross_entropy(self.label_classifier(input_data_domain_perturb), class_label, reduction="none")
            label_loss = (1 - self.alpha_l) * label_loss_original + self.alpha_l * label_loss_domain_perturb
            if self.cfg.TRAIN.OP == "SPL":
                # Self-Paced Learning
                current_lmda = math.ceil(((self.current_epoch + 1) / self.cfg.OPTIM.MAX_EPOCH) * 10) / 10
                self_paced_learning = SelfPacedLearning(lmda=current_lmda)
                label_loss = self_paced_learning(label_loss, None, difficulty_type="loss")
            elif self.cfg.TRAIN.OP == "SSPL":
                # Activate Symmetric Self-Paced Learning
                symmetric_self_paced_learning = SymmetricSelfPacedLearning(self.cfg, self.current_epoch)
                label_loss = symmetric_self_paced_learning(label_loss, None, difficulty_type="loss")
            else:
                raise NotImplementedError

        self.model_backward_and_update(label_loss, "label_classifier")

        # Update Domain Classifier
        domain_loss_original = F.cross_entropy(self.domain_classifier(input_data_original), domain_label)
        domain_loss_label_perturb = F.cross_entropy(self.domain_classifier(input_data_label_perturb), domain_label)
        domain_loss = (1 - self.alpha_d) * domain_loss_original + self.alpha_d * domain_loss_label_perturb
        self.model_backward_and_update(domain_loss, "domain_classifier")

        loss_summary = {
            "label_loss": label_loss.item(),
            "domain_loss": domain_loss.item()
        }

        if self.batch_index + 1 == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_data):
        input_data = batch_data["img"].to(self.device)
        input_data.requires_grad = True
        class_label = batch_data["class_label"].to(self.device)
        domain_label = batch_data["domain_label"].to(self.device)
        return input_data, class_label, domain_label

    def model_inference(self, input_data):
        return self.label_classifier(input_data)



