import copy
import math
from torch.nn import functional as F
from DGB.model.ops import SimpleAugOP, SelfPacedLearning, SymmetricSelfPacedLearning
from DGB.engine import TRAINER_REGISTRY, GenericTrainer
from DGB.utils import compute_top_k_accuracy


@TRAINER_REGISTRY.register()
class SimpleAug(GenericTrainer):
    """
    ERM (Empirical Risk Minimization)

    """

    def forward_backward(self, batch_data):
        input_data_original, class_label = self.parse_batch_train(batch_data)

        simple_aug = SimpleAugOP(aug_type="ColorJitter")
        input_data_augmented = simple_aug(input_data_original)
        output_original = self.model(input_data_original)
        output_augmented = self.model(input_data_augmented)

        if self.cfg.TRAIN.OP == "None":
            loss_original = F.cross_entropy(output_original, class_label)
            loss_augmented = F.cross_entropy(output_augmented, class_label)
            loss = 0.5 * loss_original + 0.5 * loss_augmented
        else:
            # Activate Sample Selection
            loss_original = F.cross_entropy(output_original, class_label, reduction="none")
            loss_augmented = F.cross_entropy(output_augmented, class_label, reduction="none")
            loss = 0.5 * loss_original + 0.5 * loss_augmented

            # temp_input_original = copy.deepcopy(input_data_original)
            # temp_input_augmented = copy.deepcopy(input_data_augmented)
            # temp_input_original.requires_grad = True
            # temp_model = copy.deepcopy(self.model)
            # temp_output_original = temp_model(temp_input_original)
            # temp_output_augmented = temp_model(temp_input_augmented)
            # temp_loss_original = F.cross_entropy(temp_output_original, class_label)
            # temp_loss_augmented = F.cross_entropy(temp_output_augmented, class_label)
            # temp_loss = 0.5 * temp_loss_original + 0.5 * temp_loss_augmented
            # temp_loss.backward()

            if self.cfg.TRAIN.OP == "SPL":
                # Self-Paced Learning
                current_lmda = math.ceil(((self.current_epoch + 1) / self.cfg.OPTIM.MAX_EPOCH) * 10) / 10
                self_paced_learning = SelfPacedLearning(lmda=current_lmda)
                loss = self_paced_learning(loss, None, difficulty_type="loss")
            elif self.cfg.TRAIN.OP == "SSPL":
                # Symmetric Self-Paced Learning
                symmetric_self_paced_learning = SymmetricSelfPacedLearning(self.cfg, self.current_epoch)
                loss = symmetric_self_paced_learning(loss, None, difficulty_type="loss")
            else:
                raise NotImplementedError

        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_top_k_accuracy(output_original, class_label)[0].item()
        }

        if self.batch_index + 1 == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_data):
        input_data = batch_data["img"].to(self.device)
        class_label = batch_data["class_label"].to(self.device)
        return input_data, class_label
