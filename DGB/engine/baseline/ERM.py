import copy
import math
from torch.nn import functional as F
from DGB.model.ops import SelfPacedLearning, SymmetricSelfPacedLearning
from DGB.engine import TRAINER_REGISTRY, GenericTrainer
from DGB.utils import compute_top_k_accuracy


@TRAINER_REGISTRY.register()
class ERM(GenericTrainer):
    """
    ERM (Empirical Risk Minimization)

    """

    def forward_backward(self, batch_data):
        input_data, class_label = self.parse_batch_train(batch_data)
        output, representations = self.model(input_data, return_feature=True)

        if self.cfg.TRAIN.OP == "None":
            loss = F.cross_entropy(output, class_label)
        else:
            # Activate Sample Selection
            loss = F.cross_entropy(output, class_label, reduction="none")

            temp_input = copy.deepcopy(input_data)
            temp_input.requires_grad = True
            temp_model = copy.deepcopy(self.model)
            temp_output = temp_model(temp_input)
            temp_loss = F.cross_entropy(temp_output, class_label)
            temp_loss.backward()

            if self.cfg.TRAIN.OP == "SPL":
                # Self-Paced Learning
                current_lmda = math.ceil(((self.current_epoch + 1) / self.cfg.OPTIM.MAX_EPOCH) * 10) / 10
                self_paced_learning_scheduler = SelfPacedLearning(lmda=current_lmda)
                loss = self_paced_learning_scheduler(loss=loss, gradients=temp_input.grad, difficulty_type="gradients")
            elif self.cfg.TRAIN.OP == "SSPL":
                # Symmetric Self-Paced Learning
                symmetric_self_paced_learning_scheduler = SymmetricSelfPacedLearning(self.cfg, self.current_epoch)
                loss, example_difficulty = symmetric_self_paced_learning_scheduler(loss=loss, gradients=temp_input.grad, difficulty_type="gradients")
            else:
                raise NotImplementedError

        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_top_k_accuracy(output, class_label)[0].item()
        }

        if self.batch_index + 1 == self.num_batches:
            self.update_lr()

        # return loss_summary, representations, batch_data["domain_label"]
        return loss_summary

    def parse_batch_train(self, batch_data):
        input_data = batch_data["img"].to(self.device)
        class_label = batch_data["class_label"].to(self.device)
        return input_data, class_label
