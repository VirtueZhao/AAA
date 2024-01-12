# Adaptive Adversarial Augmentation for Domain Generalization with Ambiguous Semantics

We introduce Adaptive Adversarial Augmentation (AAA) to address this issue, which increases data quantity and diversity by utilizing a dual adversarial network structure to augment source data with strategically generated perturbations.
Unlike conventional methods with static perturbation factors, AAA introduces a dynamic factor that adjusts perturbations according to the diversity of learned embeddings, offering a more flexible augmentation strategy.

# Baseline
* ERM
* SimpleAug (Random Erasing, Random Rotation, ColorJitter)
* CrossGrad - [Generalizing Across Domains via Cross-Gradient Training](https://openreview.net/forum?id=r1Dx7fbCW)
* DDAIG - [Deep Domain-Adversarial Image Generation for Domain Generalisation](https://arxiv.org/abs/2003.06054)
* DomainMix - [Dynamic Domain Generalization](https://arxiv.org/abs/2205.13913)
* MixStyle - [Domain Generalization with MixStyle](https://openreview.net/forum?id=6xHJ37MVxxp)
* EFDMix - [Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization](https://arxiv.org/abs/2203.07740)

# Datasets
* Digits
* PACS
* OfficeHome
* VLCS
* NICO++

# Sample Command

python train.py

                --gpu 0                                                 # Specify device
                --seed 42                                               # Random Seed
                --source_domains cartoon photo sketch                   # Source Domains
                --target_domain art_painting                            # Target Domain
                --dataset_path datasets                                 # Path of datasets
                --output_dir output/PACS                                # Output directory 
                --max_epoch 100                                         # Training Epoch
                --batch_size 256                                        # Batch Size
                --lr 0.0001                                             # Learning Rate 
                --config_path_trainer configs/trainers/AAA/pacs.yaml    # config file for trainer
                --config_path_dataset configs/datasets/pacs.yaml        # config file for dataset
                --trainer AAA                                           # Algorithm for training
