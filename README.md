# A Dual Augmentation Framework for Domain Generalization with both Covariate and Conditional Distribution Shifts

Deep learning suffers performance degradation due to domain shifts arising from discrepancies between training and testing data distributions. Domain Generalization (DG) aims to address this challenge by leveraging knowledge from diverse source domains to bolster model generalization. Two primary types of domain shift are covariate shift and conditional distribution shift, where the distribution shift occurs in the marginal distribution $\mathbb{P}(X)$ and conditional distribution $\mathbb{P}(Y|X)$, respectively. However, most existing methods primarily address covariate shifts while assuming stability in the conditional distribution across source domains. This paper addresses the more general scenario where both $\mathbb{P}(X)$ and $\mathbb{P}(Y|X)$ can vary across all domains, introducing a Dual Augmentation Framework (DAF). DAF addresses covariate and conditional distribution shifts with two augmentation networks, which augment source data with strategically generated perturbations. Furthermore, we introduce a dynamic augmentation strategy that adjusts perturbation weights according to the diversity of learned embeddings to enhance the adaptability and performance of DAF. Experiments with state-of-the-art baselines across five popular DG benchmarks demonstrate the effectiveness of DAF, showcasing its potential in mitigating both covariate and conditional distribution shifts.

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
