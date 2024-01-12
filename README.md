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
* CUHK03 - [DeepReID: Deep Filter Pairing Neural Network for Person Re-identification](https://ieeexplore.ieee.org/document/6909421)
* Market1501 - [Scalable Person Re-identification: A Benchmark](https://ieeexplore.ieee.org/document/7410490)
* MSMT17 - [Person Transfer GAN to Bridge Domain Gap for Person Re-identification](https://ieeexplore.ieee.org/document/8578114)

# Sample Command

python train.py

                --gpu 0                               # Specify device
                --seed 42                             # Random Seed
                --max_epoch 100                       # Training Epoch
                --lr 0.0001                           # Learning Rate 
                --batch_size 256                      # Batch Size
                --dataset_path datasets               # Path of datasets
                --output_dir output/MSMT17_CUHK03     # Output directory 
                --trainer AAA                         # Algorithm for training
                --source_domains MSMT17               # Source dataset
                --target_domain CUHK03                # Target dataset
