from yacs.config import CfgNode as CN

###########################
# Default Configuration
###########################


def get_cfg_default():
    ###########################
    # Config Definition
    ###########################
    _C = CN()

    # Directory to save the output files (like log.txt and model weights)
    _C.OUTPUT_DIR = "./output"
    # Set seed to Negative value to randomize everything; Set seed to Positive value to use a fixed seed
    _C.SEED = -1
    _C.USE_CUDA = True

    ###########################
    # Input
    ###########################
    _C.INPUT = CN()
    _C.INPUT.SIZE = (224, 224)
    # Mode of interpolation in resize functions
    _C.INPUT.INTERPOLATION = "bilinear"
    # For available choices please refer to transforms.py
    _C.INPUT.TRANSFORMS = ()
    # If True, tfm_train and tfm_test will be None
    _C.INPUT.NO_TRANSFORM = False
    # Default mean and std come from ImageNet
    _C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
    _C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]

    ###########################
    # Dataset
    ###########################
    _C.DATASET = CN()
    # Directory where datasets are stored
    _C.DATASET.PATH = ""
    _C.DATASET.NAME = ""
    # List of names of source domains
    _C.DATASET.SOURCE_DOMAINS = ()
    # List of names of target domains
    _C.DATASET.TARGET_DOMAIN = ()

    ###########################
    # Dataloader
    ###########################
    _C.DATALOADER = CN()
    _C.DATALOADER.NUM_WORKERS = 4
    # img0 denotes image tensor without augmentation
    # Useful for consistency learning
    _C.DATALOADER.RETURN_ORIGINAL_IMG = False
    # Setting for the train data_loader
    _C.DATALOADER.TRAIN = CN()
    _C.DATALOADER.TRAIN.SAMPLER = "RandomSampler"
    _C.DATALOADER.TRAIN.BATCH_SIZE = 16
    # Setting for the test data_loader
    _C.DATALOADER.TEST = CN()
    _C.DATALOADER.TEST.SAMPLER = "SequentialSampler"
    _C.DATALOADER.TEST.BATCH_SIZE = 64

    ###########################
    # Model
    ###########################
    _C.MODEL = CN()
    # Path to model weights (for initialization)
    _C.MODEL.INIT_WEIGHTS = ""
    _C.MODEL.BACKBONE = CN()
    _C.MODEL.BACKBONE.NAME = ""
    _C.MODEL.BACKBONE.PRETRAINED = True

    ###########################
    # Optimization
    ###########################
    _C.OPTIM = CN()
    _C.OPTIM.NAME = "SGD"
    _C.OPTIM.LR = 0.05
    _C.OPTIM.WEIGHT_DECAY = 5e-4
    _C.OPTIM.MOMENTUM = 0.9
    _C.OPTIM.SGD_DAMPENING = 0
    _C.OPTIM.SGD_NESTEROV = False
    # Learning rate scheduler
    _C.OPTIM.LR_SCHEDULER = "cosine"
    # -1 or 0 means the stepsize is equal to max_epoch
    _C.OPTIM.STEP_SIZE = -1
    _C.OPTIM.GAMMA = 0.1
    _C.OPTIM.MAX_EPOCH = 25

    ###########################
    # Train
    ###########################
    _C.TRAIN = CN()
    # How often (epoch) to save model during training
    # Set to 0 or negative value to only save the last one
    _C.TRAIN.CHECKPOINT_FREQ = 0
    # How often (batch) to print training information
    _C.TRAIN.PRINT_FREQ = 10

    ###########################
    # Test
    ###########################
    _C.TEST = CN()
    _C.TEST.EVALUATOR = "Classification"
    # Compute confusion matrix, which will be saved
    # to $OUTPUT_DIR/cmat.pt
    _C.TEST.COMPUTE_CMAT = False
    # If NO_TEST=True, no testing will be conducted
    _C.TEST.NO_TEST = False
    # Use test or val set for FINAL evaluation
    _C.TEST.SPLIT = "test"
    # Which model to test after training
    # Either last_step or best_val
    _C.TEST.FINAL_MODEL = "last_step"

    ###########################
    # Trainer specifics
    ###########################
    _C.TRAINER = CN()
    _C.TRAINER.NAME = ""

    # CrossGrad
    _C.TRAINER.CROSSGRAD = CN()  # Generalizing Across Domains via Cross-Gradient Training (ICLR'18)
    _C.TRAINER.CROSSGRAD.EPS_L = 1.0  # scaling parameter for D's gradients
    _C.TRAINER.CROSSGRAD.EPS_D = 1.0  # scaling parameter for F's gradients
    _C.TRAINER.CROSSGRAD.ALPHA_L = 0.5  # balancing weight for the label net's loss
    _C.TRAINER.CROSSGRAD.ALPHA_D = 0.5  # balancing weight for the domain net's loss
    # DDAIG
    _C.TRAINER.DDAIG = CN()  # Deep Domain-Adversarial Image Generation for Domain Generalization (AAAI'20)
    _C.TRAINER.DDAIG.G_ARCH = ""  # generator's architecture
    _C.TRAINER.DDAIG.LMDA = 0.3  # perturbation weight
    _C.TRAINER.DDAIG.CLAMP = False  # clamp perturbation values
    _C.TRAINER.DDAIG.CLAMP_MIN = -1.0
    _C.TRAINER.DDAIG.CLAMP_MAX = 1.0
    _C.TRAINER.DDAIG.WARMUP = 0
    _C.TRAINER.DDAIG.ALPHA = 0.5  # balancing weight for the losses
    # DOMAINMIX
    _C.TRAINER.DOMAINMIX = CN()
    _C.TRAINER.DOMAINMIX.TYPE = "crossdomain"
    _C.TRAINER.DOMAINMIX.ALPHA = 1.0
    _C.TRAINER.DOMAINMIX.BETA = 1.0
    # SPL
    _C.TRAINER.SPL = CN()
    _C.TRAINER.SPL.LMDA = 0.1

    return _C.clone()
