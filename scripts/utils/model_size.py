import yaml
import paddle
import argparse
from paddleseg.cvlibs.builder import SegBuilder

# A minimal config class to support SegBuilder. We only care about the model.
class SimpleConfig:
    def __init__(self, cfg):
        self.model_cfg = cfg.get('model', {})
        # Provide a dummy train_dataset config to bypass dataset-based checks in SegBuilder.model.
        # Using 'Dataset' as type to skip synchronization.
        self.train_dataset_cfg = {
            'type': 'Dataset',
            'img_channels': self.model_cfg.get('backbone', {}).get('in_channels', 3),
            'num_classes': self.model_cfg.get('num_classes', 1)
        }
        # Dummy entries to satisfy other parts of builder if necessary.
        self.optimizer_cfg = {}  
        self.lr_scheduler_cfg = {}
        self.loss_cfg = {}
        self.distill_loss_cfg = {}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model size summary')
    parser.add_argument('--config', type=str, default='configs/config.yml', help='Path to the config YAML file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    config = SimpleConfig(cfg_dict)
    builder = SegBuilder(config)
    model = builder.model
    
    paddle.summary(model, (1, 2, 1024, 1024))