import argparse
from resnet50_classifier import ResNet50_Classifier
from utils import get_config_from_yaml

def main(args=None):
    '''
        Usage:
            run a test (on genie)
            >> cd /home/hhg/Research/galaxyClassify/repo/galaxyZooNet/
            >> python3 galaxyZooNet/train_resnet50.py --config configs/resnet50_test.yaml > test.log
    '''
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
                        help='config file for training settings')

    opt = parser.parse_args()
    print(opt)

    config = get_config_from_yaml(opt.config)
    classifier = ResNet50_Classifier(config=config)
    classifier.train()

if __name__ == '__main__':
    main()
