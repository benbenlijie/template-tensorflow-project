import tensorflow as tf

from data_loader import *
from models.example_model import ExampleModel
from trainers.example_trainer import ExampleTrainer
from utils.config import process_config
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    config.num_epochs = None
    image_loader = ImageDataLoader(config, True)

    model = ExampleModel(config, image_loader)
    model.init_train_model()
    with tf.Session() as sess:
        trainer = ExampleTrainer(sess, model, config)
        trainer.train()


if __name__ == '__main__':
    main()
