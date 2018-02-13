from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class TemplateTrainer(BaseTrain):
    def __init__(self, sess, model, config):
        super(TemplateTrainer, self).__init__(sess, model, config)

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        """
        pass

    def log_step(self, elapsed_time=0):
        """
        implement log step
        :param elapsed_time:
        :return:
        """
        pass
