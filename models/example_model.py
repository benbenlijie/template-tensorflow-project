import tensorflow as tf

from base.base_model import BaseModel
slim = tf.contrib.slim


class ExampleModel(BaseModel):
    def __init__(self, config, data_loader):
        super(ExampleModel, self).__init__(config, data_loader)

    def _build_train_model(self):
        self.train_op = tf.constant([1, 2, 3], dtype=tf.float16)
        self.loss_op = tf.constant(0.1)
        tf.summary.scalar("test", self.loss_op)
        pass

    def _build_evaluate_model(self):
        pass
