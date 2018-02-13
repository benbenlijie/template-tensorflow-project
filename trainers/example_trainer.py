from base.base_train import BaseTrain
import sys


class ExampleTrainer(BaseTrain):
    def __init__(self, sess, model, config):
        super(ExampleTrainer, self).__init__(sess, model, config)

    def train_step(self):
        self.sess.run(self.model.train_op)

    def log_step(self, elapsed_time=0):
        loss = self.sess.run(self.model.loss_op)
        sys.stdout.write("total loss {}, secs/step {}".format(loss, elapsed_time))
        sys.stdout.flush()
        summary_str = self.sess.run(self.model.summary_op)
        self.model.summary.add_summary(summary_str)
        self.model.summary.flush()
