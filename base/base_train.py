import tensorflow as tf
import time


class BaseTrain:
    def __init__(self, sess, model, config):
        self.sess = sess
        self.model = model
        self.config = config
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)

    def train(self):
        tf.logging.info("Start to Train")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():
                start_time = time.time()
                self.train_step()
                elapsed_time = time.time() - start_time
                self.log_step(elapsed_time)

        except tf.errors.OutOfRangeError as e:
            tf.logging.info("Train finished")
        finally:
            self.model.save(self.sess)
            coord.request_stop()
        coord.join(threads)

    def train_step(self):
        raise NotImplementedError

    def log_step(self, elapsed_time=0):
        raise NotImplementedError
