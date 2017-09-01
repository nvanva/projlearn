import tensorflow as tf

class Baseline:
    """
    A simple baseline that estimates the projection matrix W
    given the vectors X and Y without any constraints.
    """
    def __init__(self, x_size, y_size, w_stddev, **kwargs):
        self.x_size   = x_size
        self.y_size   = y_size
        self.w_stddev = w_stddev

        self.X = tf.placeholder(tf.float32, shape=[None, self.x_size], name='X')
        self.Y = tf.placeholder(tf.float32, shape=[None, self.y_size], name='Y')
        self.Z = tf.placeholder(tf.float32, shape=[None, self.x_size], name='Z')
        self.W = tf.Variable(tf.random_normal((self.x_size, self.y_size), stddev=self.w_stddev), name='W')

        self.Y_hat   = tf.matmul(self.X, self.W)
        self.Y_error = self.Y_hat - self.Y
        self.Y_norm  = self.l2_norm(self.Y_error)

        self.Y_loss  = tf.nn.l2_loss(self.Y_norm)
        self.loss    = self.Y_loss

        tf.summary.histogram('W', self.W)
        tf.summary.histogram('y_hat_norm', tf.norm(self.Y_hat, axis=1))
        tf.summary.scalar('LOSS', self.loss)
        self.summary = tf.summary.merge_all()
        


    def __str__(self):
        return '<%s>' % self.__class__.__name__

    def l2_norm(self, t, name='l2_norm_op'):
        with tf.name_scope(name) as scope:
            l2_norm_op = tf.sqrt(tf.reduce_sum(tf.square(t), 1), name=scope)
            return l2_norm_op


