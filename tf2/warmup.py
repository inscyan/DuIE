import math

import matplotlib.pyplot as plt
import tensorflow as tf


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, total_steps, warmup, power=-1.5):
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup if isinstance(warmup, int) else int(math.floor(warmup * total_steps))
        self.power = power
        super(CustomSchedule, self).__init__()

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** self.power)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class LinearDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup):
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup if isinstance(warmup, int) else int(math.floor(warmup * total_steps))
        super(LinearDecayWithWarmup, self).__init__()

    def __call__(self, step):
        if step < self.warmup_steps:
            rate = float(step) / float(max(1, self.warmup_steps))
        else:
            rate = max(0.0, float(self.total_steps - step) /
                       float(max(1, self.total_steps - self.warmup_steps)))

        return self.base_lr * rate


# # optimizer
# learning_rate = CustomSchedule(d_model)
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

if __name__ == '__main__':
    total_steps = 100000
    warmup_steps = 0.05

    custom_schedule = CustomSchedule(768, total_steps, warmup_steps)
    linear_decay_with_warmup = LinearDecayWithWarmup(0.001, total_steps, warmup_steps)

    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    lrs = []
    for step in tf.range(1, total_steps + 1, dtype=tf.float32):
        lrs.append(custom_schedule(step))
    plt.plot(lrs)
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.title('custom_schedule')

    plt.subplot(122)
    lrs = []
    for step in tf.range(1, total_steps + 1, dtype=tf.float32):
        lrs.append(linear_decay_with_warmup(step))
    plt.plot(lrs)
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.title('linear_decay_with_warmup')

    plt.show()
