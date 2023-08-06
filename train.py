import tensorflow as tf
import numpy as np
from date_deal import data_read
import os
from tqdm import tqdm
from Co2_model import co2_model


data_dir = r"G:\project\co2_predict\co2\data\train.csv"

batch_size = 128
epochs = 1000000
c = 256
lr = 0.00001

summary_writer = tf.summary.create_file_writer(r".\log")

checkpoint_dir = './ckpt'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

best_weights_checkpoint_path_d = os.path.join(checkpoint_dir, 'best_d.hd5')

# 导入模型
#model = UNet(num_classes=c) # to use Unet
#model = UNet_Sp(num_classes=c) # to use Unet++

model = co2_model(k_num=c)
model.load_weights(r"G:\project\co2_predict\co2\ckpt\best_d.hd5")
# model.load_weights(r"D:\pj\co2\ckpt\best_d.hd5")

# 创建数据集
train_dataset,test_dataset = data_read(csv_file_path=data_dir,batch_size=batch_size)

#使用余弦退火降低学习率
class CosineAnnealingSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_max, lr_min, T):
        super(CosineAnnealingSchedule, self).__init__()
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T = T

    def __call__(self, step):

        t = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos((step/self.T) * np.pi))
        return t

def loss_fd(p_label,real_label):
    loss = tf.reduce_mean(tf.square(p_label-real_label))

    return loss

sp = 1
lr_schedule = CosineAnnealingSchedule(lr_max=0.00001, lr_min=0.0000001, T=epochs)
total_train_steps = len(train_dataset)  # 训练集中的总步数
total_test_steps = len(test_dataset)  # 训练集中的总步数
#dataset_iter = iter(dataset)  # 创建数据集迭代器

for epoch in range(epochs):
    loss_temp = []
    dataset_iter = iter(train_dataset)
    pbar = tqdm(range(total_train_steps), desc=f"Epoch {epoch+1}")
    for step in pbar:
        train_data, train_label = next(dataset_iter)  # 从迭代器中获取下一个批次数据
        with tf.GradientTape() as tape:
            p_label = model(train_data)
            loss = loss_fd(p_label, train_label)
        gp = tape.gradient(loss, model.trainable_variables)
        #tf.keras.optimizers.RMSprop(learning_rate=lr, momentum=0.9, epsilon=1e-7).apply_gradients(zip(gp, model.trainable_variables))
        tf.keras.optimizers.Adam(learning_rate=lr,beta_1=0.9,beta_2=0.99,epsilon=1e-8).apply_gradients(zip(gp, model.trainable_variables))
        #tf.keras.optimizers.SGD(learning_rate=lr).apply_gradients(zip(gp, model.trainable_variables))
        loss_temp.append(loss)

        with summary_writer.as_default():
            tf.summary.scalar('train_loss', loss, step=sp)
            pbar.set_postfix({"Loss": float(loss), "Step": sp})
            if sp % 100 == 0:
                model.save_weights(best_weights_checkpoint_path_d)
            sp += 1
    print(epoch, "train_loss:", float(np.mean(np.array(loss_temp))))
    del dataset_iter

    if epoch % 10 == 0:
        #测试集
        loss_test_temp = []
        dataset_test_iter = iter(test_dataset)
        pbar_test = tqdm(range(total_test_steps), desc=f"Epoch {epoch + 1}")
        for step in pbar_test:
            test_data, test_label = next(dataset_test_iter)  # 从迭代器中获取下一个批次数据
            p_test_label = model(test_data)
            test_loss = loss_fd(p_test_label, test_label)
            loss_test_temp.append(test_loss)
        with summary_writer.as_default():
            tf.summary.scalar('test_loss', tf.reduce_mean(test_loss), step=epoch)

        print("train_loss:",float(np.mean(np.array(loss_temp))),"test_loss:",float(np.mean(np.array(loss_test_temp))))

        del dataset_test_iter














