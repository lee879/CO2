import tensorflow as tf
from tensorflow.python.keras.layers import Dense,Activation,Dropout,Layer
from tensorflow.python.keras import Model,Sequential

class WeightedFusionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(WeightedFusionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 在build方法中根据输入形状来初始化权重
        self.w1 = self.add_weight(name='weight_1', shape=(1,), initializer='glorot_uniform', trainable=True)
        self.w2 = self.add_weight(name='weight_2', shape=(1,), initializer='glorot_uniform', trainable=True)
        self.w3 = self.add_weight(name='weight_3', shape=(1,), initializer='glorot_uniform', trainable=True)
        super(WeightedFusionLayer, self).build(input_shape)

    def call(self, inputs):
        # 解包输入
        input1, input2, input3 = inputs

        # 将权重进行归一化，使得三个权重加起来等于1
        total_weight = self.w1 + self.w2 + self.w3
        w1_normalized = self.w1 / total_weight
        w2_normalized = self.w2 / total_weight
        w3_normalized = self.w3 / total_weight

        # 加权融合
        fused_output = w1_normalized * input1 + w2_normalized * input2 + w3_normalized * input3
        return fused_output

class block_0(Layer):
    def __init__(self,out_channels,rate=0.3):
        super(block_0, self).__init__()
        self.l = Sequential([
            Dense(out_channels,activation=tf.nn.tanh),
            Dense(out_channels, activation=tf.nn.tanh),
            tf.keras.layers.BatchNormalization(),
            Dropout(rate=rate)
        ])
    def call(self, inputs, *args, **kwargs):

        return self.l(inputs)

class voter(Layer):
    def __init__(self,out_channels):
        super(voter, self).__init__()
        self.vct = Sequential([
            Dense(out_channels, activation=tf.nn.tanh),
            Dense(out_channels, activation=tf.nn.tanh),
            Dense(1)
        ])
    def call(self, inputs, *args, **kwargs):

        return self.vct(inputs)


class co2_model(Model):
    def __init__(self,k_num,ratio = 0.1):
        super(co2_model, self).__init__()
        self.l0_0 = block_0(k_num,ratio)
        self.l0_1 = block_0(k_num,ratio)

        self.l1_0 = block_0(k_num / 2,ratio)
        self.l1_1 = block_0(k_num / 2,ratio)

        self.l2_0 = block_0(k_num / 4,ratio)
        self.l2_1 = block_0(k_num / 4,ratio)

        self.vct1 = voter(k_num / 8)
        self.vct2 = voter(k_num / 8)
        self.vct3 = voter(k_num/ 8)

        self.asff = WeightedFusionLayer()

    def call(self, inputs, training=None, mask=None):
        x = self.l0_0(inputs)
        y = self.l0_1(inputs)

        x1 = self.l1_0(tf.add(x,y))
        y1 = self.l1_1(tf.concat([x,y],axis=-1))

        x2 = self.l2_0(tf.add(x1,y1))
        y2 = self.l2_1(tf.concat([x1,y1],axis=-1))

        z = tf.concat([x2,y2],axis=-1)

        out1 = self.vct1(z)
        out2 = self.vct2(z)
        out3 = self.vct3(z)

        return self.asff([out1,out2,out3])

# m = co2_model(input_channels=256)
# x = tf.random.normal(shape=(125,70))
# y = m(x)
# m.summary()
# print("")