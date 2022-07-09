import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import tensorflow.keras as keras
import argparse

#定义自编码器模型
class Autoencoder(Model):
  #初始化模型所需要的层
  def __init__(self, input_dim,latent_dim):
    super(Autoencoder, self).__init__()
    #输入数据维度
    self.input_dim=input_dim
    #隐藏层维度
    self.latent_dim = latent_dim
    #编码器
    self.encoder = tf.keras.Sequential([
      #输入层
      layers.Input(shape=(self.input_dim,),name='inputs'),
      #隐藏层
      layers.Dense(units=self.latent_dim, activation='sigmoid',name='hidden'),

    ])
    #解码器
    self.decoder = tf.keras.Sequential([
      #输出层
      layers.Dense(units=self.input_dim, activation='sigmoid',name='outputs'),
    ])
  #模型调用
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

#训练数据和测试数据获取
def get_data():
  #使用keras.datasets载入mnist数据集
  (train_data, _), (test_data, _) = keras.datasets.mnist.load_data()
  #调整train_data的数据维度
  train_data = train_data.reshape((-1, 28 * 28)) / 255.0
  #调整test_data的数据维度
  test_data = test_data.reshape((-1, 28 * 28)) / 255.0

  return train_data,test_data

#绘制原始图像和重构图像
def draw(test_data,decoded_imgs,n,pic_name):
  plt.figure(figsize=(10, 4))
  for i in range(n):
    # 绘制原始图像
    ax = plt.subplot(2, n, i + 1)

    plt.imshow(test_data[i].reshape(28,28),cmap='Greys')
    plt.title("Image {}".format(i + 1))

    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 绘制重构图像
    ax = plt.subplot(2, n, i + 1 + n)

    plt.imshow(decoded_imgs[i].reshape(28,28),cmap='Greys')
    plt.title("Image {}".format(i+1+n))

    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  # 绘制到图片文件'autoencoder.png'
  plt.savefig('{}'.format(pic_name))

def main(params):
  # 训练数据和测试数据
  train_data, test_data = get_data()
  # 调用自编码器模型
  autoencoder = Autoencoder(train_data.shape[1], params.latent_dim)
  # 模型训练使用的优化器以及损失函数
  autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
  # 模型训练
  autoencoder.fit(train_data, train_data,
                  epochs=params.epochs,
                  shuffle=True,
                  validation_data=(test_data, test_data))
  # 中间层编码表示
  encoded_embedding = autoencoder.encoder(test_data).numpy()
  # 重构图像
  decoded_imgs = autoencoder.decoder(encoded_embedding).numpy()
  # 定义展示的原始图像个数
  n = 5
  # 绘制结果
  draw(test_data, decoded_imgs, n, 'autoencoder.png')

#主函数
if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs',default=10,type=int)
  parser.add_argument('--latent_dim',default=256,type=int)
  params = parser.parse_args()
  main(params)

