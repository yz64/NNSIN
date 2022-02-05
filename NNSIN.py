# https://qiita.com/kazu-ojisan/items/fe2d3308b3ba56a80c7a#%E7%B5%90%E6%9E%9C
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy
import os

# N=学習データ数, Nte=テストデータ数
N = 1000
Nte = 200
batch = 10
epoch = 200

def getSIN():
  x = numpy.linspace(0, 2*numpy.pi, N+Nte)
  ram = numpy.random.permutation(N+Nte)
  x_train = numpy.sort(x[ram[:N]])
  x_test = numpy.sort(x[ram[N:]])
  y_train = numpy.sin(x_train)
  y_test = numpy.sin(x_test)
  return x_train, y_train, x_test, y_test

class NN_SIN(torch.nn.Module):
  def __init__(self):
      super().__init__()
      self.fc1 = torch.nn.Linear(1, 10)
      self.fc2 = torch.nn.Linear(10, 10)
      self.fc3 = torch.nn.Linear(10, 1)
  def forward(self, x):
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)
      x = F.relu(x)
      x = self.fc3(x)
      return x
  def predict(self, x):
      x = torch.from_numpy(x.astype(numpy.float32).reshape(x.shape[0],1))
      y = self.forward(x)
      return y.data

if __name__ == '__main__':

  # ディレクトリ作成
  if os.path.exists("./epoch") == False:
    os.makedirs("./epoch")
   
  x_train, y_train, x_test, y_test = getSIN()

  train_loss = []
  test_loss = []

  net = NN_SIN()
  optimizer = torch.optim.Adam(params=net.parameters())

  for e in range(1, epoch+1):
    sum_loss = 0
    perm = numpy.random.permutation(N)
    net.train(True)
    for i in range(0, N, batch):
      x_batch = x_train[i:i+batch]
      y_batch = y_train[i:i+batch]
      x_batch_torch = torch.from_numpy(x_batch.astype(numpy.float32).reshape(x_batch.shape[0],1))
      y_batch_torch = torch.from_numpy(y_batch.astype(numpy.float32).reshape(y_batch.shape[0],1))

      optimizer.zero_grad()
      x_batch_torch = net(x_batch_torch)
      loss = F.mse_loss(x_batch_torch, y_batch_torch)
      loss.backward()
      optimizer.step()
      sum_loss += loss.data * batch
    ave_loss = sum_loss / N
    train_loss.append(float(ave_loss))

    net.eval()
    x_test_torch = torch.from_numpy(x_test.astype(numpy.float32).reshape(x_test.shape[0],1))
    y_test_torch = torch.from_numpy(y_test.astype(numpy.float32).reshape(y_test.shape[0],1))
    x_test_torch = net(x_test_torch)
    loss = F.mse_loss(x_test_torch, y_test_torch)
    test_loss.append(float(loss))

    if e % 10 == 0:
      print("{:4}/{}  {:10.5}   {:10.5}".format(e, epoch, ave_loss, float(loss.data)))
      # 誤差をグラフ表示
      plt.plot(train_loss, label = "training")
      plt.plot(test_loss, label = "test")
      plt.yscale('log')
      plt.legend()
      plt.grid(True)
      plt.xlabel("epoch")
      plt.ylabel("loss(MSE)")
      plt.pause(0.1)
      plt.clf()
    if e % 20 == 0:
      plt.figure(figsize=(5, 4))
      y_predict = net.predict(x_test)
      plt.plot(x_test, y_test, label = "target")
      plt.plot(x_test, y_predict, label = "predict")
      plt.title('NN_SIN graph(epoch={})'.format(e))
      plt.legend()
      plt.grid(True)
      plt.xlim(0, 2 * numpy.pi)
      plt.ylim(-1.2, 1.2)
      plt.xlabel("x")
      plt.ylabel("y")
      plt.savefig("./epoch/ep{}.png".format(e))
      plt.clf()
      plt.close()
  torch.save(net.state_dict(), 'NNSIN.pt')