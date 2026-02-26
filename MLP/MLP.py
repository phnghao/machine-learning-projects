import numpy as np
import unittest
import sklearn
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from tqdm import tqdm

# Setup Layer base
class Layer:
    def __init__(self):
        self.training = True

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        pass

    def backward(self, grad):
        pass

    def parameters(self):
        return []
    
    def grads(self):
        return []
    
    def train(self):
        self.training = True

    def eval(self):
        self.training = False

# Linear Layer 
class Linear(Layer):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.weights = np.random.randn(n_in, n_out) * np.sqrt(2/n_in)
        self.bias = np.zeros(n_out)

        self.d_weights = np.zeros_like(self.weights)
        self.d_bias = np.zeros_like(self.bias)

    def forward(self, x):
        self.x = x
        out = np.matmul(x, self.weights) + self.bias
        return out
    
    def backward(self, grad):
        self.d_weights = np.matmul(self.x.T, grad)
        self.d_bias = np.sum(grad, axis = 0)
        dx = np.matmul(grad, self.weights.T)
        return dx
    
    def parameters(self):
        return [self.weights, self.bias]
    
    def grads(self):
        return [self.d_weights, self.d_bias]

class TestLinear(unittest.TestCase):
    def test_init(self):
        linear = Linear(3, 5)
        self.assertEqual(linear.weights.shape, (3,5))
        self.assertEqual(linear.bias.shape, (5,))

    def test_forward(self):
        linear = Linear(2, 3)
        linear.weights = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        linear.bias = np.array([10, 20, 30])
        x = np.array([
            [1, 2]
        ])
        out = linear(x)
        expected = np.array([
            [1*1 + 2*4 + 10,
             1*2 + 2*5 + 20,
             1*3 + 2*6 + 30]
        ])
        np.testing.assert_array_equal(out, expected)

    def test_backward_shape(self):
        layer = Linear(3,5)
        x = np.random.randn(10,3)
        grad = np.random.randn(10,5)
        layer(x)
        dx = layer.backward(grad)

        self.assertEqual(dx.shape,(10,3))
        self.assertEqual(layer.d_weights.shape,(3,5))
        self.assertEqual(layer.d_bias.shape,(5,))

    def test_backward_value(self):
        linear = Linear(2, 3)
        linear.weights = np.ones((2, 3))
        linear.bias = np.zeros(3)

        x = np.array([[1, 2]])
        linear.forward(x)
        grad = np.array([[1, 1, 1]])
        dx = linear.backward(grad)

        np.testing.assert_array_equal(linear.d_weights, np.array([[1, 1, 1], [2, 2, 2]]))
        np.testing.assert_array_equal(linear.d_bias, np.array([1, 1, 1]))
        np.testing.assert_array_equal(dx, np.array([[3, 3]]))

# ReLU Activation Layer
class ReLU(Layer):
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)
    
    def backward(self, grad):
        return grad * (self.x > 0 )
    
class TestReLU(unittest.TestCase):
    def test_forward(self):
        relu = ReLU()
        x = np.array([-1, 2, 5, -9])
        out = relu(x) 
        np.testing.assert_array_equal(out, np.array([0, 2, 5, 0]))
    
    def test_backward(self):
        relu = ReLU()
        x = np.array([-2, -1, 0, 1, 2])
        relu(x)
        grad = np.ones(5)
        dx = relu.backward(grad)
        np.testing.assert_array_equal(dx, np.array([0, 0, 0, 1, 1]))


# Sequential Layer
class Sequential(Layer):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
    
    def backward(self, grad):
        for layer in self.layers[::-1]:
            grad = layer.backward(grad)
        return grad
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params
    
    def grads(self):
        grds = []
        for layer in self.layers:
            grds += layer.grads()

        return grds
    
    def train(self):
        for layer in self.layers:
            layer.train()

    def eval(self):
        for layer in self.layers:
            layer.eval()
    
class TestSequential(unittest.TestCase):
    def test_forward(self):
        model = Sequential(Linear(5, 4), Linear(4, 3), Linear(3, 1))
        x = np.random.randn(10, 5)
        out = model(x)
        self.assertEqual(out.shape, (10, 1))

    def test_backward(self):
        model = Sequential(Linear(5, 4), Linear(4, 3), Linear(3, 1))
        x = np.random.randn(10, 5)
        out = model(x)
        grad = np.random.randn(10, 1)
        dx = model.backward(grad)
        self.assertEqual(dx.shape, (10, 5))

class CrossEntropy:
    @staticmethod
    def stable_softmax(x):
        exp = np.exp(x - np.max(x, axis = 1, keepdims = True))
        return exp/np.sum(exp, axis = 1, keepdims = True)
    
    def __call__(self, y_true, logits):
       return self.forward(y_true, logits)
    
    def forward(self, y_true, logits):
        samples = logits.shape[0]

        self.y_true = y_true
        self.probs = self.stable_softmax(logits)
        correct_probs = self.probs[np.arange(samples), y_true]
        correct_probs = np.clip(correct_probs, 1e-9, None)
        return -np.mean(np.log(correct_probs))

    def backward(self):
        samples = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(samples), self.y_true] -= 1
        return grad

class GD:
    def __init__(self, model, learning_rate = 0.01):
        self.model = model
        self.learning_rate = learning_rate

    def parameters(self):
        return self.model.parameters()
    
    def grads(self):
        return self.model.grads()
    
    def zero_grad(self):
        for g in self.grads():
            g.fill(0)

    def step(self):
        for p,g in zip(self.parameters(), self.grads()):
            p -= self.learning_rate*g
    
def prepare_data():
  X, y = datasets.load_digits(return_X_y = True)
  X, Xtest, y, ytest = model_selection.train_test_split(X, y, test_size=0.33, random_state=42)
  transform = preprocessing.MinMaxScaler()
  X = transform.fit_transform(X)
  Xtest = transform.transform(Xtest)
  return X, Xtest, y, ytest

def prepare_trainer_GD(model):
  optim = GD(model = model, learning_rate = 0.001)
  loss_fn = CrossEntropy()
  return optim, loss_fn

def prepare_data_loader(X, y, batch_size):
  n = X.shape[0]
  permutation = np.random.permutation(n)
  for i in range(0, n, batch_size):
    j = i+batch_size if i+batch_size <= n else n
    batch_x = X[permutation[i:j]]
    batch_y = y[permutation[i:j]]
    yield batch_x, batch_y

class Net2:
  def __init__(self):
    self.layers = self.make_layers()


  def __call__(self, x):
    return self.forward(x)

  def make_layers(self):
    return Sequential(
        Linear(64, 1024),
        ReLU(),
        Linear(1024, 512),
        ReLU(),
        Linear(512, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 64),
        ReLU(),
        Linear(64, 10)
    )

  def forward(self, x):
    return self.layers.forward(x)

  def backward(self, grad):
    return self.layers.backward(grad)

  def parameters(self):
    return self.layers.parameters()

  def grads(self):
    return self.layers.grads()

  def train(self):
    self.layers.train()

  def eval(self):
    self.layers.eval()

def train_model(model, X, y, optimizer, loss_fn, epochs = 10, batch_size = 32):
  pbar = tqdm(range(epochs))
  for epoch in pbar:
    model.train()
    total_loss = 0
    total_samples = 0

    for batch_x, batch_y in prepare_data_loader(X, y, batch_size):

      # forward
      outputs = model(batch_x)
      loss = loss_fn(batch_y, outputs)

      optimizer.zero_grad()
      # backward
      grad = loss_fn.backward()
      model.backward(grad)

      optimizer.step()

      total_loss += loss*batch_x.shape[0]
      total_samples += batch_x.shape[0]

    pbar.set_description(f'Epoch {epoch + 1}| loss {total_loss/ total_samples:.4f}')


def evaluate_model(model, X, y, loss_fn):
  model.eval()

  outputs = model(X)
  loss = loss_fn(y, outputs)

  preds = np.argmax(outputs, axis =1 )
  acc = np.mean(preds == y)
  print(f'Eval loss {loss:.4f}| Acc: {acc:.4f}')

def train():
  X, Xtest, y, ytest = prepare_data()

  print('Model with ReLU activation using Gradient Descent Optimizer')
  model = Net2()
  optimizer, loss_fn = prepare_trainer_GD(model)
  train_model(model, X, y, optimizer, loss_fn,
            epochs=200,
            batch_size=64)
  evaluate_model(model, Xtest, ytest, loss_fn)

if __name__ =='__main__':
    unittest.main(exit = False)
    train()