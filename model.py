from keras.layers import Dense, CuDNNLSTM
from keras.models import Sequential

class model:
  
  def __init__(self, lstm_units, loss, optim):
    self.lstm_units = lstm_units
    self.loss = loss
    self.optim = optim

  def create_model(self, x):
    model = Sequential()

    model.add(CuDNNLSTM(self.lstm_units, input_shape=(x.shape[1], x.shape[2]) , return_sequences=True))
    model.add(CuDNNLSTM(self.lstm_units))

    model.add(Dense(1, activation='linear'))

    model.compile(loss=self.loss, optimizer=self.optim)

    return model


