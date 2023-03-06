import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
import seaborn as sns
from sklearn.preprocessing import StandardScaler

class utils:
  
  def __init__(self):
    pass
  
  def dtw_distance(self, attributes):
    dtw_distances = []
    for i in range(attributes.shape[1]):
        distance, path = fastdtw(attributes.iloc[:,-1], attributes.iloc[:, i])
        dtw_distances.append(distance)

    dtw_results = pd.DataFrame({"attribute": attributes.columns, "distance": dtw_distances})

    dtw_results.sort_values(by="distance", inplace=True)

    return dtw_results
  
  def correlation_matrix(self, dataset):
    corr_matrix = dataset.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()

  def sliding_window(self, df, win_size):
    arr = df.to_numpy()
    x=[]
    y=[]

    for i in range(len(arr)-win_size):
        row = [[a] for a in arr[i:i+win_size]]
        x.append(row)
        label = arr[i+win_size]
        y.append(label)
    x = np.array(x)
    x = x.reshape(x.shape[0], x.shape[1], 8)
    y = np.array(y)
    y = y[:, -1]
    return x, y
  
  def upsample(self, source):
        data = source.resample('D')
        data = data.interpolate(method='linear')
        return data
    
  def scale_features(self, df, column_name):
    sc = StandardScaler()
    df[column_name] = sc.fit_transform(df[[column_name]])
    return sc
    
  def to_dataframe(self, d, col_names):
      self.attributes = {}
      for i in range(0, len(d)):
          self.attributes[col_names[i]] = d[i]

      self.attributes = pd.DataFrame(self.attributes)
      self.attributes = self.attributes.dropna(axis=0, how='any')
      return self.attributes
  
  def extra_features(self, oil_prices):
     lag_1 = oil_prices['Value'].shift(1)
     sales_diff = oil_prices['Value'].diff(1)
     mean_last_4 = oil_prices['Value'].rolling(4).mean()

     return lag_1, sales_diff, mean_last_4


  def results(self, pred, y):
    fig, ax = plt.subplots(figsize=(10,10))
    train_results = pd.DataFrame({'Predictions':pred, 'y true': y})
    ax.plot(train_results['Predictions'][200:229], label='predictions')
    ax.plot(train_results['y true'][200:229], label='actual')
    ax.legend()
    plt.show()
  
  def scale_test(self, df, col_name, sc):
     df[col_name] = sc.transform(df[[col_name]])
  
  def confidence_bounds(self, pred):
    bounds = [pred.mean() - 2*pred.std(), pred.mean() + 2*pred.std()]
    return bounds
  


  