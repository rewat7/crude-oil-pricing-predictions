from data_mining import data
from utils import utils
from model import model

# if __name__ == "__main__":

start_date = '2018-01-01'
end_date = '2021-01-30'
data = data(start_date, end_date)
utils = utils()
model = model(128, 'mse', 'adam')

oil_prices = data.oil_prices()
oil_production = data.oil_production()
oil_reserves = data.oil_reserves()
oil_imports = data.oil_imports()
gold_prices = data.gold_data()

oil_production = utils.upsample(oil_production)
oil_reserves = utils.upsample(oil_reserves)
oil_imports = utils.upsample(oil_imports)

scaler_oilprice = utils.scale_features(oil_prices, 'Value')
scaler_goldprice = utils.scale_features(gold_prices, 'USD (AM)')
scaler_oilprod = utils.scale_features(oil_production, 'Value')
scaler_oilres = utils.scale_features(oil_reserves, 'Value')
scaler_oilimp = utils.scale_features(oil_imports, 'Value')

lag_1, sales_diff, mean_last_4 = utils.extra_features(oil_prices)

cols = [oil_production['Value'], oil_reserves['Value'], oil_imports['Value'], gold_prices['USD (AM)'], lag_1, sales_diff, mean_last_4, oil_prices['Value']]
col_names = ['oil production', 'oil reserves', 'oil import', 'gold prices', 'lag 1', 'sales diff', 'rolling mean 4', 'oil prices']
dataset = utils.to_dataframe(cols, col_names)

utils.correlation_matrix(dataset)

x, y = utils.sliding_window(dataset, 5)

model = model.create_model(x)
model.fit(x, y,  epochs=1000, batch_size=32)

#Testing data
start_date = '2021-01-25'
end_date = '2021-01-31'
test_data = data(start_date, end_date)

oil_prices = test_data.oil_prices()
oil_production = test_data.oil_production()
oil_reserves = test_data.oil_reserves()
oil_imports = test_data.oil_imports()
gold_prices = test_data.gold_data()

oil_production = utils.upsample(oil_production)
oil_reserves = utils.upsample(oil_reserves)
oil_imports = utils.upsample(oil_imports)

utils.scale_test(oil_prices, 'Value', scaler_oilprice)
utils.scale_test(oil_production, 'Value', scaler_oilprod)
utils.scale_test(oil_reserves, 'Value', scaler_oilres)
utils.scale_test(oil_imports, 'Value', scaler_oilimp)
utils.scale_test(gold_prices, 'USD (AM)', scaler_goldprice)

lag_1, sales_diff, mean_last_4 = utils.extra_features(oil_prices)

cols = [oil_production['Value'], oil_reserves['Value'], oil_imports['Value'], gold_prices['USD (AM)'], lag_1, sales_diff, mean_last_4, oil_prices['Value']]
col_names = ['oil production', 'oil reserves', 'oil import', 'gold prices', 'lag 1', 'sales diff', 'rolling mean 4', 'oil prices']
dataset = utils.to_dataframe(cols, col_names)

testing = utils.to_dataframe(cols, col_names)

x, y = utils.sliding_window(testing, 5)

pred = model.predict(x)
pred = scaler_oilprice.inverse_transform(pred)
y = scaler_oilprice.inverse_transform(y.reshape(-1, 1))
utils.results(pred, y)
bounds = utils.confidence_bounds(pred, y)