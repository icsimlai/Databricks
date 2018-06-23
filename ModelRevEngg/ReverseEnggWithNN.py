import random
import os
import csv
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.io import loadmat
#from pandas import *
data = pd.read_csv(r"E:\Users\Itesh\Work\Vishwa\AzDatBricML\Azure Databricks\Azure Databricks\Reverse Engg\house_data_rev_engg.csv")
data_rev_engg = pd.read_csv(r"E:\Users\Itesh\Work\Vishwa\AzDatBricML\Azure Databricks\Azure Databricks\Reverse Engg\rev_engg_test_data.csv")

#----------------LR---------------------------------------
labels = data['price']
conv_dates = [1 if values == 2014 else 0 for values in data.date ]
data['date'] = conv_dates
print("data.shape", data.shape)
# Split into training, validation, and testing datasets
#train1 = data.drop(['id', 'price'],axis=1)
from sklearn.cross_validation import train_test_split
#x_train_lr , x_test_lr , y_train_lr , y_test_lr = train_test_split(data , labels , test_size = 0.10,random_state =2)
x_train_raw_lr , x_test_raw_lr , y_train_raw_lr , y_test_raw_lr = train_test_split(data , labels , test_size = 0.70,random_state =2)
print("x_train_raw_lr.shape, y_train_raw_lr.shape, x_test_raw_lr.shape, y_test_raw_lr.shape: ", x_train_raw_lr.shape, y_train_raw_lr.shape, x_test_raw_lr.shape, y_test_raw_lr.shape)

x_train_lr = x_train_raw_lr.drop(['id', 'price'],axis=1)
y_train_lr = y_train_raw_lr
x_test_lr = x_test_raw_lr.drop(['id', 'price'],axis=1)
y_test_lr =y_test_raw_lr

print("x_train_lr.shape, y_train_lr.shape, x_test_lr.shape, y_test_lr.shape: ", x_train_lr.shape, y_train_lr.shape, x_test_lr.shape, y_test_lr.shape)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train_lr,y_train_lr)
score_lr = reg.score(x_test_lr,y_test_lr)
print("score_lr",score_lr)
# Predict using LR
y_pred_lr  =  reg.predict(x_test_lr)

from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,
          learning_rate = 0.1, loss = 'ls')
clf.fit(x_train_lr, y_train_lr)
score_gb = clf.score(x_test_lr,y_test_lr)
print("score_gb",score_gb)

original_params = {'n_estimators': 400, 'max_leaf_nodes': None, 'max_depth': 5, 'random_state': None, 'min_samples_split': 2}
params = dict(original_params)
t_sc = np.zeros((params['n_estimators']),dtype=np.float32)

for i,y_pred_gb in enumerate(clf.staged_predict(x_test_lr)):
    t_sc[i]=clf.loss_(y_test_lr,y_pred_gb)

testsc = np.arange((params['n_estimators']))+1
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(testsc,clf.train_score_,'b-',label= 'Set dev train')
plt.plot(testsc,t_sc,'r-',label = 'set dev test')
plt.show()
plt.clf()
# Predict using GB
##X=np.array(x_test_lr)
##y_pred_gb = clf.predict(x_test_lr)
# y=np.array(y_pred2_lr.astype(int), order="F", ndmin=2)

# print(x_test.columns)
output_disp = x_test_lr.drop(['date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'view', 'condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'sqft_living15', 'sqft_lot15'] ,axis=1)
sizeoftestdata = output_disp.shape[0]
df_output_combined = pd.DataFrame(output_disp, columns=['lat', 'long', 'lr_pred_price', 'gb_pred_price'])
df_output_combined['lr_pred_price'] = y_pred_lr
df_output_combined['gb_pred_price'] = y_pred_gb
print(df_output_combined)

#----------------NN---------------------------------------x_train_lr , x_test_lr , y_train_lr , y_test_lr
def cleanup(df):
    '''Cleans data
        1. Creates new features:
            - total bathrooms = full + half bathrooms
            - total porch area = closed + open porch area
        2. Drops unwanted features
        3. Fills missing values with the mode
        4. Performs feature scaling
    '''
    # to_drop = ['MiscFeature', 'MiscVal', 'GarageArea', 'GarageYrBlt', 'Street', 'Alley',
    #           'LotShape', 'LandContour', 'LandSlope', 'RoofMatl', 'Exterior2nd', 'MasVnrType',
    #           'MasVnrArea', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    #           'BsmtFinSF1', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'Electrical',
    #           'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
    #           'HalfBath', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'FireplaceQu',
    #           'GarageType', 'GarageFinish', 'GarageQual', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
    #           'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolQC', 'MoSold']
    
    # df['Bathrooms'] = df['FullBath'] + df['HalfBath']
    # df['PorchSF'] = df['EnclosedPorch'] + df['OpenPorchSF']
    # df = df.drop(to_drop, axis=1)
    
    to_ignore = ['price', 'id', 'date']
    for column in df.columns:
        x = df[column].dropna().value_counts().index[0]
        df = df.fillna(x)
        if df[column].dtype != 'object' and column not in to_ignore:
            m = df[column].min()
            M = df[column].max()
            Range = M - m
            df[column] = (df[column] - m) / Range
    return df

# train_dataset = pd.read_csv(os.path.join(data_dir, 'house_data_train.csv'))
# test_dataset = pd.read_csv(os.path.join(data_dir, 'house_data_test.csv'))
# conv_dates1 = [1 if values == 2014 else 0 for values in train_dataset.date ]
# train_dataset['date'] = conv_dates1

# conv_dates2 = [1 if values == 2014 else 0 for values in test_dataset.date ]
# test_dataset['date'] = conv_dates2

train_dataset = cleanup(x_test_raw_lr)
# test_dataset = cleanup(test_dataset)
# train_dataset, test_dataset = encode_features(train_dataset, test_dataset)

# train_dataset = data.sample(frac=1)
train_dataset['price'] = y_pred_gb
# Split into training, validation, and testing datasets
train, valid, test = np.split(train_dataset, [int(0.8 * len(train_dataset)), int(0.9 * len(train_dataset))])
print("train_dataset.shape, train.shape, valid.shape, test.shape: ", train_dataset.shape, train.shape, valid.shape, test.shape)
# Convert into numpy arrays
x_train = train.drop(['price', 'id'], axis=1).as_matrix().astype(np.float32)
y_train = train['price'].as_matrix().astype(np.float32).reshape((np.shape(x_train)[0], 1))
x_test = test.drop(['price', 'id'], axis=1).as_matrix().astype(np.float32)
y_test = test['price'].as_matrix().astype(np.float32).reshape((np.shape(x_test)[0], 1))
x_valid = valid.drop(['price', 'id'], axis=1).as_matrix().astype(np.float32)
y_valid = valid['price'].as_matrix().astype(np.float32).reshape((np.shape(x_valid)[0], 1))
print("x_train.shape, y_train.shape, x_test.shape, y_test.shape, , x_valid.shape, y_valid.shape : ", x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_valid.shape, y_valid.shape)


num_steps = 1001
def accuracy(prediction, labels):
    return 0.5 * np.sqrt(((prediction - labels) ** 2).mean(axis=None))

def main(self):
    # train_dataset = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    # test_dataset = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    # x_train, y_train, x_test, y_test, x_valid, y_valid, test_dataset = clean(train_dataset, test_dataset)

    train_size = np.shape(x_train)[0]
    valid_size = np.shape(x_valid)[0]
    test_size = np.shape(x_test)[0]
    num_features = np.shape(x_train)[1]
    num_hidden = 16 # Number of activation units in the hidden layer

    # Neural network with one hidden layer
    graph = tf.Graph()
    with graph.as_default():

        # Input
        tf_train_dataset = tf.constant(x_train, dtype=tf.float32)
        tf_train_labels = tf.constant(y_train, dtype=tf.float32)
        tf_valid_dataset = tf.constant(x_valid)
        tf_test_dataset = tf.constant(x_test)

        # Variables
        weights_1 = tf.Variable(tf.truncated_normal([num_features, num_hidden]), dtype=tf.float32, name="layer1_weights")
        biases_1 = tf.Variable(tf.zeros([num_hidden]), dtype=tf.float32, name="layer1_biases")
        weights_2 = tf.Variable(tf.truncated_normal([num_hidden, 1]), dtype = tf.float32, name="layer2_weights")
        biases_2 = tf.Variable(tf.zeros([1]), dtype=tf.float32, name="layer2_biases")
        steps = tf.Variable(0)

        # Model
        def model(x, train=False):
            hidden = tf.nn.relu(tf.matmul(x, weights_1) + biases_1)
            return tf.matmul(hidden, weights_2) + biases_2

        # Loss Computation
        train_prediction = model(tf_train_dataset)
        loss = 0.5 * tf.reduce_mean(tf.squared_difference(tf_train_labels, train_prediction))
        cost = tf.sqrt(loss)

        # Optimizer
        # Exponential decay of learning rate
        learning_rate = tf.train.exponential_decay(0.06, steps, 5000, 0.70, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=steps)

        # Predictions
        valid_prediction = model(tf_valid_dataset)
        test_prediction = model(tf_test_dataset)

        saver = tf.train.Saver()

    # Running graph
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        print('Initialized')
        for step in range(num_steps):
            # Run the computations. We tell .run() that we want to run the optimizer,
            # and get the loss value and the training predictions returned as numpy
            # arrays.
            _, c, predictions = sess.run([optimizer, cost, train_prediction])
            if (step % 5000 == 0):
                print('Cost at step %d: %.2f' % (step, c))
                # Calling .eval() on valid_prediction is basically like calling run(), but
                # just to get that one numpy array. Note that it recomputes all its graph
                # dependencies.
                print('Validation loss: %.2f' % accuracy(valid_prediction.eval(), y_valid))
        t_pred = test_prediction.eval()
        print('Test loss: %.2f' % accuracy(t_pred, y_test))
        save_path = saver.save(sess, "./model/nn-model.ckpt")
        print('Model saved in %s' % (save_path))

    # Reconstructing graph and predicting outputs
    with tf.Session(graph=graph) as sess:
        saver.restore(sess, "./model/nn-model.ckpt")
        print("Model restored.\nMaking predictions...")
        conv_dates_rev_engg = [1 if values == 2014 else 0 for values in data_rev_engg.date ]
        data_rev_engg['date'] = conv_dates_rev_engg
        print("data.shape", data_rev_engg.shape)
        # Get the prediction from Gradient Boost LR
        x_test_gb = data_rev_engg.drop(['id', 'price'],axis=1)
        y_pred_rev_engg_gb = clf.predict(x_test_gb)
        y_pred_rev_engg_lr = reg.predict(x_test_gb)
        # Get the prediction from Reverse Engneering using NN
        data_rev_engg_cleaned = cleanup(data_rev_engg)
        x_rev_engg = data_rev_engg_cleaned.drop(['price', 'id'], axis=1).as_matrix().astype(np.float32)
        
        # xtrain_lr = x_train_raw_lr.drop(['price', 'id'], axis=1).as_matrix().astype(np.float32)
        #--------the following lines use the NN Model test data prediction
        hidden_rev_engg = tf.nn.relu(tf.matmul(x_rev_engg, weights_1) + biases_1)
        y_pred_rev_engg = tf.cast((tf.matmul(hidden_rev_engg, weights_2) + biases_2), dtype=tf.float32).eval()
        
        # data_rev_engg['price'] = y_pred_rev_engg
        # output_rev_engg = data_rev_engg[['id', 'price']] 
       
        # print("x_rev_engg.shape: ", x_rev_engg.shape)
        # print("output_rev_engg.shape: ", output_rev_engg.shape)
        # print("y_pred_rev_engg.shape: ", y_pred_rev_engg.shape)
        # output_rev_engg.to_csv('./submissions/nn-submission.csv', index=False)
        # print("Output saved to ./submissions/nn-submission.csv")

        # print(x_test.columns)
        output_disp_re = data_rev_engg.drop(['date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'view', 'condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'sqft_living15', 'sqft_lot15'] ,axis=1)
        df_output_re = pd.DataFrame(output_disp_re, columns=['lat', 'long', 'lr_pred_price', 'gb_pred_price', 'nn_pred_price', 'price'])
        df_output_re['lr_pred_price'] = y_pred_rev_engg_lr
        df_output_re['gb_pred_price'] = y_pred_rev_engg_gb
        df_output_re['nn_pred_price'] = y_pred_rev_engg
        #print(df_output_re)
        output_disp_re.to_csv('./submissions/nn-rev_engg_outcome.csv', index=False)
        print("Output saved to ./submissions/nn-rev_engg_outcome.csv")

       # ts = pd.Series(output_disp_re, index='id')
        # df = pd.DataFrame(output_disp_re, index='id', columns=['lr_pred_price', 'gb_pred_price', 'nn_pred_price', 'price'])
        # df = df.cumsum()
        # plt.figure()
        # df.plot()
if __name__ == "__main__": 
  tf.app.run()