# -*- coding: utf-8 -*-

#%% LABELS

label_nums = {
0	: "T-shirt/top",
1	: "Trouser",
2	: "Pullover",
3	: "Dress",
4	: "Coat",
5	: "Sandal",
6	: "Shirt",
7	: "Sneaker",
8	: "Bag",
9	: "Ankle boot",
}
#%% Load

data_train = pd.read_csv(os.path.join(DATA_ROOT,'fashion-mnist_train.csv'))
data_test = pd.read_csv(os.path.join(DATA_ROOT,'fashion-mnist_test.csv'))

data_train['label'].unique()

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

X = np.array(data_train.iloc[:, 1:])
y = to_categorical(np.array(data_train.iloc[:, 0]))

#%% Split
#Here we split validation data to optimiza classifier during training
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)

#Test data
X_test = np.array(data_test.iloc[:, 1:])
y_test = to_categorical(np.array(data_test.iloc[:, 0]))

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_val = X_val.astype('float32')
X_train /= 255
X_test /= 255
X_val /= 255

logging.debug("Loaded data")
