# -*- coding: utf-8 -*-
batch_size = 256
num_classes = 10
epochs = 50

#input image dimensions
img_rows, img_cols = 28, 28
channels   = 1

#%%
def create_model1():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape))
    
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    

    logging.debug("Created {}".format('baseline'))
    return model

#create_model1().summary()


#%%
def create_model2():
    '''
    Creates a sequential model
    '''
    
    model = Sequential()
    model.add(ks.layers.InputLayer(input_shape=(img_rows,img_cols,channels)))
    # Normalization
    model.add(BatchNormalization())
    # Conv + Maxpooling
    model.add(ks.layers.Convolution2D(64, (4, 4), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Dropout
    model.add(Dropout(0.1))
    # Conv + Maxpooling
    model.add(ks.layers.Convolution2D(64, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Dropout
    model.add(Dropout(0.3))
    # Converting 3D feature to 1D feature Vektor
    model.add(Flatten())
    # Fully Connected Layer
    model.add(Dense(256, activation='relu'))
    # Dropout
    model.add(Dropout(0.5))
    # Fully Connected Layer
    model.add(Dense(64, activation='relu'))
    # Normalization
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))

    logging.debug("Created {}".format('Bigger'))

    return model

#create_model().summary()

#create_model2().summary()

#%% Model 3
def create_model3(input_shape):
    
    drop = 0.3
    # l2 regularization as well as dropout can help prevent overfitting
    l2_reg = ks.regularizers.l2(0.01)
    
    
    model = Sequential()
    model.add(ks.layers.InputLayer(input_shape=(img_rows,img_cols,channels)))

    #X_input = ks.layers.Input(input_shape)
    
    model.add(BatchNormalization())
    model.add(ks.layers.Conv2D(8, (3,3), strides=(1,1), activation='relu',
               kernel_regularizer=l2_reg,
               kernel_initializer='glorot_normal'))
    model.add(ks.layers.MaxPooling2D((2,2)))
    
    model.add(ks.layers.Conv2D(16, (3,3), strides=(1,1), activation='relu',
               kernel_regularizer=l2_reg,
               kernel_initializer='glorot_normal'))
    model.add(MaxPooling2D((2,2)))
    
    model.add(ks.layers.Conv2D(32, (2,2), strides=(1,1), activation='relu',
               kernel_regularizer=l2_reg,
               kernel_initializer='glorot_normal'))
    
    model.add(MaxPooling2D((2,2)))
    
    model.add(Flatten())
    
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(drop))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    
    model.add(Dense(16, activation='relu'))
#     X = Dropout(drop)(X)    
    
    model.add(Dense(10, activation='softmax'))
    
    #model.add(ks.models.Model(inputs=[X_input], outputs=[X])
    logging.debug("Created {}".format('Smaller'))

    return model

#create_model3(input_shape=(img_rows,img_cols,channels)).summary()


#%%
model = create_model3(input_shape=(img_rows,img_cols,channels))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
#%%
json_path = os.path.join(path_run,r"model_architecture.json")
model_json = model.to_json()
with open(json_path, "w") as json_file:
    json_file.write(model_json)
    
logging.info("Saved model to {}".format(json_path))


model.summary()


