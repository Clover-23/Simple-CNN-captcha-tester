import os, cv2, time

import numpy as np
import tensorflow as tf
import configparser as cfp
import matplotlib.pyplot as plt

from random import shuffle, seed

# get model setting from ini file
class TrainIni:
    def __init__(self):
        self.__cfp = cfp.ConfigParser()
        self.__cfp.read('training.ini')

        self.model_name = self.__cfp['TRAIN']['model_name']
        self.loop_times = int(self.__cfp['TRAIN']['loop_times'])
        self.epochs = int(self.__cfp['TRAIN']['epochs'])
        self.batch_size = int(self.__cfp['TRAIN']['batch_size'])
        self.vdt_split = float(self.__cfp['TRAIN']['validation_split'])
        self.vdt_steps = int(self.__cfp['TRAIN']['validation_steps'])
        self.is_multiprocessing = bool(int(self.__cfp['TRAIN']['multiprocessing']) > 0)
        self.imgs_folder = self.__cfp['TRAIN']['imgs_folder']

# model
my_model = tf.keras.Model()

# put conv2d*2 + batchNormalization layer into layer
def make_conv_layer(layer, filter):
    buf_layer = tf.keras.layers.Conv2D(filters=filter, kernel_size=(3,3), padding='same', activation='relu')(layer)
    buf_layer = tf.keras.layers.Conv2D(filters=filter, kernel_size=(3,3), padding='same', activation='relu')(buf_layer)
    buf_layer = tf.keras.layers.BatchNormalization()(buf_layer)
    return buf_layer

# pooling layer
def add_pooling_layer(layer, pool_shape, drop):
    return tf.keras.layers.Dropout(drop)(tf.keras.layers.MaxPool2D(pool_shape, padding='same')(layer))

# make whole model, save in 'my_model'
def make_model(img_shape):
    global my_model
    
    # input layer
    input_layer = tf.keras.Input(shape=img_shape)

    # customize layers
    layerN = add_pooling_layer(make_conv_layer(input_layer, 16), (2, 2), drop=0.5)
    layerN = add_pooling_layer(make_conv_layer(layerN, 32), (2, 2), drop=0.4)
    layerN = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer='l1_l2')(layerN)
    layerN = tf.keras.layers.BatchNormalization()(layerN)
    layerL = tf.keras.layers.Dropout(0.5)(layerN)

    # flatten network before output
    fc = tf.keras.layers.Flatten()(layerL)

    # output structure
    my_model = tf.keras.Model(inputs = input_layer, outputs=[
            tf.keras.layers.Dense(36, activation='softmax', name='dense_1')(fc),
            tf.keras.layers.Dense(36, activation='softmax', name='dense_2')(fc),
            tf.keras.layers.Dense(36, activation='softmax', name='dense_3')(fc),
            tf.keras.layers.Dense(36, activation='softmax', name='dense_4')(fc),
        ])

# take captcha data
def get_data(input_path: str):
    p, p1, p2 = [], [], []
    total = 0
    input_path += '/'
    for p, d, f in os.walk(input_path):
        p = f
        # shuffle all data
        seed(time.time())
        shuffle(p)
        total = len(f)
        p1 = p[0:int(total*0.75)]   # p1 (0.75) for training
        p2 = p[int(total*0.75):]    # p2 (0.25) for testing
        break

    train_png, test_png = [], []
    train_ans, test_ans = [[],[],[],[]], [[],[],[],[]]

    train_data_size = len(p1)
    test_data_size = len(p2)
    
    # customize label code & buffer
    label_buffer = [0 for _ in range(36)]
    label_code = '9876543210ZYXWVUTSRQPONMLKJIHGFEDCBA'

    # prepare training data
    count = 0
    print('Reading training data(Total: {})...'.format(train_data_size))
    for ele in p1:
        # read captcha image which have to change BGR to GRAY
        train_png.append(np.array(cv2.cvtColor(cv2.imread('{}{}'.format(input_path, ele)), cv2.COLOR_BGR2GRAY).reshape((20, 60, 1))) / 255.0)
        
        # answer to label input
        i = 0
        for e in ele.split('.')[0]:
            label = label_buffer.copy()
            label[label_code.find(e)] = 1
            train_ans[i].append(np.array(label.copy(), dtype=np.int32))
            i += 1
        count += 1
        if count % 1000 == 0:
            print('Now {}/{}'.format(count, train_data_size))

    # prepare testing data
    count = 0
    print('Reading testing data(Total: {})...'.format(test_data_size))
    for ele in p2:
        # read captcha image which have to change BGR to GRAY
        test_png.append(np.array(cv2.cvtColor(cv2.imread('{}{}'.format(input_path, ele)), cv2.COLOR_BGR2GRAY).reshape((20, 60, 1))) / 255.0)
        
        # answer to label input
        i = 0
        for e in ele.split('.')[0]:
            label = label_buffer.copy()
            label[label_code.find(e)] = 1
            test_ans[i].append(np.array(label.copy(), dtype=np.int32))
            i += 1
        count += 1
        if count % 1000 == 0:
            print('Now {}/{}'.format(count, test_data_size))

    # x_train, y_train, x_test, y_test
    return np.array(train_png), np.array(train_ans), np.array(test_png), np.array(test_ans)

# draw plot, check evaluation
def draw_plot_and_show(data: dict, name: str) -> None:
    plt.plot(data[name])
    plt.title('Train History')
    plt.ylabel(name)
    plt.xlabel('Loops')
    plt.legend(['train'], loc='upper left')
    plt.show()

# training model
def my_main():
    global my_model

    # ini file reader
    model_setting = TrainIni()

    # get captcha data
    train_png, train_ans, test_png, test_ans = get_data(model_setting.imgs_folder)

    #input_shape = (20, 60, 1)
    input_shape = train_png[0].shape
    
    # read model, if there is no target model, make a new model
    if not os.path.isfile(model_setting.model_name):
        # build a new model
        make_model(img_shape=input_shape)
        my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        my_model.build(input_shape=input_shape)
    else:
        # load model
        my_model = tf.keras.models.load_model(model_setting.model_name)
    
    # summary
    my_model.summary()

    # training per loop, 'plot_score' save evaluation data pre loop
    plot_score = {'loss':[], 'acc':[]}
    for i in range(model_setting.loop_times):
        print('Now in {}/{} training loop'.format(i+1, model_setting.loop_times))
        # model fit according to ini-file training setting
        my_model.fit(
            x=train_png,
            y=[train_ans[0], train_ans[1], train_ans[2], train_ans[3]],
            validation_split=model_setting.vdt_split,
            validation_steps=model_setting.vdt_steps,
            epochs=model_setting.epochs,
            batch_size=model_setting.batch_size,
            use_multiprocessing=model_setting.is_multiprocessing
        )

        # save model & evaluate
        my_model.save(model_setting.model_name)
        evaList = my_model.evaluate(x=test_png, y=[test_ans[0],test_ans[1],test_ans[2],test_ans[3]])

        loss, acc = evaList[0], sum(evaList[5:])/4

        print('\n\nloss: {}\nacc: {}\n'.format(loss, acc))
        plot_score['loss'].append(loss)
        plot_score['acc'].append(acc)
    
    # draw plot
    draw_plot_and_show(plot_score, 'loss')
    draw_plot_and_show(plot_score, 'acc')

if __name__ == '__main__':
    my_main()