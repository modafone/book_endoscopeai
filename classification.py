# -*- coding: utf-8 -*-
"""
Image classification using convolutional neural network
@author: Masahiro Oda (Nagoya Univ.)
"""

import os
import cv2
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix

# GPU使用の設定
gpu_id = 0  #使用するGPU番号
import tensorflow
physical_devices = tensorflow.config.list_physical_devices('GPU')
tensorflow.config.list_physical_devices('GPU')
tensorflow.config.set_visible_devices(physical_devices[gpu_id], 'GPU')
tensorflow.config.experimental.set_memory_growth(physical_devices[gpu_id], True)    #必要な分だけGPUメモリを使用する


IMAGESIZE = 64  #読み込んだ画像は(IMAGESIZE,IMAGESIZE)のサイズにリサイズする

# ディレクトリ内の画像を読み込む
# inputpath: ディレクトリ文字列, imagesize: 画像サイズ, type_color: ColorかGray
def load_images(inputpath, imagesize, type_color):
    imglist = []
    filenamelist = []

    for root, dirs, files in os.walk(inputpath):
        for fn in sorted(files):
            bn, ext = os.path.splitext(fn)
            if ext not in [".bmp", ".BMP", ".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
                continue

            filename = os.path.join(root, fn)
            
            if type_color == 'Color':
                # カラー画像の場合
                testimage = cv2.imread(filename, cv2.IMREAD_COLOR)
                # サイズ変更
                height, width = testimage.shape[:2]
                testimage = cv2.resize(testimage, (imagesize, imagesize), interpolation = cv2.INTER_AREA)  #主に縮小するのでINTER_AREA使用
                testimage = np.asarray(testimage, dtype=np.float64)
                # 色チャンネル，高さ，幅に入れ替え．data_format="channels_first"を使うとき必要
                #testimage = testimage.transpose(2, 0, 1)
                # チャンネルをbgrの順からrgbの順に変更
                testimage = testimage[:,:,::-1]
            
            elif type_color == 'Gray':
                # グレースケール画像の場合
                testimage = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                # サイズ変更
                height, width = testimage.shape[:2]
                testimage = cv2.resize(testimage, (imagesize, imagesize), interpolation = cv2.INTER_AREA)  #主に縮小するのでINTER_AREA使用
                # チャンネルの次元がないので1次元追加する
                testimage = np.asarray([testimage], dtype=np.float64)
                testimage = np.asarray(testimage, dtype=np.float64).reshape((imagesize, imagesize, 1))
                # チャンネル，高さ，幅に入れ替え．data_format="channels_first"を使うとき必要
                #testimage = testimage.transpose(2, 0, 1)

            imglist.append(testimage)
            filenamelist.append(fn)
    imgsdata = np.asarray(imglist, dtype=np.float32)

    return imgsdata, filenamelist  # 画像リストとファイル名のリストを返す


#%%
### データ準備 ###
print('*** Loading images ***')

# 画像読み込みとラベル値作成
# クラス0
image0, filenames_image0 = load_images('./sundatabase_negative/', IMAGESIZE, 'Color')
label0 = np.full(image0.shape[0], 0)    #画像数と同数のラベル値を含むリスト

# クラス1
image1, filenames_image1 = load_images('./sundatabase_positive/', IMAGESIZE, 'Color')
label1 = np.full(image1.shape[0], 1)    #画像数と同数のラベル値を含むリスト


# 画像，ラベル値，ファイル名それぞれ1つの配列にまとめる
image_all = np.concatenate([image0, image1], axis=0)
label_all = np.append(label0, label1)
filename_all = filenames_image0 + filenames_image1

# training用とtest用に8:2の割合でランダムにデータを分割する．indexはランダム分割後にファイル名を参照するために使う．
# 注意：ここで行うランダム分割は，同一症例の画像がtraining用とtest用の両方に含まれてしまうという問題がある．同一症例の画像で学習し評価する状態となり，評価において実際の分類性能より高い精度が出てしまう．
# 研究開発で使用する際は，同一症例の画像がtraining用とtest用の両方に含まれないような分割を行う必要がある．
indices = np.array(range(image_all.shape[0]))
image_train, image_test, label_train, label_test, index_train, index_test = train_test_split(image_all, label_all, indices, test_size=0.2)

# testデータのファイル名リスト作成
filenames_test = []
for i in range(len(index_test)):
    filenames_test.append(filename_all[index_test[i]])


# 画像の画素値を0-1に正規化
image_train /= 255.0
image_test /= 255.0

# ラベルをone hot vector形式に変換
label_train_binary = to_categorical(label_train)
label_test_binary = to_categorical(label_test)

print('Loaded images: ' + repr(image_train.shape[0]) + ' for training and ' + repr(image_test.shape[0]) + ' for testing.')


#%%
### 画像分類モデル定義と学習処理の実行 ###
print('*** Training ***')

# 分類モデル定義
# 基本的な分類CNN
def classification_model():
    input_img = Input(shape=(IMAGESIZE, IMAGESIZE, 3))

    x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(input_img)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=4, strides=4)(x)

    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=4, strides=4)(x)

    x = Flatten()(x)

    x = Dense(50)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(10)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(2)(x)
    x = Activation('softmax')(x)
    
    model = Model(inputs=input_img, outputs=x)
    
    return model

# 層の数を増やした分類CNN
def classification_model_deep1():
    input_img = Input(shape=(IMAGESIZE, IMAGESIZE, 3))

    x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(input_img)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=4, strides=4)(x)

    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=4, strides=4)(x)

    x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=4, strides=4)(x)

    x = Flatten()(x)

    x = Dense(100)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(10)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(2)(x)
    x = Activation('softmax')(x)
    
    model = Model(inputs=input_img, outputs=x)
    
    return model


# 分類モデル定義関数を呼び出す
model = classification_model()
#model = classification_model_deep1()   #層の数を増やしたCNNを呼び出す場合はこちらを有効化する．上の行はコメントアウトする．

# モデルを表示
print(model.summary())

# traininigの設定
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# training実行
training = model.fit(image_train, label_train_binary, epochs=3, batch_size=2, verbose=1)


# %%
### 学習済みモデルの評価 ###
print('*** Testing ***')

# testデータの分類精度を表示
scores = model.evaluate(image_test, label_test_binary, batch_size=5, verbose=0)
print("%s on test data: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#Confusion matrixを表示
results = model.predict(image_test, batch_size=5, verbose=1)
results_argmax = list(np.argmax(results, axis=-1))
cmatrix = confusion_matrix(label_test, results_argmax)
print('Confusion matrix:')
print(cmatrix)


# testデータの分類結果をcsvファイルに書き出し
f = open('./result.csv', 'w')
writer = csv.writer(f, lineterminator='\n')

savedata = ['filename', 'true_class', 'estimate_class', 'estimate_likelihood_max', 'estimate_likelihood_class0', 'estimate_likelihood_class1']
writer.writerow(savedata)

for i in range(len(image_test)):
    savedata = [filenames_test[i], label_test[i], np.argmax(results[i]), np.max(results[i]), results[i][0], results[i][1]]
    writer.writerow(savedata)
    
f.close()
