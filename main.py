import json
from kafka import KafkaConsumer
import numpy as np
from pyspark import SparkContext, SparkConf

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from elephas.utils.rdd_utils import to_simple_rdd
from elephas.spark_model import SparkModel

conf = SparkConf().setAppName('Elephas_App').setMaster('local[8]')
sc = SparkContext(conf=conf)

consumer = KafkaConsumer(
    'BKKLearningTopic',
    group_id='bkk-group',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

labels = []
features = []

for message in consumer:
    print(message.value)
    labels.append(message.value["label"])
    features.append(message.value["features"]["values"])

labeledpoints = np.array(labels, features)

model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=SGD())

lp_rdd = to_simple_rdd(sc, features, labels, categorical=True)
spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')
spark_model.fit(lp_rdd, epochs=20, batch_size=32, verbose=0, validation_split=0.1)

spark_model.save("model.h5")