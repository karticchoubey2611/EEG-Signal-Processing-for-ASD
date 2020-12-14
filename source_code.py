import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

df=pd.read_csv('preprocessed_dataset.csv')
df.head()

sns.countplot(x='label', data=df)

# df.isnull().sum().sum()

encode = ({'NEUTRAL': 0, 'POSITIVE': 1, 'NEGATIVE': 2} )
df_encoded = df.replace(encode)

X=df_encoded.drop(["label"]  ,axis=1)
y = df_encoded.loc[:,'label'].values

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train = np.reshape(X_train, (X_train.shape[0],1,X.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0],1,X.shape[1]))


import tensorflow as tf
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
tf.keras.backend.clear_session()

def get_model():
    model = Sequential([
        LSTM(64, input_shape=(1,2548),activation="relu",return_sequences=True),
        Dropout(0.2),
        LSTM(32,activation="relu",return_sequences=True),
        Dropout(0.2),
        LSTM(16,activation="relu"),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    
    return model

model = get_model()

from keras.optimizers import SGD
model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'])

score, acc = model.evaluate(X_test, y_test)

print(f"Accuracy before Fine tuning Model {acc}")

history = model.fit(X_train, y_train, epochs = 20, validation_data= (X_test, y_test), verbose = 0)

score, acc = model.evaluate(X_test, y_test)

print(f"Accuracy after Fine tuning Model {acc}")

from sklearn.metrics import confusion_matrix
pred = model.predict(X_test)
predict_classes = np.argmax(pred,axis=1)
expected_classes = np.argmax(y_test,axis=1)

conf_matrix = confusion_matrix(expected_classes,predict_classes)
print("Confusion Matrix Accuracy: \n", conf_matrix)
