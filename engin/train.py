import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from utils import fen_to_tensor
from models import eval_model

# Load datasets
train_data = pd.read_csv("datasets/chessData.csv")
test_data = pd.read_csv("datasets/test_eval.csv")


train_data['FEN'] = train_data['FEN'].apply(lambda x: x.split(' ')[0])
test_data['FEN'] = test_data['FEN'].apply(lambda x: x.split(' ')[0])

train_data['FEN'] = train_data['FEN'].apply(fen_to_tensor)
test_data['FEN'] = test_data['FEN'].apply(fen_to_tensor)

train_data['Evaluation'] = train_data['Evaluation'].apply(lambda x: int(x[1:]) * 10000 if x.startswith('#') else int(x))
test_data['Evaluation'] = test_data['Evaluation'].apply(lambda x: int(x[1:]) * 10000 if x.startswith('#') else int(x))





# # Features and target for Dataset 1
# features_1 = data1.drop(columns=['ASD', 'age_desc', 'contry_of_res', 'used_app_before'])
# target_1 = data1['ASD']
#
# # Normalize numerical columns in Dataset 1
# scaler_1 = StandardScaler()
# features_1[['age', 'result']] = scaler_1.fit_transform(features_1[['age', 'result']])
#
#


def train_and_evaluate(X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
    model = eval_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    return model
#
# Train on Dataset 1
print("Training on Dataset 1:")
model_1 = train_and_evaluate(train_data['FEN'], train_data['Evaluation'], test_data['FEN'], test_data['Evaluation'])

# # Train on Dataset 2
# print("\nTraining on Dataset 2:")
# model_2 = train_and_evaluate(X_train_2, y_train_2, X_test_2, y_test_2)
