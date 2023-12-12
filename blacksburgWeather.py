import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import datetime
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.optimizers import Adam
import keras.backend as K

df = pd.read_csv('Blacksburg Past Weather.csv')

df['DATE'] = pd.to_datetime(df['DATE'])

numeric_cols = ['PRCP', 'SNOW', 'SNWD']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

df.infer_objects(copy=False)

df = df.interpolate()

df['day_of_week'] = df['DATE'].dt.dayofweek
df['month'] = df['DATE'].dt.month
df['year'] = df['DATE'].dt.year

features = ['day_of_week', 'month', 'year', 'PRCP', 'SNOW', 'SNWD', 'TMAX']
target_variables = ['TMAX', 'TMIN', 'TOBS', 'PRCP']

X = df[features]
y = df[target_variables]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(target_variables))
])

def custom_loss(y_true, y_pred):
    tmin_weight = 2.0
    squared_error = K.square(y_true - y_pred)
    weighted_error = K.switch(K.equal(y_true, 'TMIN'), tmin_weight * squared_error, squared_error)
    return K.mean(weighted_error, axis=-1)

model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_data=(X_test_scaled, y_test), callbacks=[early_stopping])

predictions = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

next_week_features = pd.DataFrame({
    'day_of_week': [datetime.datetime.strptime('12/06/2023', '%m/%d/%Y').weekday()],
    'month': [12],
    'year': [2023],
    'PRCP': [0],
    'SNOW': [0],
    'SNWD': [0],
    'TMAX': [0]
})

next_week_features_scaled = scaler.transform(next_week_features)

next_week_predictions = model.predict(next_week_features_scaled)
print('Predictions for the next week:')
print(next_week_predictions)
