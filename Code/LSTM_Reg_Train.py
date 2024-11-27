#!/usr/bin/env python
# coding: utf-8

# 訓練模型

# In[138]:


#%%
import tensorflow as tf
from keras.models import Sequential
from keras.models import load_model, save_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, Callback, EarlyStopping
from keras import regularizers

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, MaxAbsScaler
from sklearn.compose import ColumnTransformer

import joblib
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os

#載入訓練資料
devices = [
    'L1',
    'L2',
    'L3',
    'L4',
    'L5',
    'L6',
    'L7',
    'L8',
    'L9',
    'L10',
    'L11',
    'L12',
    'L13',
    'L14',
    'L15',
    'L16',
    'L17'
    ]
for device in devices:
    SourceData = pd.read_csv(f"..//Data//MergedSorted//{device}_Merged_Sorted.csv")


    # In[139]:

        
    # LSTM 往前看的筆數和預測筆數，一筆10分鐘
    n_timesteps = LookBackNum = 12
    epochs=100
    batch_size=32
    validation_split=0.2
    reg_pca_components=40
    learning_rate=1e-4
    
    one_hot_encode_features = [
        # 'Device_ID',
        # 'Year',
        'Month',
        'Day',
        'Hour',
        'Minute',
    ]

    input_features_model_1 = to_predict_features_model_1 = [
        'Avg_Temperature(°C)',
        'Avg_Humidity(%)',
        'Avg_Sunlight(Lux)',
        # 'Avg_Power(mW)',
        
        'Max_Temperature(°C)',
        'Max_Humidity(%)',
        'Max_Sunlight(Lux)',
        'Max_Power(mW)',
        
        'Min_Temperature(°C)',
        'Min_Humidity(%)',
        'Min_Sunlight(Lux)',
        'Min_Power(mW)',
        
        
        # 'Avg_Diff_Temperature(°C)',
        # 'Avg_Diff_Humidity(%)',
        # 'Avg_Diff_Sunlight(Lux)',
        # 'Avg_Diff_Power(mW)',
        
        # 'Max_Diff_Temperature(°C)',
        # 'Max_Diff_Humidity(%)',
        # 'Max_Diff_Sunlight(Lux)',
        # 'Max_Diff_Power(mW)',
        
        # 'Min_Diff_Temperature(°C)',
        # 'Min_Diff_Humidity(%)',
        # 'Min_Diff_Sunlight(Lux)',
        # 'Min_Diff_Power(mW)',
        
        'Avg_Lag_1_Temperature(°C)',
        'Avg_Lag_2_Temperature(°C)',
        'Avg_Lag_1_Humidity(%)',
        'Avg_Lag_2_Humidity(%)',
        'Avg_Lag_1_Sunlight(Lux)',
        'Avg_Lag_2_Sunlight(Lux)',
        'Avg_Lag_1_Power(mW)',
        'Avg_Lag_2_Power(mW)',
        
        'Max_Lag_1_Temperature(°C)',
        'Max_Lag_2_Temperature(°C)',
        'Max_Lag_1_Humidity(%)',
        'Max_Lag_2_Humidity(%)',
        'Max_Lag_1_Sunlight(Lux)',
        'Max_Lag_2_Sunlight(Lux)',
        'Max_Lag_1_Power(mW)',
        'Max_Lag_2_Power(mW)',
        
        'Min_Lag_1_Temperature(°C)',
        'Min_Lag_2_Temperature(°C)',
        'Min_Lag_1_Humidity(%)',
        'Min_Lag_2_Humidity(%)',
        'Min_Lag_1_Sunlight(Lux)',
        'Min_Lag_2_Sunlight(Lux)',
        # 'Min_Lag_1_Power(mW)',
        # 'Min_Lag_2_Power(mW)',
        
        'MA6_Temperature(°C)',
        'MA6_Max_Temperature(°C)',
        'MA6_Min_Temperature(°C)',
        'MA6_Humidity(%)',
        'MA6_Max_Humidity(%)',
        'MA6_Min_Humidity(%)',
        'MA6_Sunlight(Lux)',
        'MA6_Max_Sunlight(Lux)',
        'MA6_Min_Sunlight(Lux)',
        'MA6_Power(mW)',
        'MA6_Max_Power(mW)',
        'MA6_Min_Power(mW)',
        # 'MA6_Diff_Temperature(°C)',
        # 'MA6_Max_Diff_Temperature(°C)',
        # 'MA6_Min_Diff_Temperature(°C)',
        # 'MA6_Diff_Humidity(%)',
        # 'MA6_Max_Diff_Humidity(%)',
        # 'MA6_Min_Diff_Humidity(%)',
        # 'MA6_Diff_Sunlight(Lux)',
        # 'MA6_Max_Diff_Sunlight(Lux)',
        # 'MA6_Min_Diff_Sunlight(Lux)',
        # 'MA6_Diff_Power(mW)',
        # 'MA6_Max_Diff_Power(mW)',
        # 'MA6_Min_Diff_Power(mW)',
        # 'MA6_Lag_1_Temperature(°C)',
        # 'MA6_Max_Lag_1_Temperature(°C)',
        # 'MA6_Min_Lag_1_Temperature(°C)',
        # 'MA6_Lag_2_Temperature(°C)',
        # 'MA6_Max_Lag_2_Temperature(°C)',
        # 'MA6_Min_Lag_2_Temperature(°C)',
        # 'MA6_Lag_1_Humidity(%)',
        # 'MA6_Max_Lag_1_Humidity(%)',
        # 'MA6_Min_Lag_1_Humidity(%)',
        # 'MA6_Lag_2_Humidity(%)',
        # 'MA6_Max_Lag_2_Humidity(%)',
        # 'MA6_Min_Lag_2_Humidity(%)',
        # 'MA6_Lag_1_Sunlight(Lux)',
        # 'MA6_Max_Lag_1_Sunlight(Lux)',
        # 'MA6_Min_Lag_1_Sunlight(Lux)',
        # 'MA6_Lag_2_Sunlight(Lux)',
        # 'MA6_Max_Lag_2_Sunlight(Lux)',
        # 'MA6_Min_Lag_2_Sunlight(Lux)',
        # 'MA6_Lag_1_Power(mW)',
        # 'MA6_Max_Lag_1_Power(mW)',
        # 'MA6_Min_Lag_1_Power(mW)',
        # 'MA6_Lag_2_Power(mW)',
        # 'MA6_Max_Lag_2_Power(mW)',
        # 'MA6_Min_Lag_2_Power(mW)',
        
        # 'Sin_Hour',
        # 'Cos_Hour',
        # 'Sin_Minute',
        # 'Cos_Minute',
        # 'Sin_Month',
        # 'Cos_Month',
        # 'Sin_Day',
        # 'Cos_Day'
    ]
    
    target_column = ['Avg_Power(mW)']
    SourceData = SourceData[['SeqNumber'] + to_predict_features_model_1 + one_hot_encode_features]
    SourceData = pd.get_dummies(SourceData, columns=one_hot_encode_features, dtype='int')


    # In[140]:


    def create_dataset(data, LookBackNum):
        X = []
        y = []

        #設定每i-12筆資料(X_train)就對應到第i筆資料(y_train)
        for i in range(LookBackNum,len(data)):
            X.append(data[i-LookBackNum:i, :])
            y.append(data[i, :])

        return np.array(X), np.array(y)


    preprocess_pipe = make_pipeline(
        MinMaxScaler(),
        # PCA(n_components=11),
    )

    LSTM_TrainData = SourceData.copy()
    LSTM_TrainData.dropna(inplace=True)
    #正規化
    LSTM_TrainData[to_predict_features_model_1] = preprocess_pipe.fit_transform(LSTM_TrainData[to_predict_features_model_1])


    X_train, _ = create_dataset(LSTM_TrainData.drop(columns='SeqNumber').values, LookBackNum=LookBackNum)
    _, y_train = create_dataset(LSTM_TrainData[to_predict_features_model_1].values, LookBackNum=LookBackNum)

    n_features = X_train.shape[2]
    n_prediction = y_train.shape[1]

    # Reshaping
    #(samples 是訓練樣本數量,timesteps 是每個樣本的時間步長,features 是每個時間步的特徵數量)
    X_train = np.reshape(X_train,(X_train.shape[0], n_timesteps, n_features))
    X_train.shape

    # In[141]:


    #%%
    #============================建置&訓練「LSTM模型」============================
    #建置LSTM模型
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True
        )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5,     # 衰減率
        patience=10,    
        min_lr=1e-7
        )

    def build_lstm_model(n_timesteps, n_features, n_prediction):
        model = Sequential()
        
        model.add(LSTM(units=100, return_sequences=True, activation='tanh',input_shape=(n_timesteps, n_features)))
        model.add(Dropout(0.2))
        
        model.add(LSTM(units=100, return_sequences=False, activation='tanh'))
        model.add(Dropout(0.2))

        # model.add(Dense(units=128, activation='tanh'))
        # model.add(Dropout(0.2))
        
        model.add(Dense(units=n_prediction, activation='relu'))

        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
        model.summary()
        return model

    regressor = build_lstm_model(n_timesteps, n_features, n_prediction)


    # In[142]:


    #開始訓練
    
    history = regressor.fit(
        X_train, 
        y_train, 
        epochs=epochs, 
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[reduce_lr,early_stopping],
        )


    # In[143]:


    # import matplotlib.pyplot as plt


    # train_loss = history.history['loss']
    # val_loss = history.history['val_loss']


    # plt.figure(figsize=(10, 6))
    # plt.plot(train_loss, label='Train Loss', color='blue')
    # plt.plot(val_loss, label='Validation Loss', color='orange')
    # plt.title('Train Loss vs Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    # In[144]:


    #保存模型
    model_path = f'..//Model//WheatherLSTM_{device}.h5'
    regressor.save(model_path)
    print('Model Saved')


    # ## 訓練迴歸模型

    # In[145]:


    TrainData = pd.read_csv(f"..//Data//MergedSorted//{device}_Merged_Sorted.csv")
    TrainData.dropna(inplace=True)


    # In[146]:


    X_full = TrainData[input_features_model_1]

    if 'Avg_Power(mW)' in input_features_model_1 :
        X_full = X_full.drop(columns='Avg_Power(mW)')
    else:
        X_full = X_full
    reg_scaler = MinMaxScaler()
    X_full = reg_scaler.fit_transform(X_full.values)
    y_full = TrainData['Avg_Power(mW)'].values

    X_train, X_val, y_train, y_val = train_test_split(X_full,y_full,test_size=0.2,shuffle=True)

    reg_model = make_pipeline(
        PCA(n_components=reg_pca_components),
        LinearRegression(),
    )

    cv_scores = cross_val_score(reg_model, X_train, y_train, cv=20)
    cv_scores


    # In[147]:


    reg_model.fit(X_train, y_train)


    # In[148]:


    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    y_pred = reg_model.predict(X_val)
    y_pred = y_pred = np.clip(y_pred, 0, None)

    print('MSE: ',mean_squared_error(y_val, y_pred))
    print('MAE: ',mean_absolute_error(y_val, y_pred))
    print('R2:',r2_score(y_val, y_pred))


    # In[149]:


    reg_model.fit(X_full, y_full)


    # ## 預測答案

    # In[150]:


    #載入模型
    model_path = f'..//Model//WheatherLSTM_{device}.h5'
    model = load_model(model_path, compile=False)
    print('Model Loaded Successfully')


    # In[151]:


    TestData = pd.read_csv('..//Data/TestData//upload(no answer).csv')

    TestData = TestData[TestData['序號'] % 100 == int(device[1:])]

    to_predict_sequmber = TestData['序號'].to_list()

    # 預測的資料 的 index
    indices_1 = SourceData[SourceData['SeqNumber'].isin(to_predict_sequmber)][to_predict_features_model_1].index.to_list()
    len(indices_1)


    # In[152]:


    index_min = min(indices_1) - n_timesteps
    index_max = max(indices_1)

    indices_2 = SourceData.loc[index_min:index_max][to_predict_features_model_1].index.tolist()

    # 找出有 NaN 的 row
    rows_with_na = SourceData.loc[indices_2, to_predict_features_model_1].isnull().any(axis=1)
    rows_with_na_data = SourceData.loc[indices_2, to_predict_features_model_1][rows_with_na]

    # 有 NaN 的 row 的 index
    indices_with_na =  rows_with_na_data.index.to_list()
    len(indices_with_na)


    # In[153]:


    # 如果 LookBackNum > 12 選 indices_with_na
    # 其餘選 indices_1
    PredictedData = SourceData.copy()
    indices_to_use = indices_with_na if LookBackNum > 12 else indices_1

    for index in indices_to_use:
        X = PredictedData.loc[index-LookBackNum : index-1].drop(columns="SeqNumber")
        if 'Avg_Power(mW)' in X.columns.to_list():
            X = X.drop(columns='Avg_Power(mW)')
        
        X[to_predict_features_model_1] = preprocess_pipe.transform(X[to_predict_features_model_1])
        X = X.values
        X = np.reshape(X,(1, n_timesteps, n_features))
        
        pred = model.predict(X)
        pred = preprocess_pipe.inverse_transform(pred)
        PredictedData.loc[index, to_predict_features_model_1] = pred
        # PredictedData.loc[index, ['Avg_Power(mW)']] = PredictedData.loc[index, ['Avg_Power(mW)']].apply(lambda x: 0 if x <= 0 else x)
        
        X = PredictedData.loc[index, to_predict_features_model_1].to_frame().T
        if 'Avg_Power(mW)' in to_predict_features_model_1:
            X = X.drop(columns='Avg_Power(mW)')
        else:
            X = X
        X = reg_scaler.transform(X.values)
        pred = reg_model.predict(X)
        pred = pred[0]
        pred = np.clip(pred, 0, None)
        PredictedData.loc[index, 'Avg_Power(mW)'] = pred


    # In[ ]:


    PredictedData.loc[indices_1].to_csv(f'..//Data//PredictedData//Predicted_{device}.csv', index=False)
    PredictedData.to_csv(f'..//Data//PredictedOverAllData//Predicted_OverAll_{device}.csv', index=False)

