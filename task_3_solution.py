import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1) будем обучать линейную модель в этом задании на данных о стоимости квартиры в Москве. Перед тем, как мы перейдем к обучению моделей, необходимо выполнить предварительную подготовку данных.
# Для начала, попробуем разбить данные на обучающую часть и валидационную часть в соотношении 70 / 30 и предварительным перемешиванием. 
# Параметр `ranom_state` зафиксировать 42. Назовите функцию `split_data_into_two_samples`, которая принимает полный датафрейм, а возвращает 2 датафрейма: для обучения и для валидации.

def split_data_into_two_samples(X):
    
    train, test = train_test_split(X, 
                                                        test_size    = 0.3,
                                                        train_size   = 0.7,
                                                        random_state =42)
    
    return train, test


# 2) продолжим выполнение предварительной подготовки данных: в данных много категориальных признаков (они представлены типами `object`), пока мы с ними работать не умеем, поэтому удалим их из датафрейма.
# Кроме того, для обучения нам не нужна целевая переменная, ее нужно выделить в отдельный вектор (`price_doc`). 
# Написать функцию `prepare_data`, которая принимает датафрейм, удаляет категориальные признаки, удаляет `id`, и выделяет целевую переменную в отдельный вектор. 
# Кроме того, некоторые признаки содержат пропуски, требуется удалить такие признаки. Функция должна возвращать подготовленную матрицу признаков и вектор с целевой переменной.

def prepare_data(data):

    price_doc = data.iloc[:,-1]
    data = data.select_dtypes(['number']).drop(['SalePrice','Id','LotFrontage','GarageYrBlt','MasVnrArea','index'], axis=1)  
    
    return data, price_doc

# 3) Перед обучением линейной модели также необходимо масштабировать признаки. Для этого мы можем использовать `MinMaxScaler` или `StandardScaler`.
# Написать функцию - `scale_data`, которая принимает на вход датафрейм и трансформер, а возвращает датафрейм с отмасштабированными признаками.

def scale_data(transformer, data):
    
    numeric_data = data.select_dtypes([np.number])
    numeric_features = numeric_data.columns
    
    data_scaled = transformer.fit_transform(data[numeric_features])
    
    return data_scaled

# 4) объединить задание 2 и 3 в единую функцию `prepare_data_for_model`, функция принимает датафрейм и трансформер для масштабирования, возвращает данные в формате задания 3 и вектор целевой переменной.

def prepare_data_for_model(transformer, data):
    
    x,y = prepare_data(data)
    x = scale_data(scaler, x)
    
    return x,y

# 5) разбить данные на обучение / валидацию в соответствии с заданием 1. Обучить линейную модель (`LinearRegression`) на данных из обучения. 
# При подготовке данных использовать функцию из задания 4, в качестве трансформера для преобразования данных использовать - `StandardScaler`. 
# Создать функцию `fit_first_linear_model`, которая принимает на вход `x_train` и `y_train`, а возвращает модельку.

def fit_first_linear_model(x_train,y_train):
    
    model = LinearRegression()
    # model = SGDClassifier()
    model.fit(x_train, y_train)
    
    return model

# 6) выполнить задание 5, но с использованием `MinMaxScaler`.

def fit_first_linear_model(x_train,y_train):
    
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    return model


# 7) написать функцию для оценки качества модели - `evaluate_model`, которая принимает на вход обученную модель, выборку для построения прогнозов и вектор истинных ответов для выборки. 
# Внутри функции вычислить метрики `MSE`, `MAE`, `R2`, вернуть значения метрик, округленных до 2-го знака. Для построения / оценки качества использовать разбиение из задания 1

def evaluate_model(model,x_test,y_test):
    
    y_pred_test = model.predict(x_test)
    
    MSE = np.round( np.sqrt(mean_squared_error(y_test, y_pred_test)) ,2)
    MAE = np.round( np.sqrt(mean_absolute_error(y_test, y_pred_test)),2)
    R2  = np.round( np.sqrt(r2_score(y_test, y_pred_test)),2)
    
    return MSE, MAE, R2


# 8) написать функцию, которая принимает на вход обученную модель и список названий признаков, и создает датафрейм с названием признака и абсолютным значением веса признака. 
# Датафрейм отсортировать по убыванию важности признаков и вернуть. Назвать функцию `calculate_model_weights`. Для удобства, колонки датафрейма назвать `features` и `weights`

def calculate_model_weights(model, scales):
    
    data = {'features':scales,
            'weights':model.coef_}
    return pd.DataFrame(data).sort_values('weights',ascending = False)








