import pandas as pd
import numpy as np
# data = pd.read_csv('sberbank_housing_market.csv')


# 1 написать функцию `calculate_data_shape`, которая принимает на вход датафрейм `X` и возвращает его размерность
def calculate_data_shape(data_frame):
    return data_frame.shape

# 2 написать функцию `take_columns`, которая принимает на вход датафрейм `X` и возвращает название его столбцов
def take_columns(data_frame):
    return data_frame.columns

# 3 написать функцию `calculate_target_ratio`, которая принимает на вход датафрейм `X` и название целевой переменной `target_name` -
# возвращает среднее значение целевой переменной. Округлить выходное значение до 2-го знака внутри функции.
def calculate_target_ratio(X,target_name):
    return round(X[target_name].mean(),2)

# 4 написать функцию `calculate_data_dtypes`, которая принимает на вход датафрейм `X` и возвращает количество числовых признаков и категориальных признаков.
# Категориальные признаки имеют тип `object`.

def calculate_data_dtypes(X):
    categorical_columns = [c for c in X.columns if X[c].dtype.name == 'object']
    numerical_columns = [c for c in X.columns if X[c].dtype.name != 'object']
    return len(categorical_columns),len(numerical_columns)

# 5 написать функцию calculate_cheap_apartment, которая принимает на вход датафрейм X и возвращает количество квартир, стоимость которых не превышает 1 млн. рублей.
def calculate_cheap_apartment(X):
    return len(X[X.price_doc <= 1000000].index)

# 6. написать функцию calculate_squad_in_cheap_apartment, которая принимает на вход датафрейм `X` и возвращает среднюю площадь квартир, стоимость которых не превышает 1 млн. рублей.
# Признак, отвечающий за площадь - full_sq. Ответ округлить целого значения.
def calculate_squad_in_cheap_apartment(X):
    return round(X[X.price_doc <= 1000000 ].full_sq.mean())

# 7. написать функцию calculate_mean_price_in_new_housing,
# которая принимает на вход датафрейм X и возвращает среднюю стоимость трехкомнатных квартир в доме, который не старше 2010 года.
# Ответ округлить до целого значения.
def calculate_mean_price_in_new_housing(X):
    return round(X[(X.build_year >= 2010) & (X.num_room == 3)].price_doc.mean())

# 8. написать функцию `calculate_mean_squared_by_num_rooms`, которая принимает на вход датафрейм `X` и
# возвращает среднюю площадь квартир в зависимости от количества комнат.
# Каждое значение площади округлить до 2-го знака.
def calculate_mean_squared_by_num_rooms(X):
    return X.groupby(['num_room']).full_sq.mean().round(2)

# 9. написать функцию `calculate_squared_stats_by_material`, которая принимает на вход датафрейм `X` и возвращает максимальную и
# минимальную площадь квартир в зависимости от материала изготовления дома. Каждое значение площади округлить до 2-го знака.
def calculate_squared_stats_by_material(X):
    return X.groupby(['material'])['full_sq'].agg([('max', 'max'), ('min', 'min')]).round(2)


# 10. Написать функцию calculate_crosstab, которая принимает на вход датафрейм X и возвращает СРЕДНЮЮ СТОИМОСТЬ квартир в зависимости от района города и цели покупки.
# Ответ - сводная таблица, где индекс - район города (признак - sub_area), столбцы - цель покупки (признак - product_type).
# Каждое значение цены округлить до 2-го знака, пропуски заполнить нулем.
def calculate_crosstab(X):
    return X.pivot_table( values='price_doc', index=['sub_area'], columns=['product_type'], aggfunc=np.average).fillna(0)

