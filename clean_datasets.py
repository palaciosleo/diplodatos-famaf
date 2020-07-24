# Importación de las librerías necesarias
import numpy as np
import pandas as pd
import re
import chardet
import requests
#from sklearn import preprocessing


def get_column_stats(column):
    """

    :param column: Pandas Column to extact stats from
    :return: pd.Dataframe with count, percent and accumulative percent per value
    """
    counts = column.value_counts()
    percent = column.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
    accum = column.value_counts(normalize=True).cumsum().mul(100).round(1).astype(str) + '%'

    return pd.DataFrame({'counts': counts, 'per': percent, 'accum': accum})


def get_dummy_column(column, top_rank=10, default_word='others'):
    """

    :param column: Pandas Column with values to map
    :param top_rank: number of top items to map
    :param default_word:
    :return: list of column ready to apply pd.get_dummies() on it
    """
    dummy_list = column.value_counts().reset_index()[:top_rank]['index'].tolist()

    dummy_column = []

    for row in column:
        valor_dummy = default_word
        if row in dummy_list:
            valor_dummy = row
        dummy_column.append(valor_dummy)

    return dummy_column


# Funcion normalizadora de precios
# Primero expresamos las unidades en base a Kg, Lt, Un o Mt
# Luego calculamos el precio normalizado
def normalizar_precio(precio, unidad, cantidad):
    if unidad in ['kg', 'lt', 'un', 'mt']:
        cantidad_normalizada = 1 / cantidad
    elif unidad in ['gr', 'ml', 'cc']:
        cantidad_unitaria = cantidad / 1000
        cantidad_normalizada = 1 / cantidad_unitaria
    else:
        print(precio, unidad, cantidad)

    precio_normalizado = cantidad_normalizada * precio
    return precio_normalizado


def normalizar_precio(precio, unidad, cantidad):
    if unidad in ['kg', 'lt', 'un', 'mt']:
        cantidad_normalizada = 1 / cantidad
    elif unidad in ['gr', 'ml', 'cc']:
        cantidad_unitaria = cantidad / 1000
        cantidad_normalizada = 1 / cantidad / 1000
    else:
        print(precio, unidad, cantidad)

    precio_normalizado = cantidad_normalizada * precio
    return precio_normalizado


def get_factor(df):
    um_primaria = ['kg', 'lt', 'un', 'mt', 'pack']
    um_secundaria = ['gr', 'ml', 'cc']

    df.loc[df.um_limpia.isin(um_primaria), 'factor_normalizador'] = 1 / df['cant_limpia']
    df.loc[df.um_limpia.isin(um_secundaria), 'factor_normalizador'] = 1 / df['cant_limpia'] * 1000


def is_outlier(precio, media, desvia):
    precios_limpios.loc[precio > media + 3 * desvia, 'resultado'] = 'extremo'
    precios_limpios.loc[precio <= media + 3 * desvia, 'resultado'] = 'normal'

























