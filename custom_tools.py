import pandas as pd
import numpy as np
import re


def get_mean_std(df):
    """
    Funcion que calcula la media y desviacion estandar por 'producto_id' y 'fecha'
    e inserta estas nuevas columnas al dataset.
    :param df: Recibe el dataframe de PRECIOS
    :return: Dataframe original con las columnas ['mean' , 'std']
    """
    precio_mean_std = df.groupby(['producto_id', 'fecha']).agg(precio_mean=('precio', 'mean'),
                                                                            precio_std=('precio', np.std))
    return pd.merge(df, precio_mean_std, left_on=['producto_id', 'fecha'], right_index=True)


def get_idreferencia(df):
    """
    Funcion que extrae el ID de Referencia para solucionar el problema de inconsistencia
    en donde un mismo producto cuyo ('nombre, 'marca', 'presentacion') son iguales
    :param df: Recibe el dataframe de PRODUCTOS
    :return: columna con ID de Referencia
    """
    try:
        diccionario = df.groupby(['nombre', 'marca', 'presentacion'])['id'].apply(lambda x: x.tolist()).to_dict()
        return [sorted(diccionario.get((producto.nombre, producto.marca, producto.presentacion)))[0]
            for idx, producto in df.iterrows()]
    except Exception as e:
        raise
