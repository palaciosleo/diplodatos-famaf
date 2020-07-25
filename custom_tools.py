import pandas as pd
import numpy as np
import re


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

# sucursales.drop(['comercioId', 'banderaId', 'comercioRazonSocial', 'provincia', 'localidad', 'direccion', 'region'], axis=1)
# productos.drop(['categoria1', 'categoria2', 'categoria3'], axis=1)


def is_float(number):
    """
    Funciona auxiliar para determinar si un string puede convertirse en float
    :param number:  Numero en string
    :return: True or False si el numero puede convertirse a float
    """
    try:
        float(number)
        return True
    except Exception as e:
        return False


def get_um_presentacion_from_nombre(df):
    """
    Extraer la 'um' y 'presentacion' del nombre del producto
    :param df: Dataframe de PRODUCTOS
    :return:   Columnas ['um_en_nombre_prod', 'cant_en_nombre_prod']
    """
    try:
        cant_en_nombre_prod = []
        um_en_nombre_prod = []
        pattern_um = r'(?P<cant_en_nombre_prod>[\d\.]+)\s?(?P<um_en_nombre_prod>\D{1,3}$)'

        for idx, producto in df.iterrows():
            # Si no encuentro un match, lo dejo como esta en la presentacion
            um = producto['um_en_presentacion']
            # Si no encuentro un match, lo dejo como esta en la presentacion
            cantidad = producto['cantidad_en_presentacion']

            match = re.search(pattern_um, producto["nombre_depurado"])
            if match is not None:
                if is_float(match.group('cant_en_nombre_prod')):
                    cantidad = match.group('cant_en_nombre_prod')
                    um = match.group('um_en_nombre_prod')

            cant_en_nombre_prod.append(cantidad)
            um_en_nombre_prod.append(um)


        return cant_en_nombre_prod, um_en_nombre_prod

    except Exception as e:
        raise


def get_um_fixed(df_productos):
    """
    Aplica un set expresiones regulares para arreglar algunas 'um'
    :param df: Dataframe de PRODUCTOS
    :return: Dataframe con columna 'um_en_nombre_prod' corregida
    """
    try:
        df = df_productos.copy()
        df['um_en_nombre_prod'] = df['um_en_nombre_prod'].str.replace('[^a-zA-Z]', '')
        df['um_en_nombre_prod'] = df['um_en_nombre_prod'].str.replace('k\w+', 'kg')
        df['um_en_nombre_prod'] = df['um_en_nombre_prod'].str.replace('g\w+', 'gr')
        df['um_en_nombre_prod'] = df['um_en_nombre_prod'].str.replace('un\w+', 'un')
        df['um_en_nombre_prod'] = df['um_en_nombre_prod'].str.replace('^g$', 'gr')
        df['um_en_nombre_prod'] = df['um_en_nombre_prod'].str.replace('^l$', 'lt')
        df['um_en_nombre_prod'] = df['um_en_nombre_prod'].str.replace('^u$', 'un')
        df['um_en_nombre_prod'] = df['um_en_nombre_prod'].str.replace('^c$', 'cc')

        return df

    except Exception as e:
        raise


def custom_std(x):
    """
    Tenemos que definir esta funcion porque el STD de Pandas devuelve NaN para una muestra de una sola ocurrencia.
    Para mas informacion ver: https://stackoverflow.com/questions/50306914/pandas-groupby-agg-std-nan
    Por defecto numpy setea los Delta Degrees of Freedom en 0, mientras que pandas lo setea en 1
    :param x: Lista de precios agrupados
    :return: Lista de desviaciones estandares
    """
    return np.std(x)


def get_mean_std(df):
    """
    Funcion que calcula la media y desviacion estandar por 'producto_id' y 'fecha'
    e inserta estas nuevas columnas al dataset.
    :param df: Recibe el dataframe de PRECIOS
    :return: Dataframe original con las columnas ['mean' , 'std']
    """
    try:
        precio_mean_std = df.groupby(['producto_id', 'fecha']).agg(precio_mean=('precio', 'mean'),
                                                                            precio_std=('precio', custom_std))
        return pd.merge(df, precio_mean_std, left_on=['producto_id', 'fecha'], right_index=True)
    except Exception as e:
        raise


def get_outlier_by_mean(df_precios):
    """
    Calcula si el precio del producto es outlier considerando si es mayor a 3 veces la media
    :param df_precios: Dataframe de PRECIOS
    :return: Una copia del dataframe de PRECIOS con la columna 'outlier_by_mean'
    """
    try:
        df = df_precios.copy()
        df.loc[df['precio'] > df['precio_mean'] + 3 * df['precio_std'], 'outlier_by_mean'] = 'extremo'
        df.loc[df['precio'] <= df['precio_mean'] + 3 * df['precio_std'], 'outlier_by_mean'] = 'normal'
        return df
    except Exception as e:
        raise
