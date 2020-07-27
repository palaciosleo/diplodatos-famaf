from unidecode import unidecode
import pandas as pd
import numpy as np
import re
import time

spanish_stopwords = ['ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde',
                     'durante', 'en', 'entre', 'hacia', 'hasta', 'mediante',
                     'para', 'por', 'segun', 'sin', 'sobre', 'tras', 'la', 'las', 'los', 'del', 'el', 'a', 'y']


def get_mean(df):
    """
    Funcion que calcula la media y desviacion estandar por 'producto_id' y 'fecha'
    e inserta estas nuevas columnas al dataset.
    :param df: Recibe el dataframe de PRECIOS
    :return: Dataframe original con las columnas ['mean' , 'std']
    """
    try:
        start = time.time()
        precio_mean = df.groupby(['producto_id', 'fecha']).agg(precio_mean=('precio', 'mean'))
        stop = time.time()
        print("get_mean:", round(stop-start, 3), "segs")
        return pd.merge(df, precio_mean, left_on=['producto_id', 'fecha'], right_index=True)
    except Exception as e:
        raise


def get_df_precios_a_borrar(dict_precio):
    """
    Recorre el diccionario toma por cada key el 'precio_mean_diff' de mayor valor.
    :param dict_precio: Diccionario de ('producto_id','fecha'): [precio_mean_diff]
    :return: Dataframe de producto_id, fecha y max(precio_mean_diff)
    """
    try:
        df = pd.DataFrame()
        for idx, diff in dict_precio.items():
            df = df.append({'producto_id': idx[0], 'fecha': idx[1], 'precio_mean_diff': max(diff)}, ignore_index=True)
        return df
    except Exception as e:
        raise


def drop_precios_sin_sucursal(df):
    """

    :param df:
    :return:
    """
    try:
        print("DROP PRECIOS SIN SUCURSALES")
        reg_df = len(df)
        print("Cantidad de Registros del Dataframe:", reg_df)
        df = df[df['id'].notna()]
        print("Cantidad de Registros del Dataframe Limpio:", len(df))
        print('Se han limpiado', (reg_df - len(df)), 'registros')
        return df
    except Exception as e:
        raise


def drop_precios_duplicados(df):
    """
    Limpia del dataframe de PRECIOS aquellos registros cuyo 'producto_id', 'fecha' y 'sucursal_id' son iguales,
    manteniendo aquel registro cuya diferencia con la media por 'producto_id' y 'fecha' es  menor.
    No se utiliza la media por sucursal ya que al haber 2 productos mal reportados, la distancia de ambos con
    la media es la misma.
    :param df: Dataframe de PRECIOS
    :return: Dataframe de PRECIOS depurado
    """
    try:
        start = time.time()
        print("DROP DE PRECIOS DUPLICADOS")
        reg_df = len(df)
        print("Cantiad de Registros del Dataframe:", reg_df)
        df = get_mean(df)
        df['precio_mean_diff'] = abs(df['precio'] - df['precio_mean'])

        precios_duplicados = df[df.duplicated(subset=['producto_id', 'fecha', 'sucursal_id'], keep=False)]

        dict_precio_mean_diff = precios_duplicados.groupby(['producto_id', 'fecha'])['precio_mean_diff'].apply(
            lambda x: x.tolist()).to_dict()

        df_precios_duplicados_borrar = get_df_precios_a_borrar(dict_precio_mean_diff)

        for idx, row in df_precios_duplicados_borrar.iterrows():
            filtro = (df.producto_id == row.producto_id) \
                     & (df.fecha == row.fecha) \
                     & (df.precio_mean_diff == row.precio_mean_diff)

            df = df[~filtro]

        print("Cantidad de Registros del Dataframe Limpio:", len(df))
        print('Se han limpiado', (reg_df - len(df)), 'registros')

        stop = time.time()
        return df
    except Exception as e:
        raise


def get_quantiles(df):
    """
        Calcula cuantil 25% y 75% para los datos agrupados por 'producto_id', 'fecha' y 'region'
    :param df: Dataframe de Precios y Sucursales
    :return: La union de dataframe anterior con el resultante de los datos agrupados
    """
    try:
        start = time.time()
        grp_quantile = df.groupby(['producto_id', 'fecha', 'region']).agg(
                                                                cuartil_25=('precio', lambda x: np.quantile(x, .25)),
                                                                cuartil_75=('precio', lambda x: np.quantile(x, .75)))

        stop = time.time()
        print('get_quantiles:', round(stop - start, 3), "segs")

        return pd.merge(df, grp_quantile, left_on=['producto_id', 'fecha', 'region'],
                                                            right_on=['producto_id', 'fecha', 'region'], how='left')
    except Exception as e:
        raise


def is_outlier(df):
    try:
        df.loc[df['precio'] > df['cuartil_75'] + (3 * (df['cuartil_75'] - df['cuartil_25'])),
                                                                                    'rdo_ri_geo'] = 'extremo superior'
        df.loc[df['precio'] < df['cuartil_25'] - (3 * (df['cuartil_75'] - df['cuartil_25'])),
                                                                                    'rdo_ri_geo'] = 'extremo inferior'

        df.loc[(['dfcuartil_75'] + 3 * (df['cuartil_75'] - df['cuartil_25']) >= df['precio']) &
               (df['precio'] >= df['cuartil_25'] - 3 * (df['cuartil_75'] - df['cuartil_25'])), 'rdo_ri_geo'] = 'normal'

        df.loc[(df['cuartil_25'] == df['cuartil_75']), 'rdo_ri_geo'] = 'normal'
        return df
    except Exception as e:
        raise


def drop_outliers_precios_sucursales(df):
    """
    Elimina aquellos registros cuyo precio marcamos como outlier segun el producto_id, fecha y region
    :param df: Dataframe unido de precios y sucursales
    :return: Dataframe de PRECIOS depurado
    """
    try:
        start = time.time()

        print("DROP DE OUTLIERS DE PRECIOS_SUCURSALES")
        reg_df = len(df)
        print("Cantiad de Registros del Dataframe:", reg_df)

        df = get_quantiles(df)
        df = is_outlier(df)
        df = df[df['rdo_ri_geo'] == 'normal']

        print("Cantidad de Registros del Dataframe Limpio:", len(df))
        print('Se han limpiado', (reg_df - len(df)), 'registros')
        stop = time.time()
        print("drop_outliers_precios_sucursales:", round(stop - start, 3), "segs")
        return df
    except Exception as e:
        raise


def limpiar_palabras(columna, limpiar_stopwords=True):
    """

    :param columna:
    :param limpiar_stopwords:
    :return:
    """
    try:
        columna_limpia = []

        for contenido in columna:
            contenido = contenido.lower()
            contenido = unidecode(contenido)

            if limpiar_stopwords:
                palabra_limpia = ''
                palabras = contenido.split(' ')

                for palabra in palabras:
                    if palabra not in spanish_stopwords:
                        palabra_limpia += palabra.strip() + ' '
                columna_limpia.append(palabra_limpia.strip())
            else:
                columna_limpia.append(contenido)

        return columna_limpia
    except Exception as e:
        raise


def get_dummy_column(column, top_rank=10, default_word='others'):
    """
    Esta funcion acepta una columna y una lista de palabras mas frecuentes
    Con ello, crea una nueva lista la cual contiene la palabra original (si se encuentra dentro de las mas frecuentes)
    o le pone la palabra suministrada
    :param column: Columna sobre la cual aplicar la "dummyzacion"
    :param top_rank: Cantidad de palabras a considerar para la "dummyzacion"
    :param default_word: Se utiliza esta palabra como valor por defecto si no se encuentra dentro de las mas frecuentes
    :return: Columna con palabras dummys
    """
    try:
        dummy_list = column.value_counts().reset_index()[:top_rank]['index'].tolist()
        dummy_column = []
        for row in column:
            valor_dummy = default_word
            if row in dummy_list:
                valor_dummy = row
            dummy_column.append(valor_dummy)

        return dummy_column
    except Exception as e:
        raise


def get_provincia_dummy(df):
    """

    :param df:
    :return:
    """
    try:
        start = time.time()
        df['provincia_depurada'] = df['nom_provincia'].str.lower()
        for letra, reemplazo in zip(['á', 'é', 'í', 'ó', 'ú'], ['a', 'e', 'i', 'o', 'u']):
            df['provincia_depurada'] = df['provincia_depurada'].str.replace(letra, reemplazo)
        df['provincia_depurada'] = df['provincia_depurada'].str.replace(' ', '_')
        stop = time.time()
        print("get_provincia_dummy:", round(stop - start, 3), "segs")
        return df
    except Exception as e:
        raise


def get_banderaDescripcion_dummy(df):
    """

    :param df_sucursales:
    :return:
    """
    try:
        start = time.time()
        df['banderaDescripcion_depurado'] = df['banderaDescripcion'].str.lower()

        for letra, reemplazo in zip(['á', 'é', 'í', 'ó', 'ú'], ['a', 'e', 'i', 'o', 'u']):
            df['banderaDescripcion_depurado'] = df['banderaDescripcion_depurado'].str.replace(letra, reemplazo)

        df['banderaDescripcion_depurado'] = df['banderaDescripcion_depurado'].str.replace('.', '')
        df['banderaDescripcion_depurado'] = df['banderaDescripcion_depurado'].str.replace(' ', '_')

        df['banderaDescripcion_dummy'] = get_dummy_column(df['banderaDescripcion_depurado'], 15, 'otras_bandDesc')
        stop = time.time()

        print("get_sucursales_dummy:", round(stop - start, 3), "segs")
        return df
    except Exception as e:
        raise


def get_marca_dummy(df_productos):
    """
    Devuelve el Dataframe con la columna de "marca_dummy" lista.
    :param df_productos: Dataframe de PRODUCTOS
    :return: Devuelve el Dataframe con la columna de "marca_dummy" lista
    """
    try:
        start = time.time()
        df = df_productos.copy()
        df['marca_depurada'] = limpiar_palabras(df['marca'])
        df['marca_depurada'] = df['marca_depurada'].apply(lambda x: '_'.join(x.split(' ')))
        df['marca_dummy'] = get_dummy_column(df['marca_depurada_unida'], 10, 'otras_marcas')
        stop = time.time()
        print("get_marca_dummy:", round(stop - start, 3), "segs")
        return df
    except Exception as e:
        raise


def get_idreferencia(df):
    """
    Funcion que extrae el ID de Referencia para solucionar el problema de inconsistencia
    en donde un mismo producto cuyo ('nombre, 'marca', 'presentacion') son iguales
    :param df: Recibe el dataframe de PRODUCTOS
    :return: columna con ID de Referencia
    """
    try:
        start = time.time()
        diccionario = df.groupby(['nombre', 'marca', 'presentacion'])['id'].apply(lambda x: x.tolist()).to_dict()
        stop = time.time()
        print("get_idreferencia:", round(stop-start, 3),"segs")
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


def get_initial_cleanup(df_productos):
    """
    Realizar una serie de limpiezas y extracciones iniciales
    :param df_productos: Dataframe PRODUCTOS
    :return: Dataframe con nuevas columnas
    """
    try:
        start = time.time()
        df = df_productos.copy()
        df['nombre_depurado'] = df['nombre'].str.lower().fillna('')
        df['presentacion_depurada'] = df['presentacion'].str.lower()
        df['um_en_presentacion'] = df['presentacion_depurada'].str[-2:]
        df['cantidad_en_presentacion'] = df['presentacion_depurada'].str[0:-3]
        stop = time.time()
        print("get_initial_cleanup:", round(stop-start, 3),"segs")
        return df
    except Exception as e:
        raise


def get_um_presentacion_from_nombre(df):
    """
    Extraer la 'um' y 'presentacion' del nombre del producto
    :param df: Dataframe de PRODUCTOS
    :return:   Columnas ['um_en_nombre_prod', 'cant_en_nombre_prod']
    """
    try:
        start = time.time()
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
        stop = time.time()
        print("get_um_presentacion_from_nombre:", round(stop-start, 3),"segs")
        return cant_en_nombre_prod, um_en_nombre_prod

    except Exception as e:
        raise


def get_um_fixed(df):
    """
    Aplica un set expresiones regulares para arreglar algunas 'um'
    :param df: Dataframe de PRODUCTOS
    :return: Dataframe con columna 'um_en_nombre_prod' corregida
    """
    try:
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


def is_pack(nombre_producto):
    # En el peor de los casos, si la palabra 'pack' esta al ultimo de la cadena, la funcion demora "23.5 µs per loop"
    for palabra in nombre_producto.split(' '):
        if re.search(r'^pack', palabra) is not None:
            return True
    return False


def is_pack(nombre_producto):
    """
    Funcion auxiliar para detectar si una cadena tiene la palabra 'pack'
    :param nombre_producto:
    :return: True si la palabra contiene la regex 'pack'
    """
    try:
        for palabra in nombre_producto.split(' '):
            if re.search(r'^pack', palabra) is not None:
                return True
        return False
    except Exception as e:
        raise


def get_presentacion_limpia(df):
    """
    Obtiene la mejor presentacion posible de aquel producto cuya presentacion original y la presentacion
    presente en el nombre no coincida
    :param df: Dataframe PRODUCTOS
    :return: Columnas ['um_limpia', 'cant_limpia]
    """
    try:
        start = time.time()
        um_limpia = []
        cant_limpia = []

        pack_pattern = r'^.*\s\d{1,3}\s{0,1}\w{1,3}\s[y\+]\s.*\d{1,3}\s{0,1}\w{1,3}.*$'
        un_pattern = r'(?P<cant>[\d]+)\sun'

        for idx, producto in df.iterrows():
            producto_um_limpia = ''
            producto_cant_limpia = ''

            if is_pack(producto['nombre_depurado']):
                producto_um_limpia = 'pack'
                producto_cant_limpia = 1

            elif re.search(pack_pattern, producto['nombre_depurado']) is not None:
                producto_um_limpia = 'pack'
                producto_cant_limpia = 1

            elif producto['um_en_presentacion'] == producto['um_en_nombre_prod']:
                producto_um_limpia = producto['um_en_presentacion']

                if float(producto['cantidad_en_presentacion']) == float(producto['cant_en_nombre_prod']):
                    producto_cant_limpia = producto['cantidad_en_presentacion']
                else:
                    producto_cant_limpia = producto['cant_en_nombre_prod']

            elif float(producto['cantidad_en_presentacion']) == float(producto['cant_en_nombre_prod']):
                producto_cant_limpia = producto['cant_en_nombre_prod']
                if producto['um_en_nombre_prod'] in df['um_en_presentacion'].unique():
                    producto_um_limpia = producto['um_en_nombre_prod']
                else:
                    producto_um_limpia = producto['um_en_presentacion']

            else:
                extraer_unid = re.search(un_pattern, producto['nombre_depurado'])
                if extraer_unid is not None:
                    producto_um_limpia = 'un'
                    producto_cant_limpia = extraer_unid.group(
                        'cant')  ## el 'cant' sale de un "extractor" que se define en el patron.
                elif producto['um_en_presentacion'] == 'un':
                    producto_um_limpia = producto['um_en_presentacion']
                    producto_cant_limpia = producto['cantidad_en_presentacion']

                elif producto['um_en_nombre_prod'] in df['um_en_presentacion'].unique():
                    producto_um_limpia = producto['um_en_nombre_prod']
                    producto_cant_limpia = producto['cant_en_nombre_prod']

                else:
                    producto_um_limpia = producto['um_en_presentacion']
                    producto_cant_limpia = producto['cantidad_en_presentacion']

            um_limpia.append(producto_um_limpia)
            cant_limpia.append(float(producto_cant_limpia))

        stop = time.time()
        print("get_presentacion_limpia:", round(stop-start, 3),"segs")
        return um_limpia, cant_limpia

    except Exception as e:
        raise