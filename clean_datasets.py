import logging
import os
from logging.handlers import RotatingFileHandler

import numpy as np
import pandas as pd
from io import StringIO
import re

import custom_tools as tools
import import_datasets as impds
logger = logging.getLogger(__name__)


def setup_logger(log_path, log_name):
    log_file = '.'.join([log_name, 'log'])
    handler = RotatingFileHandler(filename=os.path.join(log_path, log_file), maxBytes=2 * 1024 * 1024, backupCount=1)
    handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR)


def main():
    try:
        productos = impds.get_productos_df()
        sucursales = impds.get_sucursales_df()
        precios = impds.get_precios_df()

        productos = productos[productos['marca'].notna()]
        precios = precios[precios['precio'].notna()]

        productos['id_referencia'] = tools.get_idreferencia(productos)
        precios = tools.get_mean_std(precios)

        precios = tools.get_outlier_by_mean(precios)

        # Llamo a la funcion extractora de 'um' y 'cantidad' del nombre_depurado
        cant_en_nombre_prod, um_en_nombre_prod = tools.get_um_presentacion_from_nombre(productos)
        # Guardo el resultado de la funcion en las respectivas NUEVAS columnas
        productos['cant_en_nombre_prod'] = cant_en_nombre_prod
        productos['um_en_nombre_prod'] = um_en_nombre_prod

        productos = tools.get_um_fixed(productos)
        print('a')

        # Combinar Datasets y eliminar filas que no matcheen
        # Precios con Sucursales
        # precios[~precios['sucursal_id'].isin(sucursales_sin_id)]
    except Exception as e:
        logger.error('%s | %s', 'main', str(e))


if __name__ == '__main__':
    script_path = os.path.dirname(__file__)
    script_name = os.path.basename(__file__).split('.')[0]
    setup_logger(script_path, script_name)
    main()
