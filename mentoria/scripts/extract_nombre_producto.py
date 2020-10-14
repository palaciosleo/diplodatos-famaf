import numpy as np
import pandas as pd

from mentoria.scripts import custom_tools as tools, import_datasets as impds

productos = impds.get_productos_df()
productos = productos[productos['marca'].notna()]

productos = tools.get_initial_cleanup(productos)
cant_en_nombre_prod, um_en_nombre_prod = tools.get_um_presentacion_from_nombre(productos)
productos = productos.drop(columns=['categoria1', 'categoria2', 'categoria3'])
productos['cant_en_nombre_prod'] = cant_en_nombre_prod
productos['um_en_nombre_prod'] = um_en_nombre_prod

um_limpia, cant_limpia = tools.get_presentacion_limpia(productos)

productos['um_limpia'] = um_limpia
productos['cant_limpia'] = cant_limpia


productos_dummy_df = tools.get_marca_producto_dummy(productos, 3000)
productos.drop(columns=['marca', 'nombre', 'presentacion', 'nombre_depurado',
       'presentacion_depurada', 'um_en_presentacion',
       'cantidad_en_presentacion', 'cant_en_nombre_prod', 'um_en_nombre_prod',
       'um_limpia', 'cant_limpia', 'marca_depurada', 'nombre_marca'], inplace=True)

productos = pd.merge(productos, productos_dummy_df, left_index=True, right_index=True)

pd.to_pickle(productos, '../models/productos_3000.pkl', compression="zip", protocol=4)