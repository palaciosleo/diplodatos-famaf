import pandas as pd
from io import StringIO


def get_sucursales_df():
    try:
        sucursal_url = 'https://raw.githubusercontent.com/solujan/mentoria_2020/master/raw_dataset/sucursales.csv'
        sucursales = pd.read_csv(sucursal_url)
        provincia_txt = """provincia	nom_provincia	region
        AR-A	Salta	Norte Grande
        AR-B	Provincia de Buenos Aires	Centro
        AR-C	Ciudad Autónoma de Buenos Aires	Centro
        AR-D	San Luis	Cuyo
        AR-E	Entre Ríos	Centro
        AR-F	La Rioja	Cuyo
        AR-G	Santiago del Estero	Norte Grande
        AR-H	Chaco	Norte Grande
        AR-J	San Juan	Cuyo
        AR-K	Catamarca	Norte Grande
        AR-L	La Pampa	Centro
        AR-M	Mendoza	Cuyo
        AR-N	Misiones	Norte Grande
        AR-P	Formosa	Norte Grande
        AR-Q	Neuquén	Patagonia
        AR-R	Río Negro	Patagonia
        AR-S	Santa Fe	Centro
        AR-T	Tucumán	Norte Grande
        AR-U	Chubut	Patagonia
        AR-V	Tierra del Fuego	Patagonia
        AR-W	Corrientes	Norte Grande
        AR-X	Córdoba	Centro
        AR-Y	Jujuy 	Norte Grande
        AR-Z	Santa Cruz	Patagonia
        """
        provincia_csv = StringIO(provincia_txt)
        entidad_provincia = pd.read_csv(provincia_csv, sep=('\t'))
        entidad_provincia['provincia'] = entidad_provincia['provincia'].str.strip()
        sucursales = sucursales.merge(entidad_provincia, on='provincia')

        return sucursales
    except Exception as e:
        raise


def get_productos_df():
    try:
        producto_url = 'https://raw.githubusercontent.com/solujan/mentoria_2020/master/raw_dataset/productos.csv'
        productos = pd.read_csv(producto_url)

        return productos
    except Exception as e:
        raise


def get_precios_df():
    try:
        precios_20200412_20200413 = pd.read_csv(
            'https://raw.githubusercontent.com/solujan/mentoria_2020/master/raw_dataset/precios_20200412_20200413.csv')
        precios_20200419_20200419 = pd.read_csv(
            'https://raw.githubusercontent.com/solujan/mentoria_2020/master/raw_dataset/precios_20200419_20200419.csv')
        precios_20200426_20200426 = pd.read_csv(
            'https://raw.githubusercontent.com/solujan/mentoria_2020/master/raw_dataset/precios_20200426_20200426.csv')
        precios_20200502_20200503 = pd.read_csv(
            'https://raw.githubusercontent.com/solujan/mentoria_2020/master/raw_dataset/precios_20200502_20200503.csv')
        precios_20200518_20200518 = pd.read_csv(
            'https://raw.githubusercontent.com/solujan/mentoria_2020/master/raw_dataset/precios_20200518_20200518.csv')

        lista_df_px = [precios_20200412_20200413, precios_20200419_20200419, precios_20200426_20200426,
                       precios_20200502_20200503, precios_20200518_20200518]
        fecha_px = ['20200412', '20200419', '20200426', '20200502', '20200518']

        precios = pd.DataFrame()
        for df, fecha in zip(lista_df_px, fecha_px):
            df['fecha'] = fecha
            precios = pd.concat([precios, df])

        return precios
    except Exception as e:
        raise
