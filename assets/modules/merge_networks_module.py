import numpy as np
#import pandas as pd
import os
import glob
from pathlib import Path
import re
import requests
from io import StringIO
import random
import pdb

#import matplotlib
#matplotlib.use('agg')  # allows notebook to be tested in Travis

import pandas as pd
#import cartopy.crs as ccrs
#import cartopy
#import matplotlib.pyplot as plt
#import pandana as pdna
import time

#!pip install pandana
import pandana as pdna

# !pip install cartopy
# import cartopy.crs as ccrs
# import cartopy

#!pip install urbanaccess
import urbanaccess as ua
from urbanaccess.config import settings
from urbanaccess.gtfsfeeds import feeds
from urbanaccess import gtfsfeeds
from urbanaccess.gtfs.gtfsfeeds_dataframe import gtfsfeeds_dfs
from urbanaccess.network import ua_network, load_network

import matplotlib
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)

q=8

# # Load data to an Urban Access TRANSIT data object

t0 = time.time()
# I manually add all the feeds from the bus transit companies of Gipuzkoa
"""
feeds.add_feed(add_dict={'dbus': 'https://www.geo.euskadi.eus/cartografia/DatosDescarga/Transporte/Moveuskadi/ATTG/dbus/google_transit.zip'})
feeds.add_feed(add_dict={'lurraldebus_areizaga': 'https://www.geo.euskadi.eus/cartografia/DatosDescarga/Transporte/Moveuskadi/ATTG/lurraldebus_areizaga/google_transit.zip'})
feeds.add_feed(add_dict={'lurraldebus_arrasate': 'https://www.geo.euskadi.eus/cartografia/DatosDescarga/Transporte/Moveuskadi/ATTG/lurraldebus_arrasate/google_transit.zip'})
feeds.add_feed(add_dict={'lurraldebus_auif_urb': 'https://www.geo.euskadi.eus/cartografia/DatosDescarga/Transporte/Moveuskadi/ATTG/lurraldebus_auif_urb/google_transit.zip'})
feeds.add_feed(add_dict={'lurraldebus_eibar': 'https://www.geo.euskadi.eus/cartografia/DatosDescarga/Transporte/Moveuskadi/ATTG/lurraldebus_eibar/google_transit.zip'})
feeds.add_feed(add_dict={'lurraldebus_ekialdebus': 'https://www.geo.euskadi.eus/cartografia/DatosDescarga/Transporte/Moveuskadi/ATTG/lurraldebus_ekialdebus/google_transit.zip'})
feeds.add_feed(add_dict={'lurraldebus_euskotren_bus': 'https://www.geo.euskadi.eus/cartografia/DatosDescarga/Transporte/Moveuskadi/ATTG/lurraldebus_euskotren_bus/google_transit.zip'})
feeds.add_feed(add_dict={'lurraldebus_garayar': 'https://www.geo.euskadi.eus/cartografia/DatosDescarga/Transporte/Moveuskadi/ATTG/lurraldebus_garayar/google_transit.zip'})
feeds.add_feed(add_dict={'lurraldebus_gipuzkoana': 'https://www.geo.euskadi.eus/cartografia/DatosDescarga/Transporte/Moveuskadi/ATTG/lurraldebus_gipuzkoana/google_transit.zip'})
feeds.add_feed(add_dict={'lurraldebus_goierrialdea': 'https://www.geo.euskadi.eus/cartografia/DatosDescarga/Transporte/Moveuskadi/ATTG/lurraldebus_goierrialdea/google_transit.zip'})
feeds.add_feed(add_dict={'lurraldebus_hernani': 'https://www.geo.euskadi.eus/cartografia/DatosDescarga/Transporte/Moveuskadi/ATTG/lurraldebus_hernani/google_transit.zip'})
feeds.add_feed(add_dict={'lurraldebus_lasarte': 'https://www.geo.euskadi.eus/cartografia/DatosDescarga/Transporte/Moveuskadi/ATTG/lurraldebus_lasarte/google_transit.zip'})
feeds.add_feed(add_dict={'lurraldebus_oiartzun': 'https://www.geo.euskadi.eus/cartografia/DatosDescarga/Transporte/Moveuskadi/ATTG/lurraldebus_oiartzun/google_transit.zip'})
feeds.add_feed(add_dict={'lurraldebus_pesa': 'https://www.geo.euskadi.eus/cartografia/DatosDescarga/Transporte/Moveuskadi/ATTG/lurraldebus_pesa/google_transit.zip'})
feeds.add_feed(add_dict={'lurraldebus_renteria': 'https://www.geo.euskadi.eus/cartografia/DatosDescarga/Transporte/Moveuskadi/ATTG/lurraldebus_renteria/google_transit.zip'})
feeds.add_feed(add_dict={'lurraldebus_tbh': 'https://www.geo.euskadi.eus/cartografia/DatosDescarga/Transporte/Moveuskadi/ATTG/lurraldebus_tbh/google_transit.zip'})
feeds.add_feed(add_dict={'lurraldebus_tolosaldea': 'https://www.geo.euskadi.eus/cartografia/DatosDescarga/Transporte/Moveuskadi/ATTG/lurraldebus_tolosaldea/google_transit.zip'})
feeds.add_feed(add_dict={'lurraldebus_tsst': 'https://www.geo.euskadi.eus/cartografia/DatosDescarga/Transporte/Moveuskadi/ATTG/lurraldebus_tsst/google_transit.zip'})
feeds.add_feed(add_dict={'lurraldebus_zarautz': 'https://www.geo.euskadi.eus/cartografia/DatosDescarga/Transporte/Moveuskadi/ATTG/lurraldebus_zarautz/google_transit.zip'})
feeds.add_feed(add_dict={'Euskotren': 'https://www.geo.euskadi.eus/cartografia/DatosDescarga/Transporte/Moveuskadi/Euskotren/google_transit.zip'})
feeds.add_feed(add_dict={'Renfe': 'https://www.geo.euskadi.eus/cartografia/DatosDescarga/Transporte/Moveuskadi/Renfe/google_transit.zip'})
feeds.add_feed(add_dict={'Renfe_cercanias': 'https://www.geo.euskadi.eus/cartografia/DatosDescarga/Transporte/Moveuskadi/Renfe_Cercanias/google_transit.zip'})
#feeds.add_feed(add_dict={'SCP_routes': '.\data\gtfsfeed_zips\routes.zip'})
# I download all these feeds, 70sg
gtfsfeeds.download()
#feeds.add_feed(add_dict={'SCP_routes': '..model\data\gtfsfeed_zips'})
"""

# Load GTFS data into an UrbanAcess transit data object
validation = True
verbose = True
# bbox for Gipuzkoa
Lat_min = 42.904155
Long_min = -2.621987
Lat_max = 43.403070
Long_max = -1.740334
bbox = (Long_min,Lat_min,Long_max,Lat_max,)
remove_stops_outsidebbox = True
append_definitions = True

filen = 'transit_ped_net.h5'
root_dir = 'C:/Users/gfotidellaf/repositories/UI_SCP/assets/data/input_data_MCM/networks/'
gtfsfeed_path = 'C:/Users/gfotidellaf/repositories/UI_SCP/assets/data/input_data_MCM/GTFS_feeds/'

timerange=['08:00:00', '09:00:00']

urbanaccess_net = ua.network.ua_network

loaded_feeds = ua.gtfs.load.gtfsfeed_to_df(gtfsfeed_path=gtfsfeed_path,
                                           validation=validation,
                                           verbose=verbose,
                                           bbox=bbox,
                                           remove_stops_outsidebbox=remove_stops_outsidebbox,
                                           append_definitions=append_definitions)

# Create transit network from existing feeds
ua.gtfs.network.create_transit_net(gtfsfeeds_dfs=loaded_feeds,
                                   day='monday',
                                   timerange=timerange,
                                   calendar_dates_lookup=None)


if (os.path.isfile(root_dir + filen)):
    
    print('\nLoading saved network...') 
    #saved_transit_net = ua.network.load_network(filename='saved_network_h5/integrated_base_network.h5')
    #ped_net = ua.network.load_network(filename='saved_network_h5/pedestrian_net.h5')
    transit_ped_net = ua.network.load_network(dir=root_dir, filename=filen)
    print('done!\n')

    # Crear la red de OSM (pedestrian)
    print('\nCreating pedestrian network from saved file...')
    ua.osm.network.create_osm_net(osm_edges=transit_ped_net.net_edges, osm_nodes=transit_ped_net.net_nodes, travel_speed_mph=3)
    print('done!\n')

    print('\nIntegrating saved network with new transit network...')
    # Integrate saved and newly created tansit network
    #ua.network.integrate_network(urbanaccess_network=saved_transit_net,
    #                         headways=False) 
    ua.network.integrate_network(urbanaccess_network=urbanaccess_net,
                                    headways=False)     

    # Add average headways to network travel time
    ua.gtfs.headways.headways(gtfsfeeds_df=loaded_feeds,
                            headway_timerange=timerange)
    loaded_feeds.headways

    ua.network.integrate_network(urbanaccess_network=urbanaccess_net,
                                headways=True,
                                urbanaccess_gtfsfeeds_df=loaded_feeds,
                                headway_statistic='mean')  
    print('successfully integrated existing and newly created networks!')
    """
    ua.plot.plot_net(nodes=urbanaccess_net.net_nodes,
                 edges=urbanaccess_net.net_edges,
                 bbox=bbox,
                 fig_height=30, margin=0.02,
                 edge_color='#999999', edge_linewidth=1, edge_alpha=1,
                 node_color='black', node_size=1.1, node_alpha=1, node_edgecolor='none', node_zorder=3, nodes_only=False)
    """
else:

    nodes, edges = ua.osm.load.ua_network_from_bbox(bbox=bbox, remove_lcn=True)
    # Crear la red de OSM (pedestrian)
    ua.osm.network.create_osm_net(osm_edges=edges, osm_nodes=nodes, travel_speed_mph=3)
    # integrate transit and pedestrian networks
    
    # Add average headways to network travel time
    ua.gtfs.headways.headways(gtfsfeeds_df=loaded_feeds,
                            headway_timerange=timerange)
    loaded_feeds.headways

    ua.network.integrate_network(urbanaccess_network=urbanaccess_net,
                                headways=True,
                                urbanaccess_gtfsfeeds_df=loaded_feeds,
                                headway_statistic='mean')  

    #urbanaccess_net.net_nodes['zone_id'] = urbanaccess_net.net_nodes['zone_id'].astype(str)

    integrated_net_pdna = pdna.network.Network(
                               urbanaccess_net.net_nodes["x"],
                               urbanaccess_net.net_nodes["y"],
                               urbanaccess_net.net_edges["from_int"],
                               urbanaccess_net.net_edges["to_int"],
                               urbanaccess_net.net_edges[["weight","distance"]])

    integrated_net_pdna.save_hdf5(root_dir + filen)
    
t1 = time.time()
print('Total time:')
print((t1-t0)/60)
"""
# Crear un diccionario con dos columnas, una para el rango de tiempo y otra para el nombre del archivo
q=8
# Definir los valores dados
timeranges = [
    ['00:00:00', '01:00:00'], ['01:00:00', '02:00:00'], ['02:00:00', '03:00:00'], ['03:00:00', '04:00:00'],
    ['04:00:00', '05:00:00'], ['05:00:00', '06:00:00'], ['06:00:00', '07:00:00'], ['07:00:00', '08:00:00'],
    ['08:00:00', '09:00:00'], ['09:00:00', '10:00:00'], ['10:00:00', '11:00:00'], ['11:00:00', '12:00:00'],
    ['12:00:00', '13:00:00'], ['13:00:00', '14:00:00'], ['14:00:00', '15:00:00'], ['15:00:00', '16:00:00'],
    ['16:00:00', '17:00:00'], ['17:00:00', '18:00:00'], ['18:00:00', '19:00:00'], ['19:00:00', '20:00:00'],
    ['20:00:00', '21:00:00'], ['21:00:00', '22:00:00'], ['22:00:00', '23:00:00'], ['23:00:00', '24:00:00']
]
nombres_archivos = [
    'transit_0001.h5', 'transit_0102.h5', 'transit_0203.h5', 'transit_0304.h5',
    'transit_0405.h5', 'transit_0506.h5', 'transit_0607.h5', 'transit_0708.h5',
    'transit_0809.h5', 'transit_0910.h5', 'transit_1011.h5', 'transit_1112.h5',
    'transit_1213.h5', 'transit_1314.h5', 'transit_1415.h5', 'transit_1516.h5',
    'transit_1617.h5', 'transit_1718.h5', 'transit_1819.h5', 'transit_1920.h5',
    'transit_2021.h5', 'transit_2122.h5', 'transit_2223.h5', 'transit_2324.h5'
]
timeranges=[timeranges[q]]
nombres_archivos=[nombres_archivos[q]]
print(timeranges)
print(nombres_archivos)

# # Definir los valores dados
# timeranges = [
#     ['04:00:00', '05:00:00'], ['05:00:00', '06:00:00'], ['06:00:00', '07:00:00'], ['07:00:00', '08:00:00'],
# ]
# nombres_archivos = [
#     'transit_0405.h5', 'transit_0506.h5', 'transit_0607.h5', 'transit_0708.h5',
# ]


for timerange, nombre_archivo in zip(timeranges, nombres_archivos):
    # Create transit network

    transit_net = ua.gtfs.network.create_transit_net(gtfsfeeds_dfs=loaded_feeds,day='monday',timerange=timerange,calendar_dates_lookup=None)
    urbanaccess_net = ua.network.ua_network

    # Create integrated network
    ua.network.integrate_network(urbanaccess_network=urbanaccess_net,
                           headways=False)

    # Add average headways to network travel time
    ua.gtfs.headways.headways(gtfsfeeds_df=loaded_feeds,
                          headway_timerange=timerange)
    loaded_feeds.headways

    ua.network.integrate_network(urbanaccess_network=urbanaccess_net,
                             headways=True,
                             urbanaccess_gtfsfeeds_df=loaded_feeds,
                             headway_statistic='mean')

    # Create
    s_time = time.time()
    transit_ped_net = pdna.network.Network(urbanaccess_net.net_nodes["x"],
                               urbanaccess_net.net_nodes["y"],
                               urbanaccess_net.net_edges["from_int"],
                               urbanaccess_net.net_edges["to_int"],
                               urbanaccess_net.net_edges[["weight"]],
                               twoway=False)
    print('Took {:,.2f} seconds'.format(time.time() - s_time))

    # Save
    transit_ped_net.save_hdf5(f'C:/Users/gfotidellaf/input_data/transit_together_24h/{nombre_archivo}')

    # Save picture
    # edgecolor = ua.plot.col_colors(df=urbanaccess_net.net_edges, col='weight', cmap='gist_heat_r', num_bins=5)
    # ua.plot.plot_net(nodes=urbanaccess_net.net_nodes,
    #                    edges=urbanaccess_net.net_edges[urbanaccess_net.net_edges['net_type']=='transit'],
    #                    bbox=bbox,
    #                    fig_height=30, margin=0.02,
    #                    edge_color=edgecolor, edge_linewidth=1, edge_alpha=1,
    #                    node_color='black', node_size=0, node_alpha=1, node_edgecolor='none', node_zorder=3, nodes_only=False)

    # # Guardar la figura en el disco. Asegúrate de especificar la ruta completa y el nombre de archivo deseado.
    # plt.savefig(f'/content/drive/MyDrive/Mobility_Choice/input_data/transit_together_24h/images/{nombre_archivo}.pdf', dpi=300)
"""