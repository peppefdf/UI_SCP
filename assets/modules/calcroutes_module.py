# -*- coding: utf-8 -*-
"""CalcRoutes_module.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ssMnCVBO67zaY5wKZA5nB9CLeImRSs38
"""

#%reset
#%pip install cplex
#%pip install docplex
#%pip install osmnx
#%pip install dash
#%pip install dash-leaflet

import numpy as np
from docplex.mp.model import Model
import random
from numpy.random import randint

import osmnx as ox
import networkx as nx
import json

import itertools

import pandas as pd
import geopandas
from geopy import distance # Biblioteca para calculos geograficos
from geopy.geocoders import Nominatim
from geopy.point import Point
from shapely.geometry import Polygon  

#import folium
#from folium import PolyLine
#import matplotlib.pyplot as plt

import datetime

#from dash import Dash
#import dash_leaflet as dl

#from google.colab import drive
#drive.mount('/content/drive')

# directory where GTFS files will be saved
#directory = '/content/drive/MyDrive/Colab Notebooks/CSL_GIPUZKOA/Proyecto Piloto_Eskuzaitzeta/GTFS_files/'

# Read the GTFS files
#stops = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/CSL_GIPUZKOA/Proyecto Piloto_Eskuzaitzeta/DATOS_GIPUZKOA/gtfs_Zubieta/gtfs/stops.txt', delimiter=',')
#stop_times = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/CSL_GIPUZKOA/Proyecto Piloto_Eskuzaitzeta/DATOS_GIPUZKOA/gtfs_Zubieta/gtfs/stop_times.txt', delimiter=',')
#trips = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/CSL_GIPUZKOA/Proyecto Piloto_Eskuzaitzeta/DATOS_GIPUZKOA/gtfs_Zubieta/gtfs/trips.txt', delimiter=',')
#routes = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/CSL_GIPUZKOA/Proyecto Piloto_Eskuzaitzeta/DATOS_GIPUZKOA/gtfs_Zubieta/gtfs/routes.txt', delimiter=',')


"""
# INPUTS
m_buses = 3 # number of available buses-> affects quality of solution for the Asymm. mTSP
#selec_trip_id = 32394  # 23 pts
#selec_trip_id =  32566 # 8 pts
selec_trip_id =  32402 # 13 pts -> just as an example, select the set of stops
#selec_trip_id =  32404 # 18 pts -> just as an example, select the set of stops
"""

tol = 1.5
tol_dist = 0.5 # % difference between longest and shortest route
cont_limit = 100 # max number of iterations for convergence

#CO2km = 1.12

def routes_list(A,m_b):
  ruta = {}
  for i in range(0, m_b):
    ruta['Bus_' + str(i+1)] = [0]
    j = i
    a = 10e10
    while a != 0:
        a = A[j,1]
        ruta['Bus_' + str(i+1)].append(a)
        #print(ruta['Bus_' + str(i+1)])
        j = np.where(A[:,0] == a)
        j = j[0][0]
        a = j
  return ruta

#def extract_RoutesMat(Xmat, Vec, n, pt, mb):
def extract_RoutesMat(Xmat, Vec, n, mb):
     paths0 = []
     out = np.zeros((n,n))
     #for (i, j) in zip(Vec, Vec):
     indeces = list(itertools.product(Vec, Vec))
     for ii in indeces:
        i = ii[0]
        j = ii[1]
        if Xmat[i][j] > 0:
           out[i][j] = Xmat[i][j]
           paths0.append([i,j])
     routes0 = routes_list(np.array(paths0),mb)
     return routes0, out

def check_Nbuses(Xmat):
  print('N buses going out from origin ={:3d}'.format(int(sum(Xmat[0,:]))))
  print('N buses returning to origin ={:3d}'.format(int(sum(Xmat[:,0]))))
  return

def check_UniqueStops(Xmat):
  for i in range(1,np.shape(Xmat)[0]):
     print('N routes out stop {:3d}: {:3d}'.format(i+1,int(sum(Xmat[i,:]))))
     print('N routes in stop {:3d}: {:3d}'.format(i+1,int(sum(Xmat[:,i]))))
  #print('Check all (should be 1):')
  #print((Xmat[1:,:].sum())/(np.shape(Xmat)[0]-1))
  return


"""
# Merge trips and routes to have the name and the trip_id together
routes = routes[["route_id","route_long_name"]]
trips = trips[["route_id","trip_id"]]
trips = pd.merge(trips, routes, on='route_id', how='left')
trips = trips.drop(columns='route_id')
#print(trips)

# Merge stop_times and stops to have the trip_id with the stop latitude and longitud
stops = stops[["stop_id","stop_lat","stop_lon"]]
stop_times = stop_times[["trip_id","stop_id",]]
stops_coord = pd.merge(stop_times, stops, on='stop_id', how='left')
stops_coord = stops_coord.drop(columns='stop_id')
#print(stops_coord)

# Merge the two DataFrames
df = pd.merge(stops_coord, trips, on='trip_id', how='left')
#print(df)

Zubieta_routes = df[df['route_long_name'].str.contains('Zubieta', case=False)] # El parámetro case=False ignora mayúsculas y minúsculas
#print("Zubieta routes:")
#print(Zubieta_routes)

# Iterar a través de los trip_id únicos y guardar el trip seleccionado
for trip_id in Zubieta_routes['trip_id'].unique():
    print(trip_id)
    trip_df = Zubieta_routes[Zubieta_routes['trip_id'] == trip_id]
    print(len(trip_df))
    if trip_id ==  selec_trip_id:
       pts = list(zip(trip_df['stop_lat'], trip_df['stop_lon']))
"""

def CalcRoutes_module(puntos,m_buses,CO2km):
      ################################################
      # Calculando la matriz de distancias
      ################################################
      n = len(puntos)
      C = np.zeros((n,n))

      """
      for i in range(0, n):
          for j in range(0, len(puntos)):
              C[i,j] = distance.distance(puntos[i], puntos[j]).km
      """

      ori_coord = puntos[0]
      
      print()
      print('Generating graph...')
      lats, lons = map(list, zip(*puntos))
      max_lat = max(lats)
      min_lat = min(lats)      
      max_lon = max(lons)
      min_lon = min(lons)
      """
      df = pd.DataFrame({'lat':lats, 'lon':lons})
      gdf = geopandas.GeoDataFrame(
          df, geometry=geopandas.points_from_xy(lons, lats), crs="EPSG:4326"
      )
      poly_convex_hull = gdf['geometry'].unary_union.convex_hull 
      #G = ox.graph_from_point(ori_coord, dist=40000, network_type="drive", simplify=True, retain_all=False)
      G = ox.graph_from_polygon(poly_convex_hull, network_type="drive", simplify=True, retain_all=False)
      """
      #G = ox.graph_from_bbox(max_lat*1.05,min_lat*0.95,max_lon*0.95,min_lon*1.05, network_type="drive", simplify=True, retain_all=False) 
      G = ox.graph_from_bbox(min_lat*0.99,max_lat*1.01,min_lon*1.01,max_lon*0.99, network_type="drive", simplify=True, retain_all=False) 

      print('Graph completed!')
      print()
      print('Adding edge speeds, lengths and travelling speeds...')
      #hwy_speeds = {"residential": 30, "secondary": 30, "tertiary": 30}
      hwy_speeds = {"residential": 20, "unclassified": 30, "maxspeed": 50 }
      #hwy_speeds = {"primary": 20, "residential": 20, "unclassified": 30, "maxspeed": 50 }
      G = ox.add_edge_speeds(G, hwy_speeds)
      #G = ox.add_edge_speeds(G)
      G = ox.add_edge_travel_times(G)
      G = ox.distance.add_edge_lengths(G)
      print('Adding edge speeds, lengths and travelling speeds completed!')

      print()
      print('Calculating distance matrix...')
      for i in range(n):
        origin = puntos[i]
        origin_node = ox.distance.nearest_nodes(G, [origin[1]], [origin[0]])[0]
        for j in range(n):
          destination = puntos[j]
          destination_node = ox.distance.nearest_nodes(G, [destination[1]], [destination[0]])[0]
          #Get the shortest path
          path_length = nx.shortest_path_length(G, origin_node, destination_node, weight='length')
          C[i][j] = path_length/1000
      print('Distance matrix calculated!')

      # Mostrando la matriz de distancias
      print('La matriz de distancia es:\n')
      print(np.round(C,4))
      n = np.shape(C)[0]

      model=Model('mTSP')

      ## Variable xij
      stops=range(n)
      x=model.binary_var_matrix(keys1=stops,keys2=stops,name='x')
      ## Varible ui
      u=model.integer_var_list(keys=stops, lb=0, ub=n,name='u')

      model.minimize(model.sum(C[i,j] * x[i,j] for i in stops for j in stops))
      model.add_constraint(model.sum(x[0 , j] for j in stops if j>0)==m_buses)
      model.add_constraint(model.sum(x[i , 0] for i in stops if i>0)==m_buses)

      for i in stops[1:]:
              model.add_constraint(model.sum(x[i , j] for j in stops )==1)

      for j in stops[1:]:
              model.add_constraint(model.sum(x[i , j] for i in stops )==1)

      for i in stops:
              model.add_constraint(x[i,i] == 0)

      Ms = int(n/m_buses) # Ms = max number of stops visited by each bus.
                          # Choose Ms = n/m_buses for a balanced load distribution among buses
      #Ms = n
      for i in stops[1:]:
          for j in stops[1:]:
              if i != j:
                model.add_constraint(u[i]-u[j]+ Ms * x[i,j] <= Ms -1)
                #model.add_constraint(u[i]-u[j] + 1 <= (Ms -1)*(1-x[i,j]) )

      model.add_constraint(u[0] == 1)
      for i in stops[1:]:
            model.add_constraint(u[i] >= 2)
      for i in stops[1:]:
            model.add_constraint(u[i] <= n) #---> Ms or n? With Ms algorithm keeps searching...


      # ## Solve
      solution = model.solve(log_output=False)
      print('Objective function:')
      print(solution.get_objective_value())
      #solution.display()
      X_sol = np.zeros((n,n))
      for index, dvar in enumerate(solution.iter_variables()):
          if dvar.to_string().split('_')[0] == 'x':
            row, col = dvar.to_string().split('_')[1:]
            X_sol[int(row),int(col)] = solution[dvar]
            #print(index, dvar, dvar.to_string().split('_'), solution[dvar], solution.get_var_value(dvar))
      ruta_EZ0, Xout = extract_RoutesMat(X_sol, stops, n, m_buses)
      check_Nbuses(Xout)
      check_UniqueStops(Xout)

      print(ruta_EZ0)
      #print(Xout)

      ruta_EZ0 = list(ruta_EZ0.values())
      cumul_dist_routes = []
      for ii in range(len(ruta_EZ0)):
          dist_temp = []
          for jj in range(len(ruta_EZ0[ii])-1):
              i0 = ruta_EZ0[ii][jj]
              i1 = ruta_EZ0[ii][jj+1]
              dist_temp.append(C[i0][i1])
          cumul_dist_routes.append(sum(dist_temp))
      print('Route cumulative lengths:')
      print(cumul_dist_routes)
      ind_max = cumul_dist_routes.index(max(cumul_dist_routes))
      ind_min = cumul_dist_routes.index(min(cumul_dist_routes))
      print()

      stop = 0
      cont = 0
      converged = 0
      objective = []
      # generate 1D list:
      ruta_EZ0_old = [0 for row in ruta_EZ0 for item in row]
      while not stop:
          if solution:
              X_sol = np.zeros((n,n))
              for index, dvar in enumerate(solution.iter_variables()):
                  if dvar.to_string().split('_')[0] == 'x':
                    row, col = dvar.to_string().split('_')[1:]
                    X_sol[int(row),int(col)] = solution[dvar]
                    #print(index, dvar, dvar.to_string().split('_'), solution[dvar], solution.get_var_value(dvar))
              ruta_EZ0, Xout = extract_RoutesMat(X_sol, stops, n, m_buses)
              print('new routes:')
              print(ruta_EZ0)

              # find index of route with max cumulative length #######################
              # find index of route with min cumulative length
              ruta_EZ0 = list(ruta_EZ0.values())
              cumul_dist_routes = []
              for ii in range(len(ruta_EZ0)):
                  dist_temp = []
                  for jj in range(len(ruta_EZ0[ii])-1):
                      i0 = ruta_EZ0[ii][jj]
                      i1 = ruta_EZ0[ii][jj+1]
                      dist_temp.append(C[i0][i1])
                  cumul_dist_routes.append(sum(dist_temp))
              print('Route cumulative lengths:')
              print(cumul_dist_routes)
              ind_max = cumul_dist_routes.index(max(cumul_dist_routes))
              ind_min = cumul_dist_routes.index(min(cumul_dist_routes))
              ########################################################################

              if cont < cont_limit and not converged:
                C_max_list = []
                x_max_list = []
                for ii in range(len(ruta_EZ0[ind_max])-1):
                    i0 = ruta_EZ0[ind_max][ii]
                    i1 = ruta_EZ0[ind_max][ii+1]
                    C_max_list.append(C[i0][i1])
                    x_max_list.append((i0,i1))

                C_min_list = []
                x_min_list = []
                for ii in range(len(ruta_EZ0[ind_min])-1):
                    i0 = ruta_EZ0[ind_min][ii]
                    i1 = ruta_EZ0[ind_min][ii+1]
                    C_min_list.append(C[i0][i1])
                    x_min_list.append((i0,i1))

                model.add_constraint(  ( (model.sum(C_max_list[ii]*x[x_max_list[ii][0],x_max_list[ii][1]] for ii in range(len(x_max_list))) - model.sum(C_min_list[ii]*x[x_min_list[ii][0],x_min_list[ii][1]] for ii in range(len(x_min_list)))) <= tol_dist*model.sum(C_max_list[ii]*x[x_max_list[ii][0],x_max_list[ii][1]] for ii in range(len(x_max_list))) ) )
                #model.add_constraint(model.sum(C_max_list[ii]*x[x_max_list[ii][0],x_max_list[ii][1]] for ii in range(len(x_max_list))) <= 60 )
                #model.add_constraint(model.sum(C_min_list[ii]*x[x_min_list[ii][0],x_min_list[ii][1]] for ii in range(len(x_min_list))) >= 1 )
                #print('indices max:')
                #print(x_max_list)
                #print('indices min:')
                #print(x_min_list)

                try:
                    solution0 = model.solve(log_output=False)
                    #solution = model.solve(clean_before_solve=True,log_output=False) #---> try this
                    print('Objective function:')
                    print(solution0.get_objective_value())
                    objective.append(solution0.get_objective_value())
                    solution = solution0
                    constr_exists = 1
                    cont_constr = 0
                    """
                    # does the following show the number of cumulated constraints?
                    while(constr_exists is not None):
                        print('constraint: ')
                        constr_exists = model.get_constraint_by_index(cont_constr)
                        print(constr_exists)
                        cont_constr+=1
                    """
                    dmax = np.sum(np.array(C_max_list))
                    print('Cumulative distance of route covering Max dist:')
                    print(dmax)
                    dmin = np.sum(np.array(C_min_list))
                    print('Cumulative distance of route covering Min dist:')
                    print(dmin)
                    print()

                    # flatten list of routes to 1D list:
                    ruta_EZ0 = [item for row in ruta_EZ0 for item in row]
                    # check whether calculation has reached convergence
                    if ruta_EZ0 == ruta_EZ0_old:
                      converged = 1
                    ruta_EZ0_old = ruta_EZ0

                except:
                    print('Solution not found!!!')
                    stop = 1

              else:
                print('Solution converged or maximum number of iteration reached!')
                stop = 1

              cont+=1

          else:
              print('Solution not found at iteration: ', cont+1)
              stop = 1
      #solution.display()
      #solution.get_objective_value()


      # CO2 calc and route visualization #############################################
      routes = []
      total_CO2 = 0.0
      coords_routes = []
      for ii in range(len(ruta_EZ0)):
          dist_temp = []
          length_route_i = 0
          coords_route_i = []
          for jj in range(len(ruta_EZ0[ii])-1):
                i0 = ruta_EZ0[ii][jj]
                i1 = ruta_EZ0[ii][jj+1]
                origin = puntos[i0]
                destination = puntos[i1]
                origin_node      = ox.distance.nearest_nodes(G, [origin[1]], [origin[0]])[0]
                destination_node = ox.distance.nearest_nodes(G, [destination[1]], [destination[0]])[0]
                route_i = nx.shortest_path(G, origin_node, destination_node, weight='length')
                length_route_i_temp = nx.shortest_path_length(G, origin_node, destination_node, weight='length')
                dist_temp.append(route_i)
                length_route_i = length_route_i + length_route_i_temp
                for node_id in route_i:
                    Lon = G.nodes[node_id]['x'] #lon
                    Lat = G.nodes[node_id]['y'] #lat
                    coords_route_i.append((Lat,Lon))
          routes.append(dist_temp)
          coords_routes.append(coords_route_i)
          print('Length route (m) '+str(ii)+':',length_route_i)
          print('CO2 emissions of route '+str(ii)+':',(1./1000)*length_route_i*CO2km)
          total_CO2 = total_CO2 + (1./1000)*length_route_i*CO2km
          print()
      print()
      print('Total CO2 emissions: ',total_CO2)

      #print(coords_routes[0])
      return coords_routes

"""
LatsLons_routes = CalcRoutes_module(pts)
app = Dash()
app.layout = dl.Map([
    dl.TileLayer(),
    dl.Polyline(positions=LatsLons_routes[0])
    ],
    center=LatsLons_routes[0][0], zoom=10, style={'height': '50vh'})

if __name__ == '__main__':
    #app.run_server(Debug=True)
    app.run_server(port=8052,Debug=True)
"""