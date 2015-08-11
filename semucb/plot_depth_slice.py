import sys
import numpy as np
import tables
import matplotlib.pyplot as plt
import scipy.interpolate as interp

import cartopy.crs as ccrs


def get_radius_list( node_array ):
  assert isinstance( node_array, tables.Array)
  radii = np.array([ np.sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]) for n in node_array.iterrows() ])
  hist, bin_edges = np.histogram( radii, bins=1000) # 1000 should probably be enough

  dr = bin_edges[1]-bin_edges[0] #width of bins
  cutoff = 0.2 * np.max(hist) # ignore radii with fewer hits than this

  radii_list = np.array([ bin_edges[i]+dr/2.0 for i in range(len(hist)) if hist[i] > cutoff ])

  return radii_list, dr


def depth_slice_array( node_array, field_array, r, dr):
  assert isinstance( node_array, tables.Array)
  assert isinstance( field_array, tables.Array)

  #pull all the entries with radius within dr/2 of r
  reduced_data = [ (n,f) for (n,f) in zip(node_array.iterrows(), field_array.iterrows()) if np.abs(np.sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]) - r) < dr/2.0 ]

  #Make an array of the cartesian coordinates
  xyz_coord = np.array([ x[0] for x in reduced_data ])

  #Convert to longitude and latitude
  lon =  180.0+np.arctan2( xyz_coord[:, 1], xyz_coord[:,0] ) * 180.0/np.pi
  lat =  90.0-np.arctan2( np.sqrt(xyz_coord[:,0]*xyz_coord[:,0] + xyz_coord[:,1]*xyz_coord[:,1]), xyz_coord[:,2] ) * 180.0/np.pi
  #grab the field values
  values = np.array( [x[1] for x in reduced_data] ).ravel()

  return lon, lat, values
  

mesh_data = tables.open_file( sys.argv[1], mode='r')
field_data = tables.open_file( sys.argv[2], mode='r')
nodes = mesh_data.root.nodes
field = field_data.root.T

radii_list, dr = get_radius_list(nodes)
lon, lat, values = depth_slice_array(nodes,field, radii_list[2], dr)

reg_lon = np.linspace(0, 360, 181)
reg_lat = np.linspace(-90,90,91)
gridlon, gridlat = np.meshgrid( reg_lon, reg_lat)
gridvals = interp.griddata( (lon,lat), values, (gridlon, gridlat) , method='nearest')

ax = plt.axes(projection=ccrs.PlateCarree())

ax.contourf(gridlon, gridlat, gridvals,  120,
             transform=ccrs.PlateCarree())

ax.coastlines()
ax.gridlines()

plt.show()
