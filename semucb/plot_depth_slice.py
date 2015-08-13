import sys
import numpy as np
import tables
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.spatial as spatial
import ssrfpy

import cartopy.crs as ccrs


def get_radius_list( node_array ):
  assert isinstance( node_array, tables.Array)
  radii = np.array([ np.sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]) for n in node_array.iterrows() ])
  hist, bin_edges = np.histogram( radii, bins=1000) # 1000 should probably be enough

  dr = bin_edges[1]-bin_edges[0] #width of bins
  cutoff = 0.2*np.max(hist)

  radius_list = np.array([ bin_edges[i]+dr/2.0 for i in range(len(hist)) if hist[i] > cutoff ])

  return radius_list, dr

def get_closest_radius( r, radius_list ):
  return radius_list[np.argmin( np.abs(r-radius_list))]


def depth_slice_array( node_array, field_array, r, dr):
  assert isinstance( node_array, tables.Array)
  assert isinstance( field_array, tables.Array)

  #pull all the entries with radius within dr/2 of r
  reduced_data = [ (n,f) for (n,f) in zip(node_array.iterrows(), field_array.iterrows()) if np.abs(np.sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]) - r) < dr/2.0 ]

  #Make an array of the cartesian coordinates
  xyz_coord = np.array([ x[0] for x in reduced_data ])

  #Convert to longitude and latitude
  lon =  np.arctan2( xyz_coord[:, 1], xyz_coord[:,0] ) * 180.0/np.pi
  lon[ np.where( lon < 0.0 ) ] += 360.0 #Get the longitudes from 0.-360.
  lat =  90.0-np.arctan2( np.sqrt(xyz_coord[:,0]*xyz_coord[:,0] + xyz_coord[:,1]*xyz_coord[:,1]), xyz_coord[:,2] ) * 180.0/np.pi
  #grab the field values
  values = np.array( [x[1] for x in reduced_data] ).ravel()

  return lon, lat, values

def interpolate_regular_grid( lon, lat, values):

  nvals = len(values)
  resolution = np.rad2deg(np.sqrt((4.0*np.pi)/nvals))

  x = np.cos( np.deg2rad(lon) )*np.cos(np.deg2rad(lat))
  y = np.sin( np.deg2rad(lon) )*np.cos(np.deg2rad(lat))
  z = np.sin(np.deg2rad(lat))

  reg_lon = np.linspace(0., 360., 360./resolution)
  reg_lat = np.linspace(-90., 90., 180./resolution)


  mesh_lon, mesh_lat = np.meshgrid( reg_lon, reg_lat ) 
  x_eval = np.cos( np.deg2rad(mesh_lon.flatten()) )*np.cos(np.deg2rad(mesh_lat.flatten()))
  y_eval = np.sin( np.deg2rad(mesh_lon.flatten()) )*np.cos(np.deg2rad(mesh_lat.flatten()))
  z_eval = np.sin(np.deg2rad(mesh_lat.flatten()))

  tree = spatial.cKDTree( zip(x,y,z) )
  d, inds = tree.query(zip(x_eval, y_eval, z_eval), k = 1)

  #get interpolated 2d field
  val_nearest = np.array([values[i] for i in inds])
  val_nearest = val_nearest.reshape( mesh_lon.shape )
  
  return mesh_lon, mesh_lat, val_nearest


def get_data_handle( step_number, field_name):

    meshfile = 'mesh-'+str(step_number).zfill(5) +'.h5'
    fieldfile = 'solution-'+str(step_number).zfill(5) +'.h5'

    mesh_data = tables.open_file( meshfile, mode='r')
    field_data = tables.open_file( fieldfile, mode='r')
    nodes = mesh_data.root.nodes
    field = getattr(field_data.root, field_name)

    return nodes, field

def spherical_harmonic_transform( mesh_vals ):
    #mesh_vals should be on a GL grid
    sys.path.append('/home/ian/SHTOOLS')
    import pyshtools
    zero, w = pyshtools.SHGLQ( mesh_vals.shape[0] )
    cilm = pyshtools.SHExpandGLQ( mesh_vals, zero, w )
    print cilm.shape

if __name__ == '__main__':
    print("Loading data handles")
    step_number = 0
    if len(sys.argv) > 2:
        step_number = sys.argv[1]
    nodes, field = get_data_handle( step_number, 'vertical_heat_flux' )

    print("Getting radius list")
    radius_list, dr = get_radius_list(nodes)
    evaluation_radius = get_closest_radius(3600.e3,radius_list)

    print("Evaluating depth slice")
    lon, lat, values = depth_slice_array(nodes,field, evaluation_radius, dr)
    print("Interpolating field")
    mesh_lon, mesh_lat, mesh_val = ssrfpy.interpolate_regular_grid(lon,lat,values,n=180)


    print("plotting")
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    image = ax.pcolormesh(mesh_lon, mesh_lat, mesh_val, transform=ccrs.PlateCarree())
    #image = ax.contourf(mesh_lon, mesh_lat, mesh_val,  120,
    #                    transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines()

    plt.colorbar(image)
    plt.show()
