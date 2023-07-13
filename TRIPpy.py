#!/usr/bin/env python3.7
# encoding: utf-8
"""
TRIPpy v1.0.0
@author: Omar V. Müller, ovmuller@gmail.com
"""

import iris
import numpy as np
import geopy.distance as gd
import time
from numba import jit
import gc
import f90nml
from warnings import filterwarnings
from tqdm import tqdm

def main():
    #############
    # TRIP MAIN #
    #############

    start_time = time.time()
    global Ct, k, sequ, dire, ne_y, ne_x, mask, length, dt, nt, ny, nx, ns, nlp
    global outflow_cube, storage_cube, storage, outflow
    global runoff_regrid, inflow, filename, version

    version = 'v1.0'
    
    filterwarnings('ignore')
 
    print('\n-------------------')
    print('Running TRIPpy '+version)
    print('-------------------\n')
    
    # SET MODEL CONFIGURATION
    set_configuration() # To modify the configuration the user can edit namelist.input
    
    print('\nForcing path: ' + forcing_path)
    print('Parameters path: ' + params_path) 
    print('Output path: ' + output_path)
    print('Spinup: %d cycles of %d years' %(spinup_cycles,spinup_years))
    print('Period: %d - %d ' %(start,end))
    print('Restart: ' + str(restart))
    if restart: print('Restart from file: '+restart_file)
    
    # LOAD RIVER ANCIHMARY FILES AND CALCULATE
    direction = iris.load(params_path+params_ifile,'flow_direction')[0]
    sequence = iris.load(params_path+params_ifile,'flow_sequence')[0]

    # CALCULATE LENGTH AND NEXT DOWNSTREAM POINT
    if preproc:
        print('\nPreprocessing parameters for TRIPpy')
        set_params(direction,antarctica,params_path+params_ofile)
    
    length = iris.load(params_path+params_ofile,'TRIPpy_length_to_next_downstream_point')[0]
    next_y = iris.load(params_path+params_ofile,'TRIPpy_index_of_next_downstream_point_in_latitude')[0]
    next_x = iris.load(params_path+params_ofile,'TRIPpy_index_of_next_downstream_point_in_longitude')[0]

    # take data for faster processing
    sequ = sequence.data
    dire = direction.data
    ne_y = next_y.data
    ne_x = next_x.data
    mask = np.ma.masked_equal(sequ,0.)

    # LOAD RUNOFF DATA FOR FIRST YEAR
    runoff_flux = iris.load_cube(forcing_path+forcing_prefix+str(start)+'.nc','runoff_flux')

    # DEFINE TRIP TIMESTEP IN SECS
    t1 = runoff_flux.coord('time').points[1]
    t2 = runoff_flux.coord('time').points[2]
    timestep = (t2-t1)*86400.
    print('Timestep: '+str(timestep/60)+' min\n')
    dt = river_timestep*timestep

    # GET DIMENSIONS TO ITERATE
    nt = len(runoff_flux.coord('time').points)
    ny = len(direction.coord('latitude').points)
    nx = len(direction.coord('longitude').points)
    ns = int(np.max(sequence.data))+1
    
    # CALCULATE TRIP COEFFICIENTS
    Ct, k = set_coef(ny,nx,dt,length,speed,meander)
    
    # INITIALISE ARRAYS AND CUBES
    storage = np.zeros((nt,ny,nx))
    t0 = 0
        
    # SPINUP
    # If restart>0, then SPINUP is not run as spinup_cycles and spinup_years = 0
    for spc in range(spinup_cycles):
        for spy in range(spinup_years):
        
            print('Running spinup  %d/%d' % (spc*spinup_years+spy+1,spinup_years*spinup_cycles))
            # LOAD RUNOFF DATA FOR A GIVEN YEAR
            print('Loading '+forcing_prefix+str(start+spy)+'.nc')
            runoff_flux = iris.load_cube(forcing_path+forcing_prefix+str(start+spy)+'.nc','runoff_flux')
            # remake_ECEarth_coords(runoff_flux)
            runoff_flux.data[runoff_flux.data.mask] = 0. # set 0 over ocean before interpolate    
        
            # REGRID RUNOFF TO TRIP GRID
            runoff_flux_regrid = runoff_flux.regrid(direction,iris.analysis.Nearest())
            if not runoff_flux_regrid.coord('latitude').has_bounds():
                runoff_flux_regrid.coord('latitude').guess_bounds()
                runoff_flux_regrid.coord('longitude').guess_bounds()
            nt = len(runoff_flux.coord('time').points)
    
            # CONVERT RUNOFF UNITS [kg m-2 s-1] --> [kg s-1]
            area = iris.analysis.cartography.area_weights(runoff_flux_regrid)
            runoff_regrid = runoff_flux_regrid*area
            runoff_regrid.units = ('kg s-1')          

            # PREPARE STORAGE FOR YEARS DIFFERENT TO FIRST
            storage_old = np.copy(storage[-1])
    
            # RUN MODEL
            filename = output_path+output_prefix+'spincycle'+str(spc)+'_spinyear'+str(spy)
            storage = run_year(storage_old,runoff_regrid,filename,t0)

        
    # SIMULATION
    # Change initial conditions if RESTART
    if restart:
        storage_2d = iris.load_cube(restart_file,'storage')[-1]
        storage[-1] = storage_2d.data
        t0 = int(restart_file[-6:-3])+1
        if t0 == nt: t0 = 0        
    
    # Run simulation
    for yr in range(start,end+1):

        # LOAD RUNOFF DATA FOR A GIVEN YEAR
        runoff_flux = iris.load_cube(forcing_path+forcing_prefix+str(yr)+'.nc','runoff_flux')
        # remake_ECEarth_coords(runoff_flux)
        runoff_flux.data[runoff_flux.data.mask] = 0. # set 0 over ocean before interpolate    
        
        # REGRID RUNOFF TO TRIP GRID
        runoff_flux_regrid = runoff_flux.regrid(direction,iris.analysis.Nearest())
        if not runoff_flux_regrid.coord('latitude').has_bounds():
            runoff_flux_regrid.coord('latitude').guess_bounds()
            runoff_flux_regrid.coord('longitude').guess_bounds()
        nt = runoff_flux_regrid.data.shape[0]
    
        # CONVERT RUNOFF UNITS [kg m-2 s-1] --> [kg s-1]
        area = iris.analysis.cartography.area_weights(runoff_flux_regrid)
        runoff_regrid = runoff_flux_regrid*area
        runoff_regrid.units = ('kg s-1')          

        # PREPARE STORAGE FOR YEARS DIFFERENT TO FIRST
        storage_old = np.copy(storage[-1])
    
        # RUN TRIP YEARS
        print('Running year %d' % (yr))
        filename = output_path+output_prefix+str(yr)
        storage = run_year(storage_old,runoff_regrid,filename,t0)
        t0 = 0 # if restarted t!=0, so it should be set to 0 for remaining years
        gc.collect()

    print("--- %s minutes ---" % ((time.time() - start_time)/60.))

def set_configuration():
    '''
    This function set the configuration that can be defined by the user.
    The function read variables in namelist.input and set default values
    when the user do not define a variable. 
    The final configuration is stored in namelist.output
    '''


    global forcing_path, forcing_prefix, output_prefix, output_path, params_path
    global params_ifile, preproc, restart, restart_file, spinup_cycles, spinup_years, start, end
    global params_ofile, speed, meander, river_timestep, antarctica

    print('Reading namelist.input')
    nml = f90nml.read('namelist.input')

    nml['configuration'].setdefault('forcing_path', './data/forcings/')
    nml['configuration'].setdefault('forcing_prefix', 'land_mrro_Lmon_HadGEM3-GC31-HM_hist-1950_r1i1p1f1_gn_')
    nml['configuration'].setdefault('output_path', './data/output/')
    nml['configuration'].setdefault('output_prefix', 'rivers_qd_HadGEM3-GC31-LM_hist-1950_r1i1p1f1_gn_')
    nml['configuration'].setdefault('params_path', './data/params/')
    nml['configuration'].setdefault('params_ifile', 'river_network.nc')
    nml['configuration'].setdefault('preproc', False)
    nml['configuration'].setdefault('params_ofile', 'aux_params.nc')
    nml['configuration'].setdefault('restart', False) # if True CHECK spinup and start
    nml['configuration'].setdefault('restart_file', '')
    nml['configuration'].setdefault('spinup_years', 1) # number of spinup years per cycle
    nml['configuration'].setdefault('spinup_cycles', 1)# number of spinup cycles
    nml['configuration'].setdefault('start', 1950) # start year, the simulation starts on 1st Jan
    nml['configuration'].setdefault('end', 1950)   # end year, the simulation ends on 30th Dec
    nml['configuration'].setdefault('speed', 0.5)
    nml['configuration'].setdefault('meander', 1.4)
    nml['configuration'].setdefault('river_timestep', 1)
    nml['configuration'].setdefault('antarctica','False')
                                    
    forcing_path = nml['configuration'].get('forcing_path')
    forcing_prefix = nml['configuration'].get('forcing_prefix')
    output_path = nml['configuration'].get('output_path')
    output_prefix = nml['configuration'].get('output_prefix')
    params_path = nml['configuration'].get('params_path')
    params_ifile = nml['configuration'].get('params_ifile')
    preproc = nml['configuration'].get('preproc')
    restart = nml['configuration'].get('restart')
    restart_file = nml['configuration'].get('restart_file')
    spinup_years = nml['configuration'].get('spinup_years')
    spinup_cycles = nml['configuration'].get('spinup_cycles')
    start = nml['configuration'].get('start')
    end = nml['configuration'].get('end')
    params_ofile = nml['configuration'].get('params_ofile')
    speed = nml['configuration'].get('speed')
    meander = nml['configuration'].get('meander')
    river_timestep = nml['configuration'].get('river_timestep')
    antarctica = nml['configuration'].get('antarctica')
    
    if restart == 0:
        restart_file = ''
        nml['configuration']['restart_file'] = ''
    elif spinup_years != 0 or spinup_cycles !=0:
        spinup_years = 0
        spinup_cycles = 0
        nml['configuration']['spinup_years'] = 0 
        nml['configuration']['spinup_cycles'] = 0 
        print('Spinup is not possible when restart a simulation: spinup_years and spinup_cycles are set to 0.')
        
    print('Writing namelist.output.TRIPpy_'+version+' to the output directory')
    nml.write(output_path+'namelist.output.TRIPpy_'+version, force=True)

def remake_ECEarth_coords(cube):
   import iris

   lon_c = cube.coord('longitude')
   lat_c = cube.coord('latitude')
   lats = lat_c.points[:,0]
   lons = lon_c.points[0,:]
   new_lon_c = iris.coords.DimCoord(lons,standard_name = lon_c.standard_name,
                                    var_name = lon_c.var_name,units = lon_c.units)
   new_lon_c.guess_bounds()
   new_lat_c = iris.coords.DimCoord(lats,standard_name = lat_c.standard_name,
                                    var_name = lat_c.var_name,units = lat_c.units)
   new_lat_c.guess_bounds()
   coords = [coord.name() for coord in cube.coords() if coord.name() not in ['time','air_pressure','pressure']]
   for coord in coords:
       cube.remove_coord(coord)
   dim_shape = cube.shape
   cube.add_dim_coord(new_lon_c,len(dim_shape)-1)
   cube.add_dim_coord(new_lat_c,len(dim_shape)-2)

@jit
def set_params(direction, antarctica, filename):
    '''
    This function set and store TRIPpy fixed parameters.
    
    input: 
        direction: direction of flow (clockwise starting with 1 for N)
        antarctica: flag to indicate if Antarctica is included in the domain
        filename: nc file to store the output
    output:
        params_list: list of cubes in netcdf filename with:
        0)riverslength: the length [m] between the current and the downstream point
        1)next_y: the vertical index of next downstream point
        2)next_x: the horizontal index of next downstream point
    '''

    lat1 = direction.coord('latitude').points[1]
    lat2 = direction.coord('latitude').points[2]
    dlat = np.abs(lat2-lat1)

    lon1 = direction.coord('longitude').points[1]
    lon2 = direction.coord('longitude').points[2]
    dlon = np.abs(lon2-lon1)

    min_lo = direction.coord('longitude').points[0]
    max_lo = direction.coord('longitude').points[-1]

    target_cube = direction.copy()
    target_cube.attributes = {'Data_source': 'This data was created by TRIPpy '+version+' based on the flow direction file', \
                              'Developer': 'Omar V. Muller from CONICET and UNL, ovmuller@gmail.com', \
                              'Reference': 'Muller et al., 2021: Does HadGEM3-GC3.1 GCM overestimate ' \
                              'land precipitation at high resolution? A constraint based on observed ' \
                              'river discharge. J. Hydrometorol.'}

    riverslength = target_cube.copy()
    riverslength.var_name = 'length'
    riverslength.long_name = 'TRIPpy_length_to_next_downstream_point'
    riverslength.units = ('m')

    next_y = target_cube.copy()
    next_y.var_name = 'next_y'
    next_y.long_name = 'TRIPpy_index_of_next_downstream_point_in_latitude'

    next_x = target_cube.copy()
    next_x.var_name = 'next_x'
    next_x.long_name = 'TRIPpy_index_of_next_downstream_point_in_longitude'

    # loop over latitude
    for la in tqdm(direction.coord('latitude').points):
        #print(la)
        # loop over longitude
        for lo in direction.coord('longitude').points:
            # check if river is active on this grid-cell
            dirlalo = int(direction.intersection(latitude=(la,la),longitude=(lo,lo),ignore_bounds=True).data[0,0])
            if (dirlalo > 0):
                y = np.argwhere(direction.coord('latitude').points == la)[0,0]
                x = np.argwhere(direction.coord('longitude').points == lo)[0,0]
                if dirlalo == 1: # N
                    next_la = la + dlat
                    next_lo = lo
                if dirlalo == 2: # NE
                    next_la = la + dlat
                    next_lo = lo + dlon
                if dirlalo == 3: # E
                    next_la = la
                    next_lo = lo + dlon
                if dirlalo == 4: # SE
                    next_la = la - dlat
                    next_lo = lo + dlon
                if dirlalo == 5: # S
                    next_la = la - dlat
                    next_lo = lo
                if dirlalo == 6: # SW
                    next_la = la - dlat
                    next_lo = lo - dlon
                if dirlalo == 7: # W
                    next_la = la 
                    next_lo = lo - dlon
                if dirlalo == 8: # NW
                    next_la = la + dlat
                    next_lo = lo - dlon
                if dirlalo == 9 or dirlalo == 12: # RIVER MOUTH or LAKE
                    next_la = la
                    next_lo = lo + dlon # taken from rivers_route_utils_mod.F90 line 741
                if next_lo>max_lo:
                    next_lo=next_lo-360.
                if next_lo<min_lo:
                    next_lo=next_lo+360.
                    
                if next_la < -90. and antarctica:
                    next_la = la
                    if next_lo < 180:
                        next_lo = next_lo + 180.
                    else:
                        next_lo = next_lo - 180



                riverslength.data[y,x] = gd.geodesic((la,lo),(next_la,next_lo)).m
                # In HR models it is an issue find the exact lat and lon 
                # given the little differences in decimals
                # To avoid this issue the code get the nearest lat and lon
                next_y.data[y,x] = np.abs(direction.coord('latitude').points-next_la).argmin()
                next_x.data[y,x] = np.abs(direction.coord('longitude').points-next_lo).argmin()
                #print(y,x)

    params_list = iris.cube.CubeList([riverslength,next_y,next_x])
    print('Storing '+filename+'\n')
    iris.save(params_list,filename)

    return params_list

def set_coef(ny,nx,dt,length,speed,meander):
    '''
    This function set TRIPpy coefficients.
    
    input: 
        ny: lenght of latitudes array
        nx: lenght of longitudes array
        dt: delta time in seconds
        length: 2d array of length [m] between the current and the downstream point
        speed: flow speed [m/s]
        meander: meandering constant
    output:
        2d arrays of:
        0) Ct: exp(-c.dt)
        1) c=1/k
    '''
    
    Ct = np.zeros((ny,nx))
    k = np.zeros((ny,nx))

    # CALCULATE TRIP COEFFICIENTS
    # loop over longitude
    for y in range(ny):
        # loop over longitude
        for x in range(nx):
            if length.data[y,x]==0.:
                c=0.0
                Ct[y,x]=1.0
                k[y,x]=0.
            else:
                c=speed/(length.data[y,x]*meander)
                Ct[y,x]=np.exp(-(c*dt))
                k[y,x]=(1-Ct[y,x])/c
            
    return Ct, k

@jit
def run_timestep(inflow,storage_old):
    '''
    This function run one TRIPpy timestep.
    
    input: 
        inflow: 1d land-only runoff [kg/s] array 
        storage_old: 1d land-only storage [kg] array of previous timestep
    output:
        1d land-only arrays of:
        0) inflow [kg/s]
        1) storage [kg]
        2) outflow [kg/s]
    '''
    storage = np.ma.zeros((ny,nx))
    outflow = np.ma.zeros((ny,nx))
    
    # loop over sequence
    for s in range(1,ns):

        # loop over landpoints
        for y in range(ny):
            for x in range(nx):

                # check if river active on this sequence step
                if ( int(sequ[y,x]) == s):
                    
                    # calculate channel storage at end of timestep (eq. 4 Oki 1999)
                    storage[y,x] = Ct[y,x] * storage_old[y,x] + k[y,x] * inflow[y,x]
                    # calculate outflow as inflow minus change in storage
                    outflow[y,x] = inflow[y,x] + (storage_old[y,x]-storage[y,x])/dt
                    
                    # if there is downstream point 
                    # assing current outflow to its inflow
                    y2=int(ne_y[y,x])
                    x2=int(ne_x[y,x])
                    if dire[y,x]<9.:
                        inflow[y2,x2] = inflow[y2,x2]+outflow[y,x]

    return inflow, storage, outflow

def save_timestep(runoff_regrid,inflow,outflow,storage,filename):
    '''
    This function store a netcdf per time-step of TRIPpy simulation, 
    converting 1d land-only arrays to 2d cube.
    
    input: 
        runoff_regrid: 3d runoff [kg/s] array
        inflow: 3d inflow [kg/s] array
        outflow: 3d outflow [kg/s] array
        storage: 3d storage [kg] array
        filename: netcdf filename to store the cubes
    output:
        3d arrays of:
        0) runoff [kg/s]
        1) inflow [kg/s] (it is runoff of current point + input from neighbors)
        2) storage [kg]
        3) outflow [kg/s]
    '''
    
    # Scalar coord time to Dimension coord time
    target_cube = iris.util.new_axis(runoff_regrid, 'time')
    target_cube.attributes = {'Data_source': 'This data was created by TRIPpy '+version, 
                              'Developer': 'Omar V. Muller from CONICET and UNL, ovmuller@gmail.com',
                              'Reference': "Muller et al., 2021: Does HadGEM3-GC3.1 GCM overestimate" \
                              'land precipitation at high resolution? A constraint based on observed ' \
                              'river discharge. J. Hydrometorol.'}
    
    outflow_cube = target_cube.copy()
    storage_cube = target_cube.copy()
    outflow_cube.var_name = 'outflow'
    storage_cube.var_name = 'storage'
    storage_cube.units = ('kg')

    outflow_cube.data[0] = np.ma.core.MaskedArray.copy(outflow)
    storage_cube.data[0] = np.ma.core.MaskedArray.copy(storage)

    # MASK OUT OCEAN POINTS
    outflow_cube.data[:].mask = mask.mask
    storage_cube.data[:].mask = mask.mask
    
    #output_list=iris.cube.CubeList([runoff_cube,inflow_cube,outflow_cube,storage_cube])
    output_list = iris.cube.CubeList([outflow_cube,storage_cube])

    iris.save(output_list,filename,zlib=True,complevel=9)

    return

def run_year(storage_old,runoff_regrid,filename,t0):
    '''
    This function run a year of TRIPpy simulation, and stores a netcdf per time-step
    
    input: 
        storage_old: 2d storage old [kg] array for first timestep
        runoff_regrid: 3d runoff [kg/s] array
        filename: netcdf filename to store the cubes
        t0: initial time-step, usually 0
    output:
        netcdf containing:
        0) runoff [kg/s]
        1) inflow [kg/s] (it is runoff of current point + input from neighbors)
        2) storage [kg]
        3) outflow [kg/s]
        storage: storage of last Lmon for next time-step
    '''
    
    global inflow, storage_0, outflow
    outflow = np.ma.zeros((nt,ny,nx))
    storage = np.ma.zeros((nt,ny,nx))
    
    inflow = np.ma.core.MaskedArray.copy(runoff_regrid.data)
    inflow[t0], storage[t0], outflow[t0] = \
        run_timestep(inflow[t0],storage_old)
    print('Storing '+filename+'_t'+'%03d'%t0+'.nc')
    save_timestep(runoff_regrid[t0],inflow[t0],outflow[t0],storage[t0],filename+'_t'+'%03d'%t0+'.nc')
        
    for t in range(t0+1,nt):
        inflow[t], storage[t], outflow[t] = \
            run_timestep(inflow[t],storage[t-1])   
        print('Storing '+filename+'_t'+'%03d'%t+'.nc')
        save_timestep(runoff_regrid[t],inflow[t],outflow[t],storage[t],filename+'_t'+'%03d'%t+'.nc')
    
    #print('Storing '+filename+'.nc')
    #save_year(runoff_regrid,inflow,outflow,storage,filename+'.nc')
    gc.collect()    
    return storage

if __name__ == "__main__":
    main()
