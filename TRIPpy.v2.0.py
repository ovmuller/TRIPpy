#!/usr/bin/env python3.9
# encoding: utf-8
"""
TRIPpy v2.0
@author: Omar V. Müller, ovmuller@gmail.com

Changes in Version 2.0:
- Optimization for faster calculations of river network parameters.
- Read meander and flow velocity for each grid cell from a nc file.
- Incorporates manual calibration.
"""

import iris
import numpy as np
import geopy.distance as gd
import time
from numba import jit
import gc
import f90nml
from warnings import filterwarnings
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def main():
    '''
    This function runs TRIPpy model.
    '''
    
    ###############
    # TRIPpy MAIN #
    ###############

    # TIMER STARTS
    start_time = time.time()

    # SET VERSION OF TRIPpy
    global version
    version = 'v2.0'
    
    # AVOID WARNING MESSAGES
    filterwarnings('ignore')
 
    # VERBOSE MESSAGE
    print('\n-------------------')
    print('Running TRIPpy ' + version)
    print('-------------------\n')
    
    ###########################
    # SET MODEL CONFIGURATION #
    ###########################
    set_configuration() # The configuration is set by the user in namelist.input

    # VERBOSE MESSAGE    
    true_steps = [step for step, value in workflow.items() if value]
    
    if len(true_steps) == 0:
        print('The workflow does not include any steps.\n')
    elif len(true_steps) == 1:
        print(f'The workflow includes {true_steps[0]}.\n')
    else:
        steps_str = ", ".join(true_steps[:-1]) + ", and " + true_steps[-1]
        print(f'The workflow includes {steps_str}.\n')
    
    #########################
    # PREPROCESSING SECTION #
    #########################
    if preprocess:
        # VERBOSE MESSAGE
        print('Preprocessing parameters for TRIPpy')
        
        # WRITE FIXED PARAMETERS
        write_length_next_params()
        write_speed_meander_params(p_params_path, p_dir_seq_ifile, \
                                   p_speed_meander_ofile, p_speed, p_meander)
        if not args.verbose: print()    
    #######################
    # CALIBRATION SECTION #
    #######################
    if calibrate:
        # VERBOSE MESSAGE
        print('Calibrating parameters for monitored catchments\n')
    
        # LOAD OBSERVED DATA, AND ORGANIZE STATIONS IN TIERS
        # TIERS REPRESENT THE LEVELS OF CALIBRATION IN THE CASCADE PROCESS.
        # THE MOST DOWNSTREAM STATION IS ASSIGNED TIER=0. TIERS INCREASE 
        # SEQUENTIALLY AS WE MOVE UPSTREAM, WITH THE MOST UPSTREAM INDEPENDENT 
        # SUB-BASINS RECEIVING THE HIGHEST TIER NUMBER IN THE BASIN.
        tiers, observations = read_all_observations()

        # CREATE OUTPUT DIRECTORY IF IT DOES NOT EXIST
        if not os.path.exists(c_output_path): os.makedirs(c_output_path)
        
        # INITIALIZE DICTIONARY LINKING STATIONS WITH SPEED AND MEANDER VALUES
        stid_s_m = {}
        
        # INITIALIZE 2D ARRAY WITH THE DOMAIN OF THE ALREADY CALIBRATED BASINS
        calibrated_basins = None

        # INITIAILIZE DICTIONARY CONTAINING THE BEST SIMULATIONS FOR CURRENT
        # TIER AND UPSTREAM TIERS
        cum_best_simulations={}
        
        # ITERATE OVER TIERS FROM MAX TO 0
        for t in sorted(tiers.keys(), reverse=True):
            
            # VERBOSE MESSAGE
            if args.verbose: print(f'Calibrating tier {t}\n')
            
            # IDENTIFY STATIONS AND OBSERVATIONS FOR CURRENT TIER
            tier_stations = tiers[t]
            tier_observations = {stid: observations[stid] for stid in 
                                 tier_stations}
        
            # DIRECTION AND SEQUENCE FILE IS CONSTRAINED TO RUN TRIPpy JUST 
            # UPSTREAM STATIONS
            # THE OUTPUT VARIABLE IS A 2D MATRIX WHERE AREAS OF CURRENT AND
            # PREVIOUS TIERS ARE IDENTIFY WITH THEIR STID
            cum_tier_basin_masks = calibration_domain(tier_observations, 
                                                      calibrated_basins)
  
            # RUN SIMULATIONS VARYING SPEED AND MEANDER
            # CUMULATIVE TIERS INCLUDES THE CURRENT TIER + THE UPSTREAM TIERS
            cum_tiers = {k: tiers[k] for k in range(t, max(tiers.keys()) + 1)}
            tier_calib_simulations, dates = run_calibration(tier_observations,
                                                            stid_s_m, 
                                                            cum_tier_basin_masks,
                                                            cum_tiers)
        
            # DETERMINE THE BEST PAIR (SPEED, MEANDER) FOR EACH MONITORED BASIN
            tier_best_simulations = select_best_params(tier_calib_simulations, 
                                                       tier_observations, 
                                                       dates, t)
            cum_best_simulations.update(tier_best_simulations)
            
            # SET THE CALIBRATED SPEED AND MEANDER FOR EACH STATION
            # BEFORE THE CALIBRATION MOVES TO THE NEXT TIER
            for stid, s_m in tier_best_simulations.items():
                stid_s_m[stid] = {'speed': s_m['speed'], 'meander': 
                                  s_m['meander']}
            
            calibrated_basins = cum_tier_basin_masks    
        
        # STORE THE NETCDF WITH CALIBRATED SPEED AND MEANDER FOR ALL STATIONS
        write_speed_meander_params(c_params_path, c_dir_seq_ifile,
                                   c_speed_meander_ofile, 0.5, 1.4, True,
                                   cum_best_simulations, cum_tier_basin_masks, 
                                   cum_tiers)

    
    ###################
    # RUNNING SECTION #
    ###################
    if run:
        # VERBOSE MESSAGE
        print('Running TRIPpy simulation')

        # RUN SIMULATION
        if not os.path.exists(r_output_path): os.makedirs(r_output_path)
        run_simulation(r_params_path, r_forcing_path, r_forcing_prefix, 
                        r_dir_seq_ifile, r_length_next_ifile, 
                        r_speed_meander_ifile, r_start, r_end, r_spinup_cycles, 
                        r_spinup_years, r_output_path, r_output_prefix)
            
    # VERBOSE MESSAGE WITH TIMER TIME
    print("--- %s minutes ---\n" % ((time.time() - start_time)/60.))

def set_configuration():
    '''
    This function sets the configuration defined by the user by:
    + Loading variables from namelist.input and assigning default values 
      to any variables that are not explicitly defined by the user.

    input:
        None

    output:
        The final configuration is stored in namelist.output.TRIPpy_{version}.
    '''

    # ALL VARIABLES FROM NAMELIST.INPUT ARE DEFINED AS GLOBAL
    # WORKFLOW VARIABLES
    global workflow, preprocess, calibrate, run
    # PREPROCESSING VARIABLES
    global p_params_path, p_dir_seq_ifile, p_length_next_ofile, p_antarctica, \
           p_speed, p_meander, p_speed_meander_ofile
    # CALIBRATION VARIABLES
    global c_obs_path, c_obs_suffix, c_dir_seq_ifile, c_dir_seq_ofile, \
           c_params_path, c_length_next_ifile, c_forcing_path, c_forcing_prefix, \
           c_speed, c_meander, c_speed_meander_ofile, c_output_path, \
           c_output_prefix, c_spinup_years, c_spinup_cycles, c_start, c_end
    # RUNNING VARIABLES
    global r_forcing_path, r_forcing_prefix, r_output_path, r_output_prefix, \
           r_params_path, r_dir_seq_ifile, r_length_next_ifile, \
           r_speed_meander_ifile, r_restart, r_restart_file, r_spinup_years, \
           r_spinup_cycles, r_start, r_end

    # VERBOSE MESSAGE
    if args.verbose: print('Reading namelist.input')
    
    # LOAD VARIABLES FROM NAMELIST.INPUT AND SET DEFAULT VALUES FOR THOSE UNDEFINED BY THE USER
    nml = f90nml.read('namelist.input')

    workflow_section = nml.get('workflow', {})
    preprocess_section = nml.get('preprocess', {})
    calibrate_section = nml.get('calibrate', {})
    run_section = nml.get('run', {})

    preprocess = workflow_section.get('preprocess', False)
    calibrate = workflow_section.get('calibrate', False)
    run = workflow_section.get('run', False)
    workflow = {'preprocess':preprocess, 'calibration':calibrate, 'running':run}

    if preprocess:
        if args.verbose: print('Read preprocess configuration')
        p_params_path = preprocess_section.get('p_params_path', './data/params/')
        p_dir_seq_ifile = preprocess_section.get('p_dir_seq_ifile', 
                                                 'DRT_16th_dir_seq.LPB.nc')
        p_length_next_ofile = preprocess_section.get('p_length_next_ofile', 
                                                     'p_DRT_16th_length_next.LPB.nc')
        p_antarctica = preprocess_section.get('p_antarctica', False)
        p_speed = preprocess_section.get('p_speed', 0.5)
        p_meander = preprocess_section.get('p_meander', 1.4)
        p_speed_meander_ofile = preprocess_section.get('p_speed_meander_ofile', 
                                                       './data/params/DRT_16th_speed_meander.LPB.nc')

    if calibrate:
        if args.verbose: print('Read calibration configuration')
        c_obs_path = calibrate_section.get('c_obs_path', './data/obs/')
        c_obs_suffix = calibrate_section.get('c_obs_suffix', 
                                             '_Q_Month.Cmd.txt')
        c_params_path = calibrate_section.get('c_params_path', 
                                              './data/params/')
        c_dir_seq_ifile = calibrate_section.get('c_dir_seq_ifile', 
                                                'DRT_16th_dir_seq.LPB.nc')
        c_dir_seq_ofile = calibrate_section.get('c_dir_seq_ofile', 
                                                'c_DRT_16th_dir_seq.LPB.nc')
        c_length_next_ifile = calibrate_section.get('c_length_next_ifile', 
                                                    'DRT_16th_length_next.LPB.nc')
        c_forcing_path = calibrate_section.get('c_forcing_path', 
                                               './data/forcings/')
        c_forcing_prefix = calibrate_section.get('c_forcing_prefix', 
                                                 'TOTAL_RUNOFF_LaPlata.monmean.')
        c_speed = calibrate_section.get('c_speed', [0.4, 0.5])
        if not isinstance(c_speed,list): c_speed = [c_speed]
        c_meander = calibrate_section.get('c_meander', [1.4, 1.6])
        if not isinstance(c_meander,list): c_meander = [c_meander]
        c_speed_meander_ofile = calibrate_section.get('c_speed_meander_ofile', 
                                                      'c_DRT_16th_speed_meander.LPB.nc')
        c_output_path = calibrate_section.get('c_output_path',
                                              './data/output/ctrl_values/')
        c_output_prefix = calibrate_section.get('c_output_prefix',
                                                'RIVER_FLOW_LaPlata.calbiration.')
        c_spinup_years =  calibrate_section.get('c_spinup_years',  1)
        c_spinup_cycles = calibrate_section.get('c_spinup_cycles',  1)
        c_start = calibrate_section.get('c_start',  2001)
        c_end = calibrate_section.get('c_end',  2020)

    if run:
        if args.verbose: print('Read run configuration')
        r_forcing_path = run_section.get('r_forcing_path', './data/forcings/')
        r_forcing_prefix = run_section.get('r_forcing_prefix', 
                                           'TOTAL_RUNOFF_LaPlata.monmean.')
        r_params_path = run_section.get('r_params_path', 
                                        './data/params/')
        r_dir_seq_ifile = run_section.get('r_dir_seq_ifile', 
                                          'DRT_16th_dir_seq.LPB.nc')
        r_length_next_ifile = run_section.get('r_length_next_ifile', 
                                              'DRT_16th_dir_seq.LPB.nc')
        r_speed_meander_ifile = run_section.get('r_speed_meander_ifile', 
                                                'DRT_16th_speed_meander.LPB.nc')
        r_output_path = run_section.get('r_output_path', 
                                        './data/output/ctrl_values/')
        r_output_prefix = run_section.get('r_output_prefix', 
                                          'RIVEr_FLOW_LaPlata.')
        r_restart = run_section.get('r_restart', False)
        r_restart_file = run_section.get('r_restart_file', '')
        r_spinup_years = run_section.get('r_spinup_years', 1)
        r_spinup_cycles = run_section.get('r_spinup_cycles', 1)
        r_start = run_section.get('r_start', 2001)
        r_end = run_section.get('r_end', 2020)

        if r_restart == 0:
            r_restart_file = ''
        elif r_spinup_years != 0 or r_spinup_cycles != 0:
            r_spinup_years = 0
            r_spinup_cycles = 0
            if args.verbose:
                print('Spinup is not possible when restarting a simulation: \
                      spinup_years and spinup_cycles are set to 0.')

    # VERBOSE MESSAGE
    if args.verbose:
        print('Writing namelist.output.TRIPpy_' + version +'\n')

    # STORE FINAL CONFIGURATION
    nml.write('namelist.output.TRIPpy_' + version, force=True)        


@jit
def write_length_next_params():
    '''
    This function set and store TRIPpy fixed length and next parameters.
    
    input:
        None
    
    output:
        params_list: list of 3 cubes:
            0) riverslength: cube - the length [m] between the current and 
                                   the downstream point
            1) next_y: cube - the vertical index of next downstream point
            2) next_x: cube - the horizontal index of next downstream point
            The cubes are stored in the p_length_next_ofile as NetCDF file.
    '''
    
    # LOAD DIRECTION AND GET GRID DETAILS
    direction = iris.load(os.path.join(p_params_path,p_dir_seq_ifile),
                          'flow_direction')[0]

    lat_points = direction.coord('latitude').points
    lon_points = direction.coord('longitude').points

    dlat = np.abs(lat_points[1] - lat_points[2])
    dlon = np.abs(lon_points[1] - lon_points[2])
    min_lo = lon_points[0]
    max_lo = lon_points[-1]


    # PRE-COMPUTE OFFSETS AND MAP THEM
    dir_offsets = np.array([
        [dlat, 0],       # N
        [dlat, dlon],    # NE
        [0, dlon],       # E
        [-dlat, dlon],   # SE
        [-dlat, 0],      # S
        [-dlat, -dlon],  # SW
        [0, -dlon],      # W
        [dlat, -dlon],   # NW
        [0, 0],          # RIVER MOUTH or LAKE
    ])

    # CREATE A CUBE WITH SAME DIMENSIONS THAN DIRECTION TO STORE PARAMETERS
    target_cube = direction.copy()
    target_cube.attributes = {'Data_source': 'This data was created by TRIPpy '
                              +version+' based on the flow direction file', 
                              'Developer': 'Omar V. Muller from CONICET and UNL, '
                              'ovmuller@gmail.com', 
                              'Reference': 'Muller et al., 2024: River flow in '
                              'the near future: a global perspective in the '
                              'context of a high-emission climate change '
                              'scenario. HESS, 28(10), 2179-2201.'}

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

    # LOOP OVER LATS AND LONGS TO ESTIMATE NEXT DOWNSTREAM POINT AND DISTANCE 
    # BETWEEN CONNECTED POINTS
    for y in range(len(lat_points)):
        for x in range(len(lon_points)):
            dirlalo = int(direction.data[y, x])
            if dirlalo > 0:
                offset = dir_offsets[dirlalo - 1]
                next_la = lat_points[y] + offset[0]
                next_lo = lon_points[x] + offset[1]

                if next_lo > max_lo:
                    next_lo -= 360.0
                if next_lo < min_lo:
                    next_lo += 360.0

                if next_la < -90.0 and p_antarctica:
                    next_la = lat_points[y]
                    next_lo = next_lo + 180.0 if next_lo < 180.0 else next_lo - 180.0

                riverslength.data[y, x] = gd.geodesic((lat_points[y], lon_points[x]), 
                                                 (next_la, next_lo)).m
                next_y.data[y, x] = np.abs(lat_points - next_la).argmin()
                next_x.data[y, x] = np.abs(lon_points - next_lo).argmin()
                
    # CREATE CUBE LIST TO STORE AS NETCDF
    params_list = iris.cube.CubeList([riverslength, next_y, next_x])
    # VERBOSE MESSAGE
    if args.verbose: print('\nStoring '+os.path.join(p_params_path,p_length_next_ofile)+'\n')
    # STORE PARAMETERS AS NETCDF
    iris.save(params_list,os.path.join(p_params_path,p_length_next_ofile))

    return params_list

def write_speed_meander_params(params_path, direction_ifile, speed_meander_ofile,
                               speed=0.5, meander=1.4, calibration=False, 
                               stid_s_m=None, basin_masks=None, cum_tiers=None):
    '''
    This function sets the parameter file with default and/or calibrated 
    speed and meander values. 
    
    input:
        params_path: str - Path to parameters directory
        direction_ifile: str - Filename of file with direction data.
        speed_meander_ofile: str - Path to save the calibrated parameters file.
        speed: float, optional - Speed value for non-calibrated points.
        meander: float, optional - Meander value for non-calibrated points.
        calibration: bool, optional - False when parameters are default in the 
                     entire domain. True for calibrated parameters. When True, 
                     it requieres best_simulations and basin_masks as inputs.    
        stid_s_m: dict, optional - A dictionary containing station IDs as keys
                  and the value is a dictionary containing the calibration 
                  parameters (speed and meander) for that station’s 
                  contributing grid-cells.
        basin_masks: 2darray, optional representing basin masks with each basin
                     identified by its station ID.
        cum_tiers: dict, optional - A dictionary containing calibration tiers as 
                   keys and the stid of each tier.

    output:
        The parameters speed and meander are stored in the speed_meander_ofile
        NetCDF file.
    '''

    # LOAD DIRECTION CUBE AS REFERENCE FOR DIMENSIONS
    direction = iris.load(os.path.join(params_path,direction_ifile),'flow_direction')[0]
    direction.data = direction.data.astype(float)

    # CREATE CUBES WITH SAME DIMENSIONS THAN DIRECTION TO STORE PARAMETERS
    speed_param = direction.copy()
    speed_param.attributes = {'Data_source': 'This data was created by TRIPpy '
                              +version+' based on the flow direction file', 
                              'Developer': 'Omar V. Muller from CONICET and UNL, '
                              'ovmuller@gmail.com', 
                              'Reference': 'Muller et al., 2024: River flow in '
                              'the near future: a global perspective in the '
                              'context of a high-emission climate change '
                              'scenario. HESS, 28(10), 2179-2201.'}
    speed_param.var_name = 'speed'
    speed_param.units = 'm s-1'

    meander_param = direction.copy()
    meander_param.attributes = {'Data_source': 'This data was created by TRIPpy '
                                +version+' based on the flow direction file', 
                                'Developer': 'Omar V. Muller from CONICET and UNL, '
                                'ovmuller@gmail.com', 
                                'Reference': 'Muller et al., 2024: River flow in '
                                'the near future: a global perspective in the '
                                'context of a high-emission climate change '
                                'scenario. HESS, 28(10), 2179-2201.'}
    meander_param.var_name = 'meander'
    meander_param.units = ''
    
    # SET FIXED VALUES FOR THE ENTIRE DOMAIN
    speed_param.data[:] = speed
    meander_param.data[:] = meander

    # SET CALIBRATED PARAMETERS TO THE BASINS WHEN CALIBRATION WAS APPLIED
    if calibration:
    # SET PARAMETERS PER TIERS TO OVERLAP THE VALUES IN UPSTREAM BASINS
        for t in sorted(cum_tiers.keys()):
            for stid in cum_tiers[t]:
                speed_param.data[basin_masks == stid] = stid_s_m[stid]['speed']
                meander_param.data[basin_masks == stid] = stid_s_m[stid]['meander']

    # CREATE CUBE LIST TO STORE AS NETCDF
    params_list = iris.cube.CubeList([speed_param,meander_param])
    # VERBOSE MESSAGE
    if args.verbose: print('Storing '+os.path.join(params_path,speed_meander_ofile)+'\n')
    # STORE PARAMETERS AS NETCDF
    iris.save(params_list,os.path.join(params_path, speed_meander_ofile), zlib=True, complevel=9)

    return

def read_observation_file(obs_ifile_path):
    '''
    This function reads an observation file and extracts relevant information.
    
    input: 
        obs_ifile_path: str - Path to the observation file
    
    output:
        stid: int - Station ID
        lat_model: float - Latitude of the model where the station is located
        lon_model: float - Longitude of the model where the station is located
        next_stid: int - Next downstream station ID. If there isn't downstream
                         station next_stid=0
        observed_values: masked 1darray - Observed river flow values
    '''
    
    # READ ALL FILE LINES
    with open(obs_ifile_path, 'r') as file:
        lines = file.readlines()

    # EXTRACT STID, LAT AND LONG (OF THE MODEL), AND NEXT DOWNSTREAM STATION ID
    stid_index = next(i for i, line in enumerate(lines) if 
                      line.startswith('# GRDC-No.:'))
    lat_index = next(i for i, line in enumerate(lines) if 
                     line.startswith('# Latitude (MODEL):'))
    lon_index = next(i for i, line in enumerate(lines) if 
                     line.startswith('# Longitude (MODEL):'))
    next_stid_index = next(i for i, line in enumerate(lines) if 
                           line.startswith('# Next downstream station:'))

    stid = int(lines[stid_index].split(':')[1].strip())
    lat_model = float(lines[lat_index].split(':')[1].strip())
    lon_model = float(lines[lon_index].split(':')[1].strip())
    next_stid_str = lines[next_stid_index].split(':')[1].strip()
    next_stid = int(next_stid_str) if next_stid_str != "-" else 0
    
    # EXTRACT OBSERVED VALUES
    data = []
    for line in lines:
        if not line.startswith('#') and not line.startswith('YYYY-MM'):
            date_str, value_str = line.strip().split(';')
            data.append(float(value_str))
    
    observed_values = np.ma.masked_equal(np.array(data), -999.000)

    return stid, lat_model, lon_model, next_stid, observed_values

def read_all_observations():
    '''
    This function reads all observation files from a given path with a 
    specified suffix and organizes them by hierarchy levels.
    
    output:
        tiers: dict - A dictionary where keys are tiers (0, 1, 2, ...)
                      and values are lists of station IDs at that level
        observations: dict - Dictionary containing station IDs as keys and 
                      dictionaries containing latitude, longitude, observed 
                      data, and next_stid as values
    '''
    
    # INITIALIZE DICTIONARIES
    observations = {}
    downstream_map = {}
    tiers = {}
    
    # READ OBSERVATION FILES 
    for obs_file in os.listdir(c_obs_path):
        if obs_file.endswith(c_obs_suffix):
            stid, lat, lon, next_stid, observed = read_observation_file(
                                            os.path.join(c_obs_path, obs_file))
            observations[stid] = {'latitude': lat, 'longitude': lon, 
                                  'observed_data': observed, 
                                  'next_stid': next_stid}
            # DOWNSTREAM MAP HAS NEXT_STID AS KEY, AND UPSTREAM STIDS
            # DIRECTLY CONTRIBUTING TO IT AS VALUES
            downstream_map.setdefault(next_stid, []).append(stid)
            
    # LOOP UNTIL ALL STATIONS GET A HIERARCHY tier IN THE CATCHMENT
    # THE SMALLER TIER THE CLOSER TO THE RIVER MOUTH
    unassigned_stations = set(observations.keys())
    current_tier = 0
    while unassigned_stations:
        tiers[current_tier] = []
        for stid in list(unassigned_stations):
            next_stid = observations[stid]['next_stid']
            if next_stid == 0 or next_stid in tiers.get(current_tier - 1, []):
                tiers[current_tier].append(stid)
                unassigned_stations.remove(stid)
        
        # BREAK THE LOOP IF NO STATIONS ARE ASSIGNED IN THE CURRENT TIER
        if not tiers[current_tier]:
            raise ValueError(f'Loop detected or broken dependency for stations: '
                             f'{unassigned_stations}')
        
        current_tier += 1
    
    return tiers, observations

def basin_mask(y,x,label,direction,rmask=np.zeros(0)):
    '''
    This function labels all grid-cells contributing to the flow at location 
    (y, x) using a region growing algorithm. It identifies connected grid-cells 
    upstream, where a neighboring cell is considered connected if its flow 
    direction points to the pivot cell. All grid-cells in the basin are
    labelled with the same label.
    
    input: 
        y: int - vertical location of the station in the direction matrix
        x: int - horizontal location of the station in the direction matrix
        label: int - value that will be assigned to the mask upstream (y,x)
        direction: 2d matrix - the flow direction
        rmask: 2d matrix, optional - a previous mask with labelled grid-cells 
               can be passed as argument. In such case consider using a 
               different label for the new mask.

    output:
        rmask: 2d matrix - output matrix with label values over the mask
               and zero in the rest.
    '''
    
    # CREATE RMASK IF EMPTY
    if rmask.size == 0:
        rmask = np.zeros(direction.shape)
    ymax = rmask.shape[0]
    xmax = rmask.shape[1]
    
    # REGION GROWING ALGORITHM
    # list is a queue where the first position is labelled and deleted from 
    # the queue and all the neighbors are checked. Those that are tributaries 
    # are appended to the queue. The loop finish when the queue is empty.
    list = []
    list.append((y, x))
    while(len(list) > 0):
#        print(list)
        pivote = list[0]
        rmask[list[0]] = label
        list.pop(0)

        #check N neighbor
        if pivote[0]+1<ymax: N = (pivote[0]+1,pivote[1])
        else: N = (pivote[0]+1-ymax,pivote[1])
        if (direction[N] == 5.0) and (N not in list):
            list.append((N))
            
        #check NE neighbor
        if pivote[0]+1 < ymax and pivote[1]+1 < xmax: 
            NE = (pivote[0]+1,pivote[1]+1)
        elif pivote[0]+1 < ymax: NE = (pivote[0]+1,pivote[1]+1-xmax)
        elif pivote[1]+1 < xmax: NE = (pivote[0]+1-ymax,pivote[1]+1)
        if (direction[NE] == 6.0) and (NE not in list):
            list.append((NE))
            
        #check E neighbor
        if pivote[1]+1<xmax: E = (pivote[0],pivote[1]+1)
        else: E = (pivote[0],pivote[1]+1-xmax)
        if (direction[E] == 7.0) and (E not in list):
            list.append((E))
        
        #check SE neighbor
        if pivote[1]+1 < xmax: SE = (pivote[0]-1,pivote[1]+1)
        else: SE = (pivote[0]-1,pivote[1]+1-xmax)
        if (direction[SE] == 8.0) and (SE not in list):
            list.append((SE))
        
        #check S neighbor
        S = (pivote[0]-1,pivote[1])  
        if (direction[S]==1.0) and (S not in list):
            list.append((S))
        
        #check SW neighbor
        SW = (pivote[0]-1,pivote[1]-1)    
        if (direction[SW] == 2.0) and (SW not in list):
            list.append((SW))  

        #check W neighbor
        W = (pivote[0],pivote[1]-1)    
        if (direction[W] == 3.0) and (W not in list):
            list.append((W))
        
        #check NW neighbor
        if pivote[0]+1 < ymax: NW = (pivote[0]+1,pivote[1]-1)
        else: NW = (pivote[0]+1-ymax,pivote[1]-1)
        if (direction[NW] == 4.0) and (NW not in list):
            list.append((NW))           
                                 
    return rmask

def ll2yx(lati,lonj,lat,lon):
    '''
    This function returns the grid indices of a given latitude and longitude.

    input:
        lati: float - latitude to be converted
        lonj: float - longitude to be converted
        lat: 1darray - Array of latitude values in the domain
        lon: 1darray - Array of longitude values in the domain
    
    output:
        idy: index of the point in the vertical axis (latitudes)        
        idx: index of the point in the horizontal axis (longitudes)
    '''
    
    # IDENTIFY THE (LAT, LON) LOCATION IN THE GRID
    idy = int(np.where(lat==lati)[0])
    idx = int(np.where(lon==lonj)[0])
    
    return idy, idx

def calibration_domain(observations,calibrated_basins=None):
    '''
    This function modifies the direction and sequence parameter file 
    to run TRIPpy only upstream stations. It also returns a mask with the
    calibration domain.

    input:
        observations: dict - A dictionary containing station IDs as keys and 
                      nested dictionaries containing latitude and longitude as 
                      values.
        calibrated_basins: 2d array, optional - Is a matrix similar to the 
                           output but having the stid values of basins 
                           calibrated in previous tiers.
        
    output:
        basin_masks: 2d array - Is a matrix, where each observed basin is 
                     identify with its corresponding stid value. 
        The direction and sequence constrained to the calibration domain is
        stored in the c_dir_seq_ofile as NetCDF file.
    '''
    
    # LOAD DIRECTION AND SEQUENCE DATA
    direction = iris.load(os.path.join(c_params_path, c_dir_seq_ifile), 
                          'flow_direction')[0]
    sequence = iris.load(os.path.join(c_params_path, c_dir_seq_ifile), 
                         'flow_sequence')[0]

    # GET THE DOMAIN LATITUDES AND LONGITUDES
    latitudes = direction.coord('latitude').points
    longitudes = direction.coord('longitude').points

    # INITIALIZE BASIN MASKS
    basin_masks = np.zeros(direction.data.shape)

    # THE LOOP LABELS MONITORED BASINS WITH THE STID VALUES
    for stid, station_data in observations.items():
        lat = station_data['latitude']
        lon = station_data['longitude']
        y, x = ll2yx(lat, lon, latitudes, longitudes)
        basin_masks = basin_mask(y, x, stid, direction.data, basin_masks)

    # COPY ORIGINAL DIRECTION AND SEQUENCE AND MASKOUT GRID-CELLS OF 
    # NON-MONITORED BASINS
    direction_basins = direction.copy()
    sequence_basins = sequence.copy()
    direction_basins.data = direction_basins.data * (basin_masks > 0)
    sequence_basins.data = sequence_basins.data * (basin_masks > 0)

    # CREATE CUBE LIST TO STORE AS NETCDF
    output_dir_seq = iris.cube.CubeList([direction_basins, sequence_basins])
    # VERBOSE MESSAGE
    if args.verbose: print('Storing '+
                           os.path.join(c_params_path + c_dir_seq_ofile)+'\n')
    # STORE PARAMETERS CONSTRAINED TO THE CALIBRATION DOMAIN AS NETCDF
    iris.save(output_dir_seq, os.path.join(c_params_path+c_dir_seq_ofile))
    
    # OVERLAP MASK VALUES IF THERE ARE SUB-BASINS ALREADY CALIBRATED
    if calibrated_basins is not None:
        basin_masks[calibrated_basins>0] = calibrated_basins[calibrated_basins>0]
        
    return basin_masks


def set_coef(ny,nx,dt,length_2d,speed_2d,meander_2d):
    '''
    This function set TRIPpy coefficients Ct and k.
    
    input: 
        ny: int - Lenght of latitudes array
        nx: int - Lenght of longitudes array
        dt: int - Time delta in seconds
        length_2d: 2d array - Length [m] between current and downstream point
        speed_2d: 2d array - Flow speed [m/s]
        meander_2d: 2d array - Meandering ratio
        
    output:
        Ct: 2d array - Coefficient Ct = exp(-c.dt)
        k: 2d array - Coefficient k = (1-Ct)/c
    '''
    
    # CREATE OUTPUT ARRAYS
    Ct = np.zeros((ny,nx))
    k = np.zeros((ny,nx))
    
    # CALCULATE TRIPpy COEFFICIENTS
    for y in range(ny):
        for x in range(nx):
            if length_2d[y,x] == 0.:
                c = 0.0
                Ct[y,x] = 1.0
                k[y,x] = 0.
            else:
                c = speed_2d[y,x]/(length_2d[y,x]*meander_2d[y,x])
                Ct[y,x] = np.exp(-(c*dt))
                k[y,x] = (1-Ct[y,x])/c
            
    return Ct, k

@jit
def run_timestep(inflow, storage_old, dire, sequ, Ct, k, dt, ne_y, ne_x, ny, nx, ns):
    '''
    This function run one TRIPpy timestep.
    
    input: 
        inflow: 2d array - Local runoff + inflow from linked neighbours
        storage_old: 2d array - Land-only storage of previous timestep
        dire: 2d array - Direction data
        sequ: 2d array - Sequence data
        Ct: 2d array - Ct = e^(-c) x dt
        k: 2d array - k = [1 - e^(-c) x dt ] / c
        dt: float - Delta time [s]
        ne_y: int - Next downstream point in latitude
        ne_x: int - Next downstream point in longitude
        ny: int - Number of latitudes
        nx: int - Number of longitudes
        ns: int - Maximum sequence value
        
    output:
        storage: 2d array - Storage of water [kg]
        outflow: 2d array - Outflow [kg/s]
    '''
    
    # INITIALIZE OUTPUT VARIABLES
    storage = np.ma.zeros((ny,nx))
    outflow = np.ma.zeros((ny,nx))

    # LOOP OVER SEQUENCE
    for s in range(1,ns):

        # FOR ALL LAND-POINTS
        for y in range(ny):
            for x in range(nx):

                # RIVER IS ACTIVE ON THIS SEQUENCE?
                if ( int(sequ[y,x]) == s):
                    # CALCULATE STORATE AT THE END OF THE TIMESTEP (S(t+dt), 
                    # eq. 4 Oki 1999)
                    storage[y,x] = Ct[y,x] * storage_old[y,x] + \
                                   k[y,x] * inflow[y,x]
                    # CALUCATE OUTFLOW AS INFLOW PLUS CHANGE IN STORAGE (Qo(t+dt), 
                    # eq. 2 Oki 1999)
                    outflow[y,x] = inflow[y,x] + \
                                   (storage_old[y,x]-storage[y,x])/dt
                    
                    # IF THERE IS A DOWNSTREAM POINT
                    # ASSIGN CURRENT OUTFLOW TO ITS INFLOW
                    y2 = int(ne_y[y,x])
                    x2 = int(ne_x[y,x])
                    if dire[y,x]<9.:
                        inflow[y2,x2] = inflow[y2,x2] + outflow[y,x]
    #print(y2,x2,dire[y,x],inflow[300,200])
    return storage, outflow

def save_timestep(runoff_regrid,inflow,outflow,storage,mask,filename):
    '''
    This function store a NetCDF with a timestep of a TRIPpy simulation.
    
    input: 
        runoff_regrid: 3d cube - runoff [kg/s] with the TRIPpy grid and 1 time
        inflow: 2d array - Inflow [kg/s] array
        outflow: 2d array - Outflow [kg/s] array
        storage: 2d array - Storage [kg] array
        mask: 2d array - Ocean points are masked out
        filename: str - Filename to store the cubes as NetCDF
        
    output:
        The storage [kg] and outflow [kg/s] in a given timestep are stored as 
        NetCDF file. The file is 3d because include time information.
    '''    
    
    # CREATE CUBE TO STORE OUTPUT VARIABLES
    target_cube = iris.util.new_axis(runoff_regrid, 'time')
    target_cube.attributes = {'Data_source': 'This data was created by TRIPpy '
                              +version+' based on the flow direction file', 
                              'Developer': 'Omar V. Muller from CONICET and UNL, '
                              'ovmuller@gmail.com', 
                              'Reference': 'Muller et al., 2024: River flow in '
                              'the near future: a global perspective in the '
                              'context of a high-emission climate change '
                              'scenario. HESS, 28(10), 2179-2201.'}
    
    outflow_cube = target_cube.copy()
    storage_cube = target_cube.copy()
    outflow_cube.var_name = 'outflow'
    storage_cube.var_name = 'storage'
    storage_cube.units = ('kg')

    # FILL CUBES WITH SIMULATED VALUES
    outflow_cube.data[0] = np.ma.core.MaskedArray.copy(outflow)
    storage_cube.data[0] = np.ma.core.MaskedArray.copy(storage)

    # MASK OUT OCEAN POINTS
    outflow_cube.data[:].mask = mask.mask
    storage_cube.data[:].mask = mask.mask
    
    # CREATE THE LIST OF CUBES TO SAVE AS NETCDF
    #output_list=iris.cube.CubeList([runoff_cube,inflow_cube,outflow_cube,storage_cube])
    output_list = iris.cube.CubeList([outflow_cube,storage_cube])

    # SAVE THE NETCDF
    iris.save(output_list,filename,zlib=True,complevel=9)

    return

def run_year(storage_old, runoff_regrid, dire, sequ, mask, Ct, k, dt, 
             ne_y, ne_x, nt, ny, nx, ns, filename):
    '''
    This function runs a year of TRIPpy simulation and stores a NetCDF per 
    timestep.
    
    input: 
        storage_old: 2d array - Initial storage of the year [kg]
        runoff_regrid: 3d cube - runoff for all time-steps in the year [kg/s]
        dire: 2d array - Direction data        
        sequ: 2d array - Sequence data
        mask: 2d array - Ocean points are masked out
        Ct: 2d array - Ct = e^(-c) x dt
        k: 2d array - k = [1 - e^(-c) x dt ] / c
        dt: float - Delta time [s]
        ne_y: int - Next downstream point in latitude
        ne_x: int - Next downstream point in longitude
        nt: int - Number of time-steps
        ny: int - Number of latitudes
        nx: int - Number of longitudes
        ns: int - Maximum sequence number
        filename: str - Filename prefix to store the output cubes
        
    output:
        storage: 3d array - Storage for all time-steps in the year
        The storage [kg] and outflow [kg/s] for each timestep of the year
        are stored as NetCDF file. 
    '''

    # INITIALIZE VARIABLES
    outflow = np.ma.zeros((nt,ny,nx))
    storage = np.ma.zeros((nt,ny,nx))    
    inflow = np.ma.core.MaskedArray.copy(runoff_regrid.data)
    
    # RUN INITIAL TIME-STEP
    storage[0], outflow[0] = run_timestep(inflow[0], storage_old, dire, sequ, \
                                          Ct, k, dt, ne_y, ne_x, ny, nx, ns)
    # VERBOSE MESSAGE
    if args.verbose: print('Storing '+filename+'_t000.nc')
    # STORE TIME-STEP
    save_timestep(runoff_regrid[0], inflow[0], outflow[0], storage[0], mask, \
                  filename+'_t000.nc')
    
    # RUN EACH TIME-STEP OF THE YEAR
    for t in range(1,nt):
        storage[t], outflow[t] = run_timestep(inflow[t], storage[t-1], \
                                              dire,sequ, Ct, k, dt, \
                                              ne_y, ne_x, ny, nx, ns)   
        # VERBOSE MESSAGE
        if args.verbose: print('Storing '+filename+'_t'+'%03d'%t+'.nc')
        # STORE TIME-STEP
        save_timestep(runoff_regrid[t], inflow[t], outflow[t], storage[t], \
                      mask, filename+'_t'+'%03d'%t+'.nc')
    
    #print('Storing '+filename+'.nc')
    #save_year(runoff_regrid,inflow,outflow,storage,filename+'.nc')
    gc.collect()  
    
    return storage

def calculate_timestep(t1, t2, unit):
    """
    This function calculates the time step in seconds based on the given
    time points and unit.

    input:
        t1: float - The first time point.
        t2: float - The second time point.
        unit: str - The unit of time ('days', 'hours', 'minutes', 'seconds').

    output:
        dt: float - The time step in seconds.
    """
    # CONVERSION FACTORS
    conversion_factors = {
        'days': 86400,
        'hours': 3600,
        'minutes': 60,
        'seconds': 1
    }

    # GET THE CONVERSION FACTOR BASED ON THE UNIT
    factor = conversion_factors.get(unit.lower(), None)

    # CHECK THE FACTOR IS VALID
    if factor is None:
        raise ValueError(f"Unsupported time unit: {unit}")

    # CALCULATE THE TIMESTEP IN SECONDS
    dt = (t2 - t1) * factor
    
    return dt

def run_simulation(params_path, forcing_path, forcing_prefix, 
                   dir_seq_file, length_next_file, speed_meander_file,
                   start_year, end_year, spinup_cycles, spinup_years,
                   output_path, output_prefix):
    '''
    This function performs the TRIPpy simulation by: 
    + Loading and processing direction, sequence, and runoff data
    + Regridding and converting units of forcing files
    + running spinup and simulation loops for specified years
    + Storing results for each timestep in netCDF format
    
    input:
        params_path: str - Path to the directory containing parameter files.
        forcing_path: str - Path to the directory containing forcing files.
        forcing_prefix: str - Prefix for forcing file names.
        dir_seq_file: str - File containing flow direction and sequence.
        length_next_file: str - File containing length to next downstream 
                          point data.
        speed_meander_file: str - File containing speed and meander data.
        start_year: int - The starting year for the simulation.
        end_year: int - The ending year for the simulation.
        spinup_cycles: int - Number of spinup cycles to run.
        spinup_years: int - Number of spinup years for each cycle.
        output_path: str - Path to output directory.
        output_prefix: str - Prefix for output file names.

    output:
        Simulation files in NetCDF format.
    '''
    
    # LOAD PARAMETERS
    direction = iris.load(params_path + dir_seq_file, 'flow_direction')[0]
    sequence = iris.load(params_path + dir_seq_file, 'flow_sequence')[0]
    length = iris.load(params_path + length_next_file, 
                       'TRIPpy_length_to_next_downstream_point')[0]
    next_y = iris.load(params_path + length_next_file, 
                       'TRIPpy_index_of_next_downstream_point_in_latitude')[0]
    next_x = iris.load(params_path + length_next_file, 
                       'TRIPpy_index_of_next_downstream_point_in_longitude')[0]

    # LOAD SPEED AND MEANDER FROM FILES
    speed = iris.load(params_path + speed_meander_file, 'speed')[0]
    meander = iris.load(params_path + speed_meander_file, 'meander')[0]

    # TAKE DATA FOR FASTER PROCESSING
    sequ = sequence.data
    dire = direction.data
    ne_y = next_y.data
    ne_x = next_x.data
    mask = np.ma.masked_equal(sequ, 0.)

    # LOAD RUNOFF OF THE FIRST YEAR
    runoff_flux = iris.load_cube(forcing_path + forcing_prefix +
                                 str(start_year) + '.nc', 'runoff_flux')

    # DEFINE TRIPPY TIME-STEP
    t1 = runoff_flux.coord('time').points[1]
    t2 = runoff_flux.coord('time').points[2]
    unit = str(runoff_flux.coord('time').units).split()[0]
    dt = calculate_timestep(t1, t2, unit)
    
    # VERBOSE MESSAGE
    print(f'Timestep: {int(dt/60)} min.\n')

    # GET DIMENSIONS TO ITERATE
    nt = len(runoff_flux.coord('time').points)
    ny = len(direction.coord('latitude').points)
    nx = len(direction.coord('longitude').points)
    ns = int(np.max(sequence.data)) + 1

    # SET TRIPPY COEFFICIENTS
    speed_2d = speed.data
    meander_2d = meander.data
    length_2d = length.data
    Ct, k = set_coef(ny, nx, dt, length_2d, speed_2d, meander_2d)

    # INITIALIZE STORAGE
    storage = np.zeros((nt, ny, nx))

    # RUN SPINUP
    for spc in range(spinup_cycles):
        for spy in range(spinup_years):

            # VERBOSE MESSAGES
            if args.verbose:
                print(f'Running spinup  '
                      f'{spc*spinup_years+spy+1}/{spinup_years*spinup_cycles}')
            if args.verbose:
                print(f'Loading {forcing_prefix}{start_year + spy}.nc')
            # LOAD RUNOFF DATA
            runoff_flux = iris.load_cube(forcing_path + forcing_prefix + 
                                         str(start_year + spy) + '.nc', 
                                         'runoff_flux')
            # SET 0 OVER OCEAN BEFORE INTERPOLATE 
            runoff_flux.data[runoff_flux.data.mask] = 0.     

            # REGRID RUNNOF TO TRIPPY GRID
            runoff_flux_regrid = runoff_flux.regrid(direction, 
                                                    iris.analysis.Nearest())
            if not runoff_flux_regrid.coord('latitude').has_bounds():
                runoff_flux_regrid.coord('latitude').guess_bounds()
                runoff_flux_regrid.coord('longitude').guess_bounds()
            nt = len(runoff_flux.coord('time').points)

            # CONVERT RUNOFF UNITS [kg m-2 s-1] --> [kg s-1]
            area = iris.analysis.cartography.area_weights(runoff_flux_regrid)
            runoff_regrid = runoff_flux_regrid * area
            runoff_regrid.units = ('kg s-1')

            # PREPARE STORAGE FOR YEARS DIFFERENT TO INITIAL YEAR
            storage_old = np.copy(storage[-1])
            
            # RUN A YEAR
            filename = output_path + output_prefix + 'spincycle' + \
                       str(spc) + '_spinyear' + str(spy)
            storage = run_year(storage_old, runoff_regrid, dire, sequ, mask,\
                               Ct, k, dt, ne_y, ne_x, nt, ny, nx, ns, filename)

    # RUN SIMULATION FOR THE ENTIRE PERIOD
    for yr in range(start_year, end_year + 1):

        # LOAD RUNOFF FOR THE CORRESPONDING YEAR
        runoff_flux = iris.load_cube(forcing_path + forcing_prefix + str(yr) + \
                                     '.nc', 'runoff_flux')
        # SET 0 OVER OCEAN BEFORE INTERPOLATE
        runoff_flux.data[runoff_flux.data.mask] = 0.     

        # REGRID RUNNOF TO TRIPPY GRID
        runoff_flux_regrid = runoff_flux.regrid(direction, iris.analysis.Nearest())
        if not runoff_flux_regrid.coord('latitude').has_bounds():
            runoff_flux_regrid.coord('latitude').guess_bounds()
            runoff_flux_regrid.coord('longitude').guess_bounds()
        nt = runoff_flux_regrid.data.shape[0]

        # CONVERT RUNOFF UNITS [kg m-2 s-1] --> [kg s-1]
        area = iris.analysis.cartography.area_weights(runoff_flux_regrid)
        runoff_regrid = runoff_flux_regrid * area
        runoff_regrid.units = ('kg s-1')

        # PREPARE STORAGE FOR YEARS DIFFERENT TO INITIAL YEAR
        storage_old = np.copy(storage[-1])

        # VERBOSE MESSAGE
        if args.verbose:
            print(f'Running year {yr}')
        # RUN A YEAR
        filename = output_path + output_prefix + str(yr)
        storage = run_year(storage_old, runoff_regrid, dire, sequ, mask, \
                           Ct, k, dt, ne_y, ne_x, nt, ny, nx, ns, filename)

    return

def read_simulation(directory, prefix):
    '''
    This function reads simulation data from NetCDF files.
    
    input: 
        directory: str - directory path where simulation files are located
        prefix: str - file prefix to identify simulation files
    
    output:
        simulation_cube: 3d cube - Concatenated cube containing simulation data
    '''

    # LOAD SIMULATION FILES
    file_pattern = os.path.join(directory, f'{prefix}????_t???.nc')
    cube_list = iris.load(file_pattern,'outflow')
    
    # CONCATENATE THE FILES
    simulation_cube = iris.cube.CubeList(cube_list).concatenate_cube()
    
    return simulation_cube

def get_simulation_ts(simulation_cube, lat, lon):
    '''
    This function extracts a time series from the simulation cube at a 
    specified latitude and longitude.
    
    input: 
        simulation_cube: 3d cube - Simulation data
        lat: float - Latitude of the gauge station in the model grid
        lon: float - Longitude of the gauge station in the model grid
    
    output:
        simulated: 1d array - Simulated river flow values for (lat, lon) [m3/s]
        dates: 1d datetime array - Dates corresponding to the simulated values
    '''
    
    # EXTRACT SIMULATED VALUES AT (LAT,LON)
    time_series = simulation_cube.intersection( longitude=(lon,lon), 
                                                latitude=(lat,lat))[:,0,0]
    simulated = time_series.data/1000.
    
    # EXTRACT CORRESPONDING DATES
    time_coord = time_series.coord('time')
    units = time_coord.units
    time_points = time_coord.points

    dates_with_time = np.array([units.num2date(point) for 
                                point in time_points])
    date_strings = np.array([date.strftime('%Y-%m-%d') for 
                             date in dates_with_time])
    dates = np.array([np.datetime64(date) for date in date_strings])
     
    return simulated, dates

def plot_comparison(stid, observed, simulations, dates):
    '''
    This function graphically compares observed and simulated river flows and
    saves the plots as png.
    
    input: 
        stid: int - station ID
        observed: 1d array - Observed river flow values
        simulations: dict - Simulation data including speed, meander, and 
                            simulated values
        dates: 1d datetime array - Dates of the observed and simulated values

    output:
        Figures comparing observations with simulations in the output directory
    '''

    # INITIALIZE ARRAY TO STORE METRICS
    max_simulations = len(simulations)
    matrix = np.zeros((max_simulations, 3))
    
    for idx, s in enumerate(simulations):        
        # FILL THE ARRAY WITH SPEED, MEANDR, AND CORRELATION
        matrix[idx, 0] = s['speed']
        matrix[idx, 1] = s['meander']
        matrix[idx, 2] = s['correlation']
            
    # CREATE FIGURE WITH CUSTOM GRID
    fig= plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 12, wspace=3)
    
    # PLOT TIME-SERIES ON THE LEFT (ax1)
    ax1 = plt.subplot(gs[:8])
    for s in simulations:
        ax1.plot(dates, np.ma.array(s['simulated'], mask = observed.mask), 
                 label = f"$v$={s['speed']}, $m$={s['meander']}: "
                         f"$r$={s['correlation']:.3f}")
    ax1.plot(dates, observed, label='Observations', color='k')
    ax1.set_xlim([dates[0], dates[-1]])
    ax1.set_ylim(bottom=0)
    ax1.set_title(f"Observations vs simulations for stid: {stid}")
    ax1.set_xlabel('Time')
    ax1.set_ylabel('River flow [$m^3 s^{-1}$]')
    #ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), 
               ncol=5, fancybox=True, shadow=True)

    # PLOT SCATTERPLOT SPEED VS MEANDER WHERE THE COLOR INDICATES CORRELATION 
    # ON THE RIGHT (ax2)
    ax2 = plt.subplot(gs[8:])
    sc = ax2.scatter(matrix[:, 0], matrix[:, 1], c=matrix[:, 2], \
                     cmap='brg', s=50)
    
    # HIGHLIGHT DOT WITH HIGHEST CORRELATION
    max_corr_idx = np.argmax(matrix[:, 2])
    ax2.scatter(matrix[max_corr_idx, 0], matrix[max_corr_idx, 1], 
                edgecolors='k', facecolor='None', cmap='brg', s=50)
    
    ax2.set_title('Correlation ($r$)')
    ax2.set_xlabel('Speed ($v$)')
    ax2.set_ylabel('Meander ($m$)')
    
    # ADD COLORBAR
    plt.colorbar(sc)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.9)

    # SAVE FIGURE AND CLOSE IT
    plt.savefig(c_output_path + str(stid) + '.png', \
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    return

def run_calibration(observations,stid_s_m,basin_masks,cum_tiers):
    '''
    This function runs and evaluates the model for each combination of 
    speed and meander.
    
    input:
        observations: dict - A dictionary containing station IDs as keys and
                      dictionaries containing latitude, longitude, and 
                      observed data as values.
        stid_s_m: dict - A dictionary containing station IDs as keys
                  and the value is a dictionary containing the calibration 
                  parameters (speed and meander) for that station’s 
                  contributing grid-cells.
        cum_tiers: dict, optional - A dictionary containing calibration tiers as 
                   keys and the stid of each tier. It includes the tier to be 
                   calibrated in this iteration, alogn with the tiers already 
                   calibrated in previous iterations.
        
    output:
        calib_simulations: dict - A dictionary where each key 
                                  corresponds to a station ID ('stid'), 
                                  and each value is a list of 
                                  dictionaries containing calibration 
                                  results for different combinations of 
                                  speed and meander values.
        dates: 1d array - Dates of the simulation
    '''
    
    # INITIALIZE THE OUTPUT DICTIONARY
    calib_simulations = {}

    # RUN THE MODEL FOR EACH COMBINATION OF PARAMETERS
    for s in c_speed:
        for m in c_meander:
            
            # VERBOSE MESSAGE
            print('Running TRIPpy ' + version + ' with speed='+str(s) + \
                  ' and meander='+str(m))

            # ITERATE OVER EACH STID IN THE OBSERVATIONS DICTIONARY
            for stid in observations.keys():
                # ADD A DICTIONARY ENTRY FOR EACH STID WITH PREDEFINED
                # SPEED AND MEANDER VALUES
                stid_s_m[stid] = {'speed': s, 'meander': m}
            
            # WRITE THE PARAMETERS TO BE TESTED
            write_speed_meander_params(c_params_path, c_dir_seq_ifile,
                                       c_speed_meander_ofile, 0.5, 1.4, True,
                                       stid_s_m, basin_masks, cum_tiers)

            # RUN THE SIMULATION
            run_simulation(c_params_path, c_forcing_path, c_forcing_prefix, \
                           c_dir_seq_ofile, c_length_next_ifile, \
                           c_speed_meander_ofile, c_start, c_end, \
                           c_spinup_cycles, c_spinup_years, \
                           c_output_path, c_output_prefix)
            
            # READ THE SIMULATION
            simulation_cube = read_simulation(c_output_path, c_output_prefix)
            
            # VERBOSE MESSAGE
            if not args.verbose: print()
            print('Estimating skill scores\n')
            # COMPARE SIMULATIONS VS OBSERVATIONS
            for stid, station_data in observations.items():
                lat = station_data['latitude']  
                lon = station_data['longitude']
                obs = station_data['observed_data']
                sim, dates = get_simulation_ts(simulation_cube, lat, lon)

                correlation_coefficient = np.ma.corrcoef(obs, sim)[0,1]
                
                # STORE RESULTS IN THE DICTIONARY
                if stid not in calib_simulations:
                    calib_simulations[stid] = []
                
                result = {'speed': s, 'meander': m,
                          'correlation': correlation_coefficient,
                          'simulated': sim}
                
                calib_simulations[stid].append(result)
    
    return calib_simulations, dates

def select_best_params(calib_simulations, observations, dates, tier):
    '''
    This function selects the best parameterization for each station from the 
    calibration results, writes the results to a file, and 
    generates comparison plots.
    
    input:
        calib_simulations: dict - A dictionary where each key corresponds 
                                  to a station ID ('stid'), and each value 
                                  is a list of dictionaries containing 
                                  calibration results for different 
                                  combinations of speed and meander values.
        observations: dict - A dictionary containing station IDs as keys 
                             and dictionaries containing latitude, 
                             longitude, observed data, and dates as values.
        dates: 1d array - Dates of the simulation
        tier: int - Number of calibrated tier
        
    output:
        best_params_per_station: dict - A dictionary containing the best 
                                        parameterization results for 
                                        each station. Each key corresponds 
                                        to a station ID ('stid'), and 
                                        each value is a dictionary
                                        containing latitude, longitude, 
                                        best speed, best meander, 
                                        correlation, simulated data, 
                                        and observed data.
    '''
    
    # INITIALIZE DICTIONARY TO STORE THE BEST PARAMETER PER STID
    best_params = {}  

    # WRITE THE SCORTES IN A TEXT FILE
    with open(c_output_path + f'calibration_tier_{tier}.txt', 'w') as f:
        if args.verbose: print('Scores:\n')
        for stid, data in calib_simulations.items():
            if args.verbose:
                print(f"Station: {stid}")
                print('speed     meander   correlation')
        
            # WRITE TO FILE
            f.write(f"Station: {stid}\n")
            f.write('speed     meander   correlation\n')
        
            for simulation in data:
                # PRINT AND WRITE SIMULATION RESULTS WITH PROPER FORMAT
                if args.verbose:
                    print(f"{simulation['speed']:<10}{simulation['meander']:<10}"
                          f"{simulation['correlation']:.3f}")
                f.write(f"{simulation['speed']:<10}{simulation['meander']:<10}"
                        f"{simulation['correlation']:.3f}\n")

            # IDENTIFY SIMULATION WITH BEST CORRELATION FOR THE CURRENT STID
            best_simulation = max(data, key=lambda x: x['correlation'])
            observation = observations[stid]
                
            best_params[stid] = {'lat': observation['latitude'],
                                 'lon': observation['longitude'],
                                 'speed': best_simulation['speed'],
                                 'meander': best_simulation['meander'],
                                 'correlation': best_simulation['correlation'],
                                 'simulated': best_simulation['simulated'],
                                 'observed': observation['observed_data']}
            
            print(f"Best parameterisation for stid {stid}: "
                  f"speed={best_params[stid]['speed']}, "
                  f"meander={best_params[stid]['meander']}\n")
            f.write(f"Best parameterisation for stid {stid}: "
                    f"speed={best_params[stid]['speed']}, "
                    f"meander={best_params[stid]['meander']}\n\n")
            
            # PLOT THE COMPARISON BETWEEN SIMULATIONS AND OBSERVATIONS
            plot_comparison(stid, observation['observed_data'], data, dates)
            
        # VERBOSE MESSAGES
        print('Results saved to ' + 
              os.path.join(c_output_path + f'calibration_tier_{tier}.txt'))
        print('Plots saved to ' + 
              os.path.join(c_output_path,'stid*.png')+'\n')
    
    return best_params


if __name__ == "__main__":
    '''
    This function reads command-line args and call main() to run TRIPpy.
    '''
    
    # READ ARGUMENTS
    parser = argparse.ArgumentParser(description='run TRIPpy script with '
                                     'optional verbose output.')
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='Enable verbose output')
    args = parser.parse_args()
    
    # CALL THE MAIN FUNCTION
    main()
