# Release Notes

## Version 2.3 - 2025-10-17
- **Optimization:** Replacing the netCDF I/O operations during parameter calibration with in-memory storage, reducing disk operations.

## Version 2.2 - 2025-10-16
- **Optimization:** Replacing the netCDF I/O operations during calibration with in-memory storage, significantly reducing disk operations and processing time. 

## Version 2.1 - 2025-04-14
- **Restart:** Incorporates an option in the namelist to restart the calibration process.

## Version 2.0 - 2024-08-27
- **Optimization:** Faster calculations for river network parameters.
- **Data Handling:** Ability to read meander and flow velocity for each grid cell from a netCDF file.
- **Manual Calibration:** Incorporates manual calibration for enhanced control.
- **Enhanced Configuration:** Improved and modulated `namelist.input` for better flexibility.

## Version 1.0 - 2023-07-31
- **Initial Release:** This is the very first version of the model, which allows regional and global river routing simulations.
