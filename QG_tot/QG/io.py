"""Model input/output tools."""
# (c) Copyright 2021 Nick Lutsko; Will Chapman; Momme Helle.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import (absolute_import, division, print_function)  #noqa
import os.path
from netCDF4 import Dataset
import numpy as np

class NetCDFWriter(object):
    """Write model output to a NetCDF file."""

    def __init__(self, model, filename, overwrite=True):
        """
        Initialize a netCDF output writer object for a given model.
        Arguments:
        * model
            An instance of `QG.model.QG2d_Model` to provide
            output services to.
        * filename
            The name of the NetCDF file to use for output.
        Keyword argument:
        * overwrite
            If `True` the writer will overwrite the specified file if it
            already exists, and if `False` an error will be raised if
            the specified filename already exists. Default is `True`
            (existing file will be overwritten).
        """
        self.model = model
        self.filename = filename
        if not overwrite and os.path.exists(filename):
            msg = ('cannot write output to "{}", the file already '
                   'exists but overwrite=False')
            raise IOError(msg.format(filename))
        # Open a new netCDF dataset:
        self.ds = Dataset(filename, mode='w')
        # Create dimensions for time, latitude and longitude, the time
        # dimension has unlimited size:
        self.ds.createDimension('time')
        self.ds.createDimension('x', size=model.x.shape[0])
        self.ds.createDimension('y', size=model.y.shape[0])
        # Create coordinate variables for time, x and y, the
        # values of depth can be set immediately:
        self.time = self.ds.createVariable('time', 'f4', dimensions=['time'])
        #time_units = 'seconds since {}'.format(
            #model.start_time.strftime('%Y-%m-%d %H:%M:%S'))
        #self.time.setncatts({'standard_name': 'time', 'units': time_units})
        x = self.ds.createVariable('x', 'f4',
                                          dimensions=['x'])
        y = self.ds.createVariable('y', 'f4',
                                           dimensions=['y'])
        x.setncatts({'standard_name': 'x',
                            'units': 'degrees_north'})
        y.setncatts({'standard_name': 'y',
                             'units': 'degrees_east'})
        x[:] = self.model.x
        y[:] = self.model.y
        #lat_lon = self.ds.createVariable('latitude_longitude', 'i4')
        #lat_lon.setncatts({'grid_mapping_name': 'latitude_longitude',
                           #'longitude_of_prime_meridian': 0.,'semi_major_axis': 6371229.,
                   #'semi_minor_axis': 6371229.})
        # Create variables to hold the model state:
        self.psic1 = self.ds.createVariable(
            'psic1',
            'f4',
            dimensions=['time', 'x', 'y'],
            zlib=True)
        self.psic2 = self.ds.createVariable(
            'psic2',
            'f4',
            dimensions=['time', 'x', 'y'],
            zlib=True)
        self.q1 = self.ds.createVariable(
            'PV1',
            'f4',
            dimensions=['time', 'x', 'y'],
            zlib=True)
        
        self.q2 = self.ds.createVariable(
            'PV2',
            'f4',
            dimensions=['time', 'x', 'y'],
            zlib=True)
        self.psic1.setncatts({'standard_name': 'stream funct1',
                          'units': 'dimensionless',
                          'grid_mapping': 'x_y'})
        self.psic2.setncatts({'standard_name': 'stream funct2',
                          'units': 'dimensionless',
                          'grid_mapping': 'x_y'})
        self.q1.setncatts({'standard_name': 'potential_vorticity1',
                            'units': 'dimensionless',
                            'grid_mapping': 'x_y'})
        
        self.q2.setncatts({'standard_name': 'potential_vorticity2',
                            'units': 'dimensionless',
                            'grid_mapping': 'x_y'})

    def save(self):
        """Save the current model state to the output netCDF file."""
        if not self.ds.isopen():
            msg = 'cannot save output: the NetCDF writer is already closed'
            raise IOError(msg)
        index = self.time.size
        self.time[index] = self.model.t
        self.psic1[index] = self.model.Q_1 + np.fft.irfft2( self.model.qc_1[1] )
        self.psic2[index] = self.model.Q_1 + np.fft.irfft2( self.model.qc_1[1] )
        self.q1[index] = self.model.Q_1 + np.fft.irfft2( self.model.qc_1[1] )
        self.q2[index] = self.model.Q_2 + np.fft.irfft2( self.model.qc_2[1] )

    def flush(self):
        """
        Write the output file to disk.
        The netCDF file may be buffered. Whilst calling `save` will
        append a record to the output, it may not be written to disk
        immediately.
        """
        if self.ds.isopen():
            self.ds.sync()

    def close(self):
        """Close the netCDF output file."""
        if self.ds.isopen():
            self.ds.close()