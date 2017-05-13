""" Holds any run parameters and in future will be able to load new ones from
 a par file
"""

import os
import os.path
import urllib

import numpy as np

_rootdir = os.path.dirname(__file__)  # get current directory location
_data_dir = os.path.join(_rootdir, 'data')


class WayneDataError(Exception):
    pass


# darks, flats etc
files_location = os.path.abspath(os.path.dirname(__file__))
files_directory_path = os.path.join(files_location, 'wayne_calibration_files')
calibration_last_update_file_path = os.path.join(files_directory_path, 'last_update_calibration.txt')
calibration_zip_file_path = os.path.join(files_location, 'wayne_calibration_files.zip')
calibration_directory_path = files_directory_path
calibration_url = 'http://zuserver2.star.ucl.ac.uk/~atsiaras/wayne_calibration_files.zip'
calibration_last_update = 20170410

# update calibration files

calibration_update = False
if not os.path.isdir(files_directory_path):
    calibration_update = True
elif not os.path.isfile(calibration_last_update_file_path):
    calibration_update = True
elif int(open(calibration_last_update_file_path).readlines()[0]) < calibration_last_update:
    calibration_update = True

if calibration_update:
    try:
        print '\nDownloading calibration files...'

        if os.path.isdir(calibration_directory_path):
            os.system('rm -rf {0}'.format(calibration_directory_path))

        urllib.urlretrieve(calibration_url, calibration_zip_file_path)

        os.system('unzip {0} -d {1}{2}'.format(calibration_zip_file_path, files_location, os.sep))
        os.system('rm {0}'.format(calibration_zip_file_path))
        os.system('rm -rf {0}{1}__MACOSX'.format(files_location, os.sep))

    except IOError:
        raise WayneDataError('Failed to update wayne calibration files.')


_calb_dir = files_directory_path

seed = np.random.get_state()[1][0]