__all__ = ['oec_catalogue', 'find_oec_parameters']

import gzip
import os
import socket
import time
import urllib

import exodata


def oec_catalogue():
    data_base_location = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'oec_data_base')

    data_base_url = 'https://github.com/OpenExoplanetCatalogue/oec_gzip/raw/master/systems.xml.gz'

    data_base_file_path = os.path.join(data_base_location, 'systems.xml')
    last_update_file_path = os.path.join(data_base_location,
                                         'systems_last_update.txt')

    date = time.strftime('%y%m%d')
    update = False
    if not os.path.isfile(last_update_file_path):
        update = True
    elif not os.path.isfile(data_base_file_path):
        update = True
    elif int(open(last_update_file_path).readlines()[0]) < int(date):
        update = True

    if update:

        print 'Updating OEC...'

        try:
            socket.setdefaulttimeout(5)
            urllib.urlretrieve(data_base_url, data_base_file_path + '.gz')
            socket.setdefaulttimeout(30)

            w = open(data_base_file_path, 'w')
            for i in gzip.open(data_base_file_path + '.gz'):
                w.write(i)

            w.close()

            os.remove('{0}.gz'.format(data_base_file_path))

            w = open(last_update_file_path, 'w')
            w.write(date)
            w.close()

        except IOError:
            print 'Updating OEC failed.'
            pass

    return exodata.OECDatabase(data_base_file_path, stream=True)
