import os

def pyparallel_compile(threads):
    files_location = os.path.abspath(os.path.dirname(__file__))
    compiled = int(open(os.path.join(files_location, 'pyparallel_compiling_log.txt')).readlines()[0].split()[0])
    if not compiled:
        if threads is None:
            threads = 2
        ww = open(os.path.join(files_location, 'pyparallel_menu.c'), 'w')
        ww.write(open(os.path.join(files_location, 'pyparallel_menu_model.c')).read().replace('xxxxxx', str(threads)))
        ww.close()
        os.system("python {0}".format(os.path.join(files_location, 'pyparallel_setup.py build_ext --inplace')))
        ww=open(os.path.join(files_location, 'pyparallel_compiling_log.txt'), 'w')
        ww.write('1\n'+str(threads))
    else:
        if threads is not None:
            if int(open(os.path.join(files_location, 'pyparallel_compiling_log.txt')).readlines()[1].split()[0]) != threads:
                ww = open(os.path.join(files_location, 'pyparallel_menu.c'), 'w')
                ww.write(open(os.path.join(files_location, 'pyparallel_menu_model.c')).read().replace('xxxxxx', str(threads)))
                ww.close()
                os.system("python {0}".format(os.path.join(files_location, 'pyparallel_setup.py build_ext --inplace')))
                ww=open(os.path.join(files_location, 'pyparallel_compiling_log.txt'), 'w')
                ww.write('1\n'+str(threads))                
   