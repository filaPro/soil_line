from cx_Freeze import setup, Executable

base = None
executables = [Executable("app.py", base=base),
               Executable("preprocess.py", base=base),
               Executable("classify.py", base=base)]
packages = ["os", "json", "numpy", "pandas", "osgeo", "osgeo._gdal", "argparse", "lib", "numpy.lib.format", "lib.imp"]
options = {
    'build_exe': {
        'packages': packages
    },
}
setup(
    name="<any name>",
    options=options,
    version="0.0.1",
    description='<any description>',
    executables=executables, requires=['numpy', 'pandas', 'cv2']
)
