# -*- mode: python ; coding: utf-8 -*-

import sys

block_cipher = None

from PyInstaller.utils.hooks import collect_data_files  # this is very helpful

import os
rasterio_imports_paths = os.listdir(r'D:\PythonProjects\soil_line\venv\Lib\site-packages\rasterio')
rasterio_imports = ['rasterio._shim']

for item in rasterio_imports_paths:
    current_module_filename = os.path.split(item)[-1]
    if current_module_filename.endswith('py'):
        current_module_filename = 'rasterio.' + current_module_filename[:-3]
        rasterio_imports.append(current_module_filename)

a = Analysis(['runner_predict_catboost_v1.py'],
             pathex=[],
             binaries=[
                    #(os.path.join(bins,'*.dll'),'.'),
                    ],
             datas=collect_data_files('torch', subdir='lib') +
                   collect_data_files('geopandas', subdir='datasets') +
                   collect_data_files('osgeo'),
             hiddenimports=[
                    'fiona',
                    'fiona._loading',
                    'fiona._shim',
                    'fiona.schema',
                    ] + rasterio_imports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='predict_catboost_v1',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='predict_catboost_v1')
