# -*- mode: python ; coding: utf-8 -*-
import sys

block_cipher = None


a = Analysis(['run.py'],
             pathex=['D:\\PythonProjects\\soil_line'],
             binaries=[],
             datas=[(os.path.join(os.path.split(sys.executable)[0], '..\\Lib\\site-packages\\osgeo\\data\\proj\\*'), 'osgeo\\data\\proj')],
             hiddenimports=[],
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
          name='run',
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
               name='soilline_coefficients')
