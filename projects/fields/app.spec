# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['projects\\fields\\app.py'],
             pathex=['.'],
             binaries=[],
             datas=[("C:\\ProgramData\\Miniconda3\\envs\\soil_line\\Library\\share\\proj\\*", 'proj')],
             hiddenimports=["os", "json", "numpy", "pandas", "osgeo", "osgeo._gdal", "argparse", "lib", "numpy.lib.format", "lib.imp", "numpy.core._dtype_ctypes"],
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
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='app',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
