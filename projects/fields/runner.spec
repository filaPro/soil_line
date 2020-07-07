# -*- mode: python ; coding: utf-8 -*-
import os
import sys

spec_root = os.path.abspath(SPECPATH)

block_cipher = None


a = Analysis(['runner.py'],
            pathex=[spec_root],
             binaries=[],
             datas=[(os.path.join(os.path.split(sys.executable)[0], 'Library', 'share', 'proj', '*'), 'proj')],
             hiddenimports=['six'],
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
          name='runner',
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
               name='runner')
