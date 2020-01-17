# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


app_a = Analysis(['projects\\fields\\app.py'],
             pathex=['C:\\PythonProjects\\soil_line'],
             binaries=[],
             datas=[("C:\\ProgramData\\Miniconda3\\envs\\soil_line\\Library\\share\\proj\\*", 'proj')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
app_pyz = PYZ(app_a.pure, app_a.zipped_data,
             cipher=block_cipher)
app_exe = EXE(app_pyz,
          app_a.scripts,
          app_a.binaries,
          app_a.zipfiles,
          app_a.datas,
          [],
          name='app',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )


preprocess_a = Analysis(['projects\\fields\\preprocess.py'],
             pathex=['C:\\PythonProjects\\soil_line'],
             binaries=[],
             datas=[("C:\\ProgramData\\Miniconda3\\envs\\soil_line\\Library\\share\\proj\\*", 'proj')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
preprocess_pyz = PYZ(preprocess_a.pure, preprocess_a.zipped_data,
             cipher=block_cipher)
preprocess_exe = EXE(preprocess_pyz,
          preprocess_a.scripts,
          preprocess_a.binaries,
          preprocess_a.zipfiles,
          preprocess_a.datas,
          [],
          name='preprocess',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )


classify_a = Analysis(['projects\\fields\\classify.py'],
             pathex=['C:\\PythonProjects\\soil_line'],
             binaries=[],
             datas=[("C:\\ProgramData\\Miniconda3\\envs\\soil_line\\Library\\share\\proj\\*", 'proj')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
classify_pyz = PYZ(classify_a.pure, classify_a.zipped_data,
             cipher=block_cipher)
classify_exe = EXE(classify_pyz,
          classify_a.scripts,
          classify_a.binaries,
          classify_a.zipfiles,
          classify_a.datas,
          [],
          name='classify',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
