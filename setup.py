from setuptools import setup

setup(name="molgnn",
	version='1.0',
	description='Molecular Graph Neural Network',
	packages=['molgnn'],
	author='Mesfin Diro',
	author_email='mesfindiro@gmail.com',
	zip_safe=False,
    install_requires=['rdkit-pypi','pyg','pytorch','tqdm','torch-scatter'])