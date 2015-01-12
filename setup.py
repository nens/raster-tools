from setuptools import setup

version = '0.1dev'

long_description = '\n\n'.join([
    open('README.rst').read(),
    open('CREDITS.rst').read(),
    open('CHANGES.rst').read(),
    ])

install_requires = [
    'setuptools',
    ],

tests_require = [
    'nose',
    'coverage',
    ]

setup(name='raster-tools',
      version=version,
      description="Tools for processing of elevation and landuse raster data.",
      long_description=long_description,
      # Get strings from http://www.python.org/pypi?%3Aaction=list_classifiers
      classifiers=[],
      keywords=[],
      author='Arjan Verkerk',
      author_email='arjan.verkerk@nelen-schuurmans.nl',
      url='',
      license='GPL',
      packages=['raster_tools'],
      zip_safe=False,
      install_requires=install_requires,
      tests_require=tests_require,
      extras_require={'test': tests_require},
      entry_points={
          'console_scripts': [
              'juggle = raster_tools.juggle:main',
              'interpolate = raster_tools.interpolate:main',
              'fillnodata = raster_tools.fillnodata:main',
              'ahn2aig2tif = raster_tools.ahn2aig2tif:main',
              'ahn2zip2tif = raster_tools.ahn2zip2tif:main',
              'rasterize = raster_tools.rasterize:main',
              'rasterize-landuse = raster_tools.rasterize_landuse:main',
              'extract = raster_tools.extract:main',
              'watershed = raster_tools.watershed:main',
              'interpolate_points = raster_tools.interpolate_points:main',
              'filter_paths = raster_tools.filter_paths:main',
              # pointclouds
              'txt2tif = raster_tools.txt2tif:main',
          ]},
      )
