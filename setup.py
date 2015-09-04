from setuptools import setup

version = '0.1dev'

long_description = '\n\n'.join([
    open('README.rst').read(),
    open('CREDITS.rst').read(),
    open('CHANGES.rst').read(),
    ])

install_requires = [
    'gdal',
    'pypotrace',
    'psycopg2',
    'scipy',
    'setuptools',
    'unipath',
    'numpy >= 1.8.2',
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
              # organization
              'ahn2aig2tif        = raster_tools.ahn2aig2tif:main',
              'ahn2zip2tif        = raster_tools.ahn2zip2tif:main',
              'srtm-make-index    = raster_tools.srtm_make_index:main',
              'srtm-fix-nodata    = raster_tools.srtm_fix_nodata:main',
              'srtm-organize      = raster_tools.srtm_organize:main',
              # modification
              'fillnodata         = raster_tools.fillnodata:main',
              'rebase             = raster_tools.rebase:main',
              'smooth             = raster_tools.smooth:main',
              # rasterization
              'bag2tif            = raster_tools.bag2tif:main',
              'interpolate        = raster_tools.interpolate:main',
              'rasterize-landuse  = raster_tools.rasterize_landuse:main',
              'rasterize          = raster_tools.rasterize:main',
              # model generation
              'extract            = raster_tools.extract:main',
              # pointclouds
              'txt2tif            = raster_tools.txt2tif:main',
          ]},
      )
