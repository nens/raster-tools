from setuptools import setup

version = '0.1dev'

long_description = '\n\n'.join([
    open('README.rst').read(),
    open('CREDITS.rst').read(),
    open('CHANGES.rst').read(),
    ])

install_requires = [
    'gdal',
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
              # ahn2
              'ahn2-aig2tif       = raster_tools.ahn2.aig2tif:main',
              'ahn2-zip2tif       = raster_tools.ahn2.zip2tif:main',
              'ahn2-constant      = raster_tools.ahn2.constant:main',
              # # ahn3
              'ahn3-download      = raster_tools.ahn3.download:main',
              # # srtm
              'srtm-make-index    = raster_tools.srtm.make_index:main',
              'srtm-fix-nodata    = raster_tools.srtm.fix_nodata:main',
              'srtm-organize      = raster_tools.srtm.organize:main',
              # modification
              'mfillnodata         = raster_tools.mfillnodata:main',
              'fillnodata         = raster_tools.fillnodata:main',
              'hillshade          = raster_tools.hillshade:main',
              'aggregate          = raster_tools.aggregate:main',
              'rebase             = raster_tools.rebase:main',
              'shadow             = raster_tools.shadow:main',
              'reindex              = raster_tools.reindex:main',
              # rasterization
              'bag2tif            = raster_tools.bag2tif:main',
              'rasterize-landuse  = raster_tools.rasterize_landuse:main',
              'rasterize          = raster_tools.rasterize:main',
              # modeling
              'extract            = raster_tools.extract:main',
              'line-up            = raster_tools.line_up:main',
              # pointclouds
              'txt2tif            = raster_tools.txt2tif:main',
          ]},
      )
