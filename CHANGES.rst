Changelog of raster-tools
===================================================


0.6 (unreleased)
----------------

- Added support for a password file to rextract.


0.5 (2020-05-28)
----------------

- Replaced geom by bbox in rextract requests.


0.4 (2019-09-04)
----------------

- Replaced buildout with pipenv.

- Added 'ruimtekaart' tools.

- Fixed use of deprecated mismatched boolean index size.

- Port various scripts to python3

- Modify extract for a specific task.

- Update bag2tif.

- Added rextract script to extract rasters from Lizard.


0.3 (2018-06-07)
----------------

- Extended extract tool with 3di-ahn3 option.

- Added meta about raster-server layers to extract tool.

- Leanified and documented AHN3 downloader.

- Extended extract tool with 3di-ahn3-hhnk option.

- Extended extract tool with 3di-rd option.

- Add difference script from the past.

- Extended extract tool with 3Di-ahn3-almere option.

- Added option to specify floor.

- Enabled multiple source-files support in retile.

- Add rasterize2 that keeps close to the sql.

- Have extract script use new landuse sources.

- Removed dem:water from extract.

- Rewrite rebase to not use index shapefile.


0.1 (2016-12-12)
----------------

- Intend to release this package more often.

- Fix bug for polygon containing exacly four rings.

- Extract tool added for 3Di 2D model generation.

- Initial project structure created with nensskel 1.34.dev0.
