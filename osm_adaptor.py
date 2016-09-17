#!/usr/bin/env python
#******************************************************************************
# From $Id: gdal2tiles.py 19288 2010-04-02 18:36:17Z rouault $
# VERSION MODIFIED FROM ORIGINAL, come with no warranty
# Yves Cainaud
# input: vrt file (-addalpha) in 3857 projection (projection is forced due
# to weird effect in AutoCreateWarpedVRT)
# 2 bands: 1 grayscale, one alpha mask

import sqlite3

import os
import math

__version__ = "$Id: gdal2tiles.py 19288 2010-04-02 18:36:17Z rouault $"

class SqliteTileStorage():
    """ Sqlite files methods for simple tile storage"""

    def __init__(self, type):
        self.type=type
    
    def create(self, filename, overwrite=False):
        """ Create a new storage file, overwrite or not if already exists"""
        self.filename=filename
        CREATEINDEX=True
        
        if overwrite:
            if os.path.isfile(self.filename):
                os.unlink(self.filename)
        else:
            if os.path.isfile(self.filename):
                CREATEINDEX=False
                
        self.db = sqlite3.connect(self.filename)
        
        cur = self.db.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tiles (
                x int,
                y int,
                z int, 
                s int,
                image blob,
                PRIMARY KEY(x,y,z,s))
            """)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS info (
                desc TEXT,
                tilenumbering TEXT,
                minzoom int,
                maxzoom int)
            """)
        
        if CREATEINDEX:
            cur.execute(
                """
                CREATE INDEX IND
                ON tiles(x,y,z,s)
                """)
                
        cur.execute("insert into info(desc, tilenumbering) values('Simple sqlite tile storage..', (?))",  (self.type, ))
        
        self.minzoom = None
        self.maxzoom = None
        self.written = set()
        self.db.commit()
        self.pending_images = []
        
    def open(self, filename) :
        """ Open an existing file"""
        self.filename=filename
        if os.path.isfile(self.filename):
            self.db = sqlite3.connect(self.filename)
            return True
        else:
            return False
            
    def close(self):
        self.commitData(force=True)
        cur = self.db.cursor()
        cur.execute("UPDATE Info SET minzoom = (?), maxzoom = (?)",  (self.minzoom,  self.maxzoom))
        self.db.commit()
    
    def writeImageFile(self, x, y, z, f) :
        """ write a single tile from a file """
        self.writeImage(x,  y,  z,  f.read())
        
    def writeImage(self, x, y, z, image) :
        """ write a single tile from string """
        if (x, y, z) in self.written:
            return
        self.written.add((x, y, z))
        self.pending_images.append((z, x, y, 0, sqlite3.Binary(image)))
        if self.minzoom is None or z < self.minzoom:
            self.minzoom = z
        if self.maxzoom is None or z > self.maxzoom:
            self.maxzoom = z
        self.commitData()
        
    def commitData(self,  force = False):
        if len(self.pending_images) > 500 or force:
            cur = self.db.cursor()
            cur.executemany('insert into tiles (z, x, y,s,image) \
                            values (?,?,?,?,?)',
                                        self.pending_images)
            self.pending_images = []
            self.db.commit()
            
    def readImage(self, x, y, z) :
        """ read a single tile as string """
        
        cur = self.db.cursor()
        cur.execute("select image from tiles where x=? and y=? and z=?", (x, y, z))
        res = cur.fetchone()
        if res:
            image = str(res[0])
            return image
        else :
            print ("None found")
            return None
        
    def createFromDirectory(self, filename, basedir, overwrite=False) :
        """ Create a new sqlite file from a z/y/x.ext directory structure"""
        
        
        self.create(filename, overwrite)
        
        for zs in os.listdir(basedir):
            zz=int(zs)
            for xs in os.listdir(basedir+'/'+zs+'/'):
                xx=int(xs)
                for ys in os.listdir(basedir+'/'+zs+'/'+'/'+xs+'/'):
                    yy=int(ys.split('.')[0])
                    print (zz, yy, xx)
                    z=zz
                    x=xx
                    y=yy
                    print (basedir+'/'+zs+'/'+'/'+xs+'/'+ys)
                    f=open(basedir+'/'+zs+'/'+'/'+xs+'/'+ys)
                    self.writeImageFile(x,  y,  z,  f)
                    #cur.execute('insert into tiles (z, x, y,image) \
                      #          values (?,?,?,?)',
                        #        (z, x, y,  sqlite3.Binary(f.read())))
                                
    def createBigPlanetFromTMS(self, targetname, overwrite=False):
        """ Create a new sqlite with BigPlanet numbering scheme from a TMS one"""
        target=SqliteTileStorage('BigPlanet')
        target.create(targetname, overwrite)
        cur = self.db.cursor()
        cur.execute("select x, y, z from tiles")
        res = cur.fetchall()
        for (x, y, z) in res:
            xx= x
            zz= 17 - z
            yy= 2**zz - y -1
            im=self.readImage(x,y,z)
            target.writeImage(xx,yy,zz,im)
        
    def createTMSFromBigPlanet(self, targetname, overwrite=False):
        """ Create a new sqlite with TMS numbering scheme from a BigPlanet one"""
        target=SqliteTileStorage('TMS')
        target.create(targetname, overwrite)
        cur = self.db.cursor()
        cur.execute("select x, y, z from tiles")
        res = cur.fetchall()
        for (x, y, z) in res:
            xx= x
            zz= 17 - z
            yy= 2**zz - y -1
            im=self.readImage(x,y,z)
            target.writeImage(xx,yy,zz,im)
    
    def createTMSFromOSM(self, targetname, overwrite=False):
        """ Create a new sqlite with TMS numbering scheme from a OSM/Bing/Googlemaps one"""
        target=SqliteTileStorage('TMS')
        target.create(targetname, overwrite)
        cur = self.db.cursor()
        cur.execute("select x, y, z from tiles")
        res = cur.fetchall()
        for (x, y, z) in res:
            xx= x
            zz= z
            yy= 2**zz - y
            im=self.readImage(x,y,z)
            target.writeImage(xx,yy,zz,im)
    
    def createOSMFromTMS(self, targetname, overwrite=False):
        """ Create a new sqlite with OSM/Bing/Googlemaps numbering scheme from a TMS one"""
        target=SqliteTileStorage('OSM')
        target.create(targetname, overwrite)
        cur = self.db.cursor()
        cur.execute("select x, y, z from tiles")
        res = cur.fetchall()
        for (x, y, z) in res:
            xx= x
            zz= z
            yy= 2**zz - y
            im=self.readImage(x,y,z)
            target.writeImage(xx,yy,zz,im)
        

# =============================================================================
# =============================================================================
# =============================================================================

__doc__globalmaptiles = """
globalmaptiles.py
Global Map Tiles as defined in Tile Map Service (TMS) Profiles
==============================================================
Functions necessary for generation of global tiles used on the web.
It contains classes implementing coordinate conversions for:
  - GlobalMercator (based on EPSG:900913 = EPSG:3785)
       for Google Maps, Yahoo Maps, Microsoft Maps compatible tiles
  - GlobalGeodetic (based on EPSG:4326)
       for OpenLayers Base Map and Google Earth compatible tiles
More info at:
http://wiki.osgeo.org/wiki/Tile_Map_Service_Specification
http://wiki.osgeo.org/wiki/WMS_Tiling_Client_Recommendation
http://msdn.microsoft.com/en-us/library/bb259689.aspx
http://code.google.com/apis/maps/documentation/overlays.html#Google_Maps_Coordinates
Created by Klokan Petr Pridal on 2008-07-03.
Google Summer of Code 2008, project GDAL2Tiles for OSGEO.
In case you use this class in your product, translate it to another language
or find it usefull for your project please let me know.
My email: klokan at klokan dot cz.
I would like to know where it was used.
Class is available under the open-source GDAL license (www.gdal.org).
"""


MAXZOOMLEVEL = 32

class GlobalMercator(object):
    """
    TMS Global Mercator Profile
    ---------------------------
    Functions necessary for generation of tiles in Spherical Mercator projection,
    EPSG:900913 (EPSG:gOOglE, Google Maps Global Mercator), EPSG:3785, OSGEO:41001.
    Such tiles are compatible with Google Maps, Microsoft Virtual Earth, Yahoo Maps,
    UK Ordnance Survey OpenSpace API, ...
    and you can overlay them on top of base maps of those web mapping applications.
    
    Pixel and tile coordinates are in TMS notation (origin [0,0] in bottom-left).
    What coordinate conversions do we need for TMS Global Mercator tiles::
         LatLon      <->       Meters      <->     Pixels    <->       Tile     
     WGS84 coordinates   Spherical Mercator  Pixels in pyramid  Tiles in pyramid
         lat/lon            XY in metres     XY pixels Z zoom      XYZ from TMS 
        EPSG:4326           EPSG:900913                                         
         .----.              ---------               --                TMS      
        /      \     <->     |       |     <->     /----/    <->      Google    
        \      /             |       |           /--------/          QuadTree   
         -----               ---------         /------------/                   
       KML, public         WebMapService         Web Clients      TileMapService
    What is the coordinate extent of Earth in EPSG:900913?
      [-20037508.342789244, -20037508.342789244, 20037508.342789244, 20037508.342789244]
      Constant 20037508.342789244 comes from the circumference of the Earth in meters,
      which is 40 thousand kilometers, the coordinate origin is in the middle of extent.
      In fact you can calculate the constant as: 2 * math.pi * 6378137 / 2.0
      $ echo 180 85 | gdaltransform -s_srs EPSG:4326 -t_srs EPSG:900913
      Polar areas with abs(latitude) bigger then 85.05112878 are clipped off.
    What are zoom level constants (pixels/meter) for pyramid with EPSG:900913?
      whole region is on top of pyramid (zoom=0) covered by 256x256 pixels tile,
      every lower zoom level resolution is always divided by two
      initialResolution = 20037508.342789244 * 2 / 256 = 156543.03392804062
    What is the difference between TMS and Google Maps/QuadTree tile name convention?
      The tile raster itself is the same (equal extent, projection, pixel size),
      there is just different identification of the same raster tile.
      Tiles in TMS are counted from [0,0] in the bottom-left corner, id is XYZ.
      Google placed the origin [0,0] to the top-left corner, reference is XYZ.
      Microsoft is referencing tiles by a QuadTree name, defined on the website:
      http://msdn2.microsoft.com/en-us/library/bb259689.aspx
    The lat/lon coordinates are using WGS84 datum, yeh?
      Yes, all lat/lon we are mentioning should use WGS84 Geodetic Datum.
      Well, the web clients like Google Maps are projecting those coordinates by
      Spherical Mercator, so in fact lat/lon coordinates on sphere are treated as if
      the were on the WGS84 ellipsoid.
     
      From MSDN documentation:
      To simplify the calculations, we use the spherical form of projection, not
      the ellipsoidal form. Since the projection is used only for map display,
      and not for displaying numeric coordinates, we don't need the extra precision
      of an ellipsoidal projection. The spherical projection causes approximately
      0.33 percent scale distortion in the Y direction, which is not visually noticable.
    How do I create a raster in EPSG:900913 and convert coordinates with PROJ.4?
      You can use standard GIS tools like gdalwarp, cs2cs or gdaltransform.
      All of the tools supports -t_srs 'epsg:900913'.
      For other GIS programs check the exact definition of the projection:
      More info at http://spatialreference.org/ref/user/google-projection/
      The same projection is degined as EPSG:3785. WKT definition is in the official
      EPSG database.
      Proj4 Text:
        +proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0
        +k=1.0 +units=m +nadgrids=@null +no_defs
      Human readable WKT format of EPGS:900913:
         PROJCS["Google Maps Global Mercator",
             GEOGCS["WGS 84",
                 DATUM["WGS_1984",
                     SPHEROID["WGS 84",6378137,298.257223563,
                         AUTHORITY["EPSG","7030"]],
                     AUTHORITY["EPSG","6326"]],
                 PRIMEM["Greenwich",0],
                 UNIT["degree",0.0174532925199433],
                 AUTHORITY["EPSG","4326"]],
             PROJECTION["Mercator_1SP"],
             PARAMETER["central_meridian",0],
             PARAMETER["scale_factor",1],
             PARAMETER["false_easting",0],
             PARAMETER["false_northing",0],
             UNIT["metre",1,
                 AUTHORITY["EPSG","9001"]]]
    """

    def __init__(self, tileSize=256):
        "Initialize the TMS Global Mercator pyramid"
        self.tileSize = tileSize
        self.initialResolution = 2 * math.pi * 6378137 / self.tileSize
        # 156543.03392804062 for tileSize 256 pixels
        self.originShift = 2 * math.pi * 6378137 / 2.0
        # 20037508.342789244

    def LatLonToMeters(self, lat, lon ):
        "Converts given lat/lon in WGS84 Datum to XY in Spherical Mercator EPSG:900913"

        mx = lon * self.originShift / 180.0
        my = math.log( math.tan((90 + lat) * math.pi / 360.0 )) / (math.pi / 180.0)

        my = my * self.originShift / 180.0
        return mx, my

    def MetersToLatLon(self, mx, my ):
        "Converts XY point from Spherical Mercator EPSG:900913 to lat/lon in WGS84 Datum"

        lon = (mx / self.originShift) * 180.0
        lat = (my / self.originShift) * 180.0

        lat = 180 / math.pi * (2 * math.atan( math.exp( lat * math.pi / 180.0)) - math.pi / 2.0)
        return lat, lon

    def PixelsToMeters(self, px, pyr, zoom):
        "Converts pixel coordinates in given zoom level of pyramid to EPSG:900913"

        mapSize = self.tileSize << zoom
        py = mapSize - pyr
        res = self.Resolution( zoom )
        mx = px * res - self.originShift
        my = py * res - self.originShift
        return mx, my
        
    def MetersToPixels(self, mx, my, zoom):
        "Converts EPSG:900913 to pyramid pixel coordinates in given zoom level"
                
        res = self.Resolution( zoom )
        px = (mx + self.originShift) / res
        py = (my + self.originShift) / res
        mapSize = self.tileSize << zoom
        return px, mapSize - py
    
    def PixelsToTile(self, px, py):
        "Returns a tile covering region in given pixel coordinates"

        tx = int( math.ceil( px / float(self.tileSize) ) - 1 )
        ty = int( math.ceil( py / float(self.tileSize) ) - 1 )
        return tx, ty

    #def PixelsToRaster(self, px, py, zoom):
    #    "Move the origin of pixel coordinates to top-left corner"
    #    
    #    mapSize = self.tileSize << zoom
    #    return px, mapSize - py
        
    def MetersToTile(self, mx, my, zoom):
        "Returns tile for given mercator coordinates"
        
        px, py = self.MetersToPixels( mx, my, zoom)
        return self.PixelsToTile( px, py)

    def TileBounds(self, tx, ty, zoom):
        "Returns bounds of the given tile in EPSG:900913 coordinates"
        
        minx, miny = self.PixelsToMeters( tx*self.tileSize, (ty+1)*self.tileSize, zoom )
        maxx, maxy = self.PixelsToMeters( (tx+1)*self.tileSize, (ty)*self.tileSize, zoom )
        return ( minx, miny, maxx, maxy )

    def TileLatLonBounds(self, tx, ty, zoom ):
        "Returns bounds of the given tile in latutude/longitude using WGS84 datum"

        bounds = self.TileBounds( tx, ty, zoom)
        minLat, minLon = self.MetersToLatLon(bounds[0], bounds[1])
        maxLat, maxLon = self.MetersToLatLon(bounds[2], bounds[3])
         
        return ( minLat, minLon, maxLat, maxLon )

    def TileLatLonCorners(self,  tx,  ty,  zoom ):
        p1_lat,  p1_lon,  p3_lat,  p3_lon = self.TileLatLonBounds(tx,  ty,  zoom)
        p2_lat,  p2_lon,  _ , _ = self.TileLatLonBounds(tx+1,  ty,  zoom)
        p4_lat,  p4_lon,  _,  _ = self.TileLatLonBounds(tx,  ty-1,  zoom)
        return (p1_lat,  p1_lon,  p2_lat,  p2_lon,  p3_lat,  p3_lon,  p4_lat,  p4_lon)
    
    def Resolution(self, zoom ):
        "Resolution (meters/pixel) for given zoom level (measured at Equator)"
        
        # return (2 * math.pi * 6378137) / (self.tileSize * 2**zoom)
        return self.initialResolution / (2**zoom)
        
    def ZoomForPixelSize(self, pixelSize ):
        "Maximal scaledown zoom of the pyramid closest to the pixelSize."
        
        for i in range(MAXZOOMLEVEL):
            if pixelSize > self.Resolution(i):
                if i!=0:
                    return i-1
                else:
                    return 0 # We don't want to scale up
        
    def GoogleTile(self, tx, ty, zoom):
        "Converts TMS tile coordinates to Google Tile coordinates"
        
        # coordinate origin is moved from bottom-left to top-left corner of the extent
        return tx, (2**zoom - 1) - ty

    def QuadTree(self, tx, ty, zoom ):
        "Converts TMS tile coordinates to Microsoft QuadTree"
        
        quadKey = ""
        ty = (2**zoom - 1) - ty
        for i in range(zoom, 0, -1):
            digit = 0
            mask = 1 << (i-1)
            if (tx & mask) != 0:
                digit += 1
            if (ty & mask) != 0:
                digit += 2
            quadKey += str(digit)
            
        return quadKey
