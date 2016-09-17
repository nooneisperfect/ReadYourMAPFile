# -*- coding: utf-8 -*-

# python imports
import glob
import os
import re
import bisect
from threading import RLock
# numpy imports
import numpy as np
import numpy
import numpy.linalg as linalg
# local imports
from smlogging import *
from mapcoord import UTMZoneForward, UTMZoneReverse, SwissReverse, SwissForward
from SatmapFormat import load_smt
from PIL import Image

def mapCoord2deg(x, y, zone, scale, code):
    if code == 326:
        (E,N) = list(map(lambda v: v*1000./scale, (x,y)))
        return list(map(lambda x: np.degrees(x), 
                        UTMZoneReverse(E, N, zone, FN = 328.)))
    elif code == 217:
        (X,Y) = list(map(lambda v: v*1000./scale, (x,y)))
        deg =  list(map(lambda x: np.degrees(x), 
                   SwissReverse(X, Y)))
        # heuristic mapping, unclear why this is necessary
        deg[0] = deg[0] - 0.000933
        deg[1] = deg[1] - 0.001408
        return deg
    else:
        assert False
        
def deg2mapCoord(lmb, phi, zone, scale, code):
    if code == 326:
        (l,p) = list(map(np.radians, (lmb, phi)))
        (E,N) = UTMZoneForward(l, p, zone)
        return list(map(lambda x: x*scale/1000., (E,N)))
    elif code == 217:
        # heuristic mapping, unclear why this is necessary
        lmb += 0.000933
        phi += 0.001408
        (l,p) = list(map(np.radians, (lmb, phi)))
        (X,Y) = SwissForward(l, p)
        return list(map(lambda x: x*scale/1000., (X,Y)))
    else:
        assert False
        
class FastTileCalculator:
    
    def __init__(self, minx, maxx, w, miny, maxy, h, zone, scale, code):
        log( INFO, "Setting up fast tile calculator")
        self.minx = minx
        self.w = w
        self.miny = miny
        self.h = h
        self.y, self.x = np.mgrid[miny:maxy+h*1.1:h, minx:maxx+w*1.1:w]
        self.P_deg = mapCoord2deg(self.x, self.y, zone, scale, code)
        # small overlapping (2*o pixel) between neighbored tiles to overcome 
        # errors in affine transformation
        o = 0.1
        A = np.array([[o  , o  , 1, 0  , 0  , 0],
                      [0  , 0  , 0, o  , o  , 1],
                      [w-o, o  , 1, 0  , 0  , 0],
                      [0  , 0  , 0, w-o, o  , 1],
                      [w-o, h-o, 1, 0  , 0  , 0],
                      [0  , 0  , 0, w-o, h-o, 1],
                      [o  , h-o, 1, 0  , 0  , 0],
                      [0  , 0  , 0, o  , h-o, 1]])
        Ai = linalg.pinv(A)
        b = np.array([self.P_deg[0][:-1,:-1],self.P_deg[1][:-1,:-1],
                      self.P_deg[0][:-1, 1:],self.P_deg[1][:-1, 1:],
                      self.P_deg[0][ 1:, 1:],self.P_deg[1][ 1:, 1:],
                      self.P_deg[0][ 1:,:-1],self.P_deg[1][ 1:,:-1],
                      ])
        br = np.reshape(b, (b.shape[0], b.shape[1]*b.shape[2]))
        self.M = np.reshape(np.dot(Ai, br), (6, b.shape[1], b.shape[2]))
        log( INFO, "Setting up fast tile calculator done [P_degx = %s P_degy = %s]" % (str(self.P_deg[0][0:2,0:2]), str(self.P_deg[1][0:2,0:2])))
        
    def calc(self, x, y):
        ix = (x - self.minx)/self.w
        iy = (y - self.miny)/self.h
        try:
            return self.M[:, iy, ix]
        except IndexError:
            print (x, y, ix, iy, self.minx, self.w, self.miny, self.h, self.M.shape)
            raise


class Tile:
        def __init__(self,  tc,  smt,  openfct,  x,  y,  zone,  scale,  code,  w,  h,  fastTileCalculator):
            self.tc = tc
            self.smt = smt
            self.openfct = openfct
            self.x_mc = x
            self.y_mc = y
            self.zone = zone
            self.scale = scale
            self.code = code
            self.w = w
            self.h = h
            long1,  lat1 = mapCoord2deg(x,  y,  zone,  scale,  code)
            long2,  lat2 = mapCoord2deg(x+w, y+h,  zone,  scale,  code)
            self.long_min = min(long1,  long2)
            self.long_max = max(long1,  long2)
            self.lat_min = min(lat1,  lat2)
            self.lat_max = max(lat1,  lat2)
            log( INFO,  "Tile(%s %s %s %s %s %s %s %.5f %.5f %.5f %.5f" % (x,  y,  w,  h,  zone,  scale,  code,  self.lat_min,  self.lat_max,  self.long_min,  self.long_max))
        
        def load(self):
            im = self.tc.load_tile(self.smt, self.openfct)
            return im
            

class TileCache:
    
    cacheSize = 400
    
    def __init__(self):
        self.cache = {}
        self.cache_access = []
        self.mutex = RLock()

    def load_tile(self, smt, openfct):
        with self.mutex:
            def imgConstruct(f):
                cnt = 0
                while cnt < 10:
                    try:
                        res = Image.open(f)
                        return res.transpose(Image.FLIP_TOP_BOTTOM)
                    except OSError:
                        cnt += 1
                return Image.new('RGB',  (400,  400))
            if smt in self.cache:
                self.cache_access.remove(smt)
                self.cache_access.append(smt)
                return self.cache[smt]
            if len(self.cache_access) > self.cacheSize:
                pop = self.cache_access[0]
                self.cache_access = self.cache_access[1:]
                del self.cache[pop]
            self.cache_access.append(smt)
            self.cache[smt] = load_smt(smt, openfct, imgConstruct)
            assert set(self.cache_access) == set(self.cache.keys()), str(set(self.cache.keys())) + "<->" + str(set(self.cache_access))
            return self.load_tile(smt, openfct)

tileCache = TileCache()

class TileContainer:
    
    tileH = 400
    tileW = 400
    cacheSize = 40
    # filename decoding:
    # AAAABBBCCDDD[+-]EFFF[+-]GHHH[+-]JKKKKKKKLLLLLLL.SMT
    # with 
    #    AAAA is a map indicator (probably)
    #     BBB is the coordinate code
    #      CC is the zone in UTM or otherwise part of coordinate code
    #     DDD mantissa of scale
    #       E exponent of scale
    #     FFF unused (maybe mantissa of offset?)
    #       G unused (maybe exponent of offset?)
    #     HHH unused 
    #       J unused
    # KKKKKKK first element of coordinate
    # LLLLLLL second element ofo coordinate
    tilere = re.compile("[0-9]{4}([0-9]{3})([0-9]{2})([0-9]{3})([+-][0-9])[0-9]{3}[+-][0-9]{4}[+-][0-9]([0-9]{7})([0-9]{7}).SMT")
    #                    AAAA     BBB       CC        DDD            E     FFF         GHHH        J    KKKKKKK   LLLLLLL
    def __init__(self, smtglob, progress):
        ftc = None
        if type(smtglob) == type(""):
            smts = [(x, lambda f: open(f, "rb")) for x in glob.glob(smtglob)]
            self.name = smtglob
            ftcArgs = None
        else:
            xExtend = smtglob[1]
            yExtend = smtglob[2]
            smts = [(x[0], lambda f, z=x[1]: z.open(f, "r")) for x in smtglob[0]]
            ftcArgs = [min(xExtend), max(xExtend), self.tileW, min(yExtend), max(yExtend), self.tileH]
            self.name = None
        self.tiles = []
        log(INFO, "Number of tiles", len(smts))
        self.extends = {}
        #if len(smts) > 1000:
        #    smts = smts[:1000]
        for smt, openfct in smts:
            log(DEBUG, "Creating Tile for ", smt)
            code, zone, scale_man, scale_exp, x, y = list(map(int, 
                self.tilere.search(smt).groups()))
            if code in [326, 217]:
                scale = scale_man*(10**scale_exp)
                if not ftcArgs is None and ftc is None:
                    ftcArgs.extend([zone, scale, code])
                    ftc = FastTileCalculator(*ftcArgs)
                tile = Tile(self, smt, openfct, x, y, zone, scale, code, self.tileW, self.tileH, ftc)
                self.tiles.append(tile)
                if not tile.scale in self.extends:
                    self.extends[tile.scale] = dict(lat_min=tile.lat_min,  lat_max=tile.lat_max,  long_min=tile.long_min,  long_max=tile.long_max)
                else:
                    self.extends[tile.scale]['lat_min'] = min(self.extends[tile.scale]['lat_min'],  tile.lat_min)
                    self.extends[tile.scale]['lat_max'] = max(self.extends[tile.scale]['lat_max'],  tile.lat_max)
                    self.extends[tile.scale]['long_min'] = min(self.extends[tile.scale]['long_min'],  tile.long_min)
                    self.extends[tile.scale]['long_max'] = max(self.extends[tile.scale]['long_max'],  tile.long_max)
                progress.incValue(1)
            else:
                log(ERROR, "unknown code", code, "in smt file", smt)
        if self.name is None and len(self.tiles) > 0:
            zfname = smtglob[0][0][1].filename
            smtname = smtglob[0][0][0]
            self.name = os.path.split(zfname)[0] + "/%s@scale=%03d.%02d" % (
                    smtname[:4], 
                    int(self.tiles[0].scale), 
                    int(self.tiles[0].scale * 100) % 100)
        self.idx_sorted_lat_min = sorted(range(len(self.tiles)),  key=lambda x: self.tiles[x].lat_min)
        self.sorted_lat_min = [self.tiles[i].lat_min for i in self.idx_sorted_lat_min]
        self.idx_sorted_lat_max = sorted(range(len(self.tiles)),  key=lambda x: self.tiles[x].lat_max)
        self.sorted_lat_max = [self.tiles[i].lat_max for i in self.idx_sorted_lat_max]
        self.idx_sorted_long_min = sorted(range(len(self.tiles)),  key=lambda x: self.tiles[x].long_min)
        self.sorted_long_min = [self.tiles[i].long_min for i in self.idx_sorted_long_min]
        self.idx_sorted_long_max = sorted(range(len(self.tiles)),  key=lambda x: self.tiles[x].long_max)
        self.sorted_long_max = [self.tiles[i].long_max for i in self.idx_sorted_long_max]
        self.visited_tiles = set()
   
    def populate(self, scene):
        res = scene.createItemGroup(self.tiles)
        if len(self.tiles) > 0:
            res.setZValue(self.tiles[0].scale)
        return res

    def load_tile(self, smt, openfct):
        return tileCache.load_tile(smt, openfct)

    def find_tiles_colliding(self,  p1_lat,  p1_lon,  p2_lat,  p2_lon,  p3_lat,  p3_lon,  p4_lat, p4_lon,  extend = False,  debug=False):
        toRender = []
        def collide2d(x11,  x12,  y11,  y12,  x21,  x22,  y21,  y22):
            def collide1d(p11,  p12,  p21,  p22):
                if max(p11,  p12) < min(p21,  p22) or max(p21,  p22) < min(p11,  p12):
                    return False
                else:
                    return True
            if collide1d(x11,  x12,  x21,  x22) and collide1d(y11,  y12,  y21,  y22):
                return True
            else:
                return False
        minLat = min(p1_lat,  p2_lat,  p3_lat,  p4_lat)
        maxLat = max(p1_lat,  p2_lat,  p3_lat,  p4_lat)
        minLong = min(p1_lon,  p2_lon,  p3_lon,  p4_lon)
        maxLong = max(p1_lon,  p2_lon,  p3_lon,  p4_lon)
        if extend:
            # extend the region by 30% in each direction. Needed because of some black pieces in image
            extLat = (maxLat - minLat)*.3
            minLat -= extLat
            maxLat += extLat
            extLong = (maxLong - minLong)*.3
            minLong -= extLong
            maxLong += extLong
            if debug:
                print("Extending lat: %f long: %f" % (extLat,  extLong))

        i = bisect.bisect_right(self.sorted_lat_min,  maxLat)
        # everything right to this cannot collide
        idx1 = set(self.idx_sorted_lat_min[:i])
        
        i = bisect.bisect_left(self.sorted_lat_max,  minLat)
        # everything left to this cannot collide
        idx2 = set(self.idx_sorted_lat_max[i:])
        
        i = bisect.bisect_right(self.sorted_long_min,  maxLong)
        # everything right to this cannot collide
        idx3 = set(self.idx_sorted_long_min[:i])
        
        i = bisect.bisect_left(self.sorted_long_max,  minLong)
        # everything left to this cannot collide
        idx4 = set(self.idx_sorted_long_max[i:])
        
        idx = idx1.intersection(idx2).intersection(idx3).intersection(idx4)
        if debug:
            print("Number of tiles:",  len(idx))
        for i in idx:
            t = self.tiles[i]
            if collide2d(t.lat_min,  t.lat_max,  t.long_min,  t.long_max,  minLat,  maxLat,  minLong,  maxLong):
                toRender.append(t)
                self.visited_tiles.add(i)
            else:
                if debug:
                    print("Ignoring tile %d because not colliding." % i)
        return toRender
        

    def render_tile(self,  p1_lat,  p1_lon,  p2_lat,  p2_lon,  p3_lat,  p3_lon,  p4_lat, p4_lon,  targetSize,  debug=False):
        #res = Image.new("RGB",  (targetSize, targetSize),  "red")
        #return res
        
        toRender = self.find_tiles_colliding(p1_lat,  p1_lon,  p2_lat,  p2_lon,  p3_lat,  p3_lon,  p4_lat, p4_lon,  extend=True,  debug=debug)
        if len(toRender) == 0:
            return None
        assert len(toRender) <= 32
        #print ("num_tiles = ",  len(toRender))
        minx = min([tt.x_mc for tt in toRender]) 
        miny = min([tt.y_mc for tt in toRender])
        maxx = max([tt.x_mc+tt.w for tt in toRender])
        maxy = max([tt.y_mc+tt.h for tt in toRender])
        w = maxx - minx
        h = maxy - miny
        images = [self.load_tile(tt.smt,  tt.openfct) for tt in toRender]
        res = Image.new("RGB",  (w, h))
        for i in range(len(images)):
            res.paste(images[i], (toRender[i].x_mc - minx,   toRender[i].y_mc - miny))
        ul = mapCoord2deg(minx,  miny,  toRender[0].zone,  toRender[0].scale,  toRender[0].code)
        ur = mapCoord2deg(minx+w,  miny,  toRender[0].zone,  toRender[0].scale,  toRender[0].code)
        ll = mapCoord2deg(minx,  miny+h,  toRender[0].zone,  toRender[0].scale,  toRender[0].code)
        lr= mapCoord2deg(minx+w,  miny+h,  toRender[0].zone,  toRender[0].scale,  toRender[0].code)
        
        ul_lat,  ul_lon = ul[::-1]
        ur_lat,  ur_lon = ur[::-1]
        ll_lat,  ll_lon = ll[::-1]
        lr_lat,  lr_lon = lr[::-1]
        
        # P(ul_lat, ul_lon) = (0,0)
        # P(ur_lat, ur_lon) = (w,0)
        # P(ll_lat, ll_lon) = (0, h)
        # P(lr_lat, lr_lon) = (w, h)
        
        # transform DEG (lat, lon) -> Pimg: Pimg = S*(lat, lon)' + T
        # cx = s11*lat+s12*lon + t1
        # cy = s21*lat+s22_lon + t2
        # upper triangle:
        #           
        #   A * x = b,   x = (s11, s12, s21, s22, t1, t2)', b = (0, 0, w, 0, 0, h)'
        #           ul_lat, ul_lon, 0       , 0         , 1,     0
        #           0       , 0       , ul_lat , ul_lon , 0,     1
        #   A =  ur_lat, ur_lon, 0       , 0         , 1,     0
        #           0       , 0       , ur_lat , ur_lon , 0,     1
        #           ll_lat , ll_lon , 0       , 0         , 1,     0
        #           0       , 0       , ll_lat  , ll_lon  , 0,     1
        q = []
        A = numpy.array([[ul_lat,  ul_lon,  0, 0, 1, 0],  
                                    [0,  0,  ul_lat,  ul_lon,  0,  1], 
                                    [ur_lat,  ur_lon,  0,  0,  1, 0], 
                                    [0,  0,  ur_lat,  ur_lon,  0,  1], 
                                    [lr_lat,  lr_lon,  0,  0,  1,  0], 
                                    [0,  0,  lr_lat,  lr_lon,  0,  1], 
                                    [ll_lat,  ll_lon,  0,  0,  1,  0], 
                                    [0,  0,  ll_lat,  ll_lon,  0,  1]])
        b = numpy.array([0,  0,  w,  0,  w,  h,  0,  h])
        s11,  s12,  s21,  s22,  t1,  t2 = numpy.dot(linalg.pinv(A),  b)
        for lat, lon in [(p2_lat,  p2_lon),  (p3_lat,  p3_lon),  (p4_lat,  p4_lon),  (p1_lat,  p1_lon),  ]:
            # upper triangle
            q.extend(numpy.dot(numpy.array([[s11,  s12], [s21,  s22]]),  [lat,  lon]) + numpy.array([t1,  t2]))
            #print( "lat",  lat,  "long",  lon,  "->",  q[-2],  q[-1])
        if debug:
            print("ul", ul,  "ur", ur,  "ll", ll,  "lr", lr)
            print([str(x) for x in q])
        tile = res.transform((targetSize,  targetSize),  Image.QUAD,  q,  Image.BILINEAR).transpose(Image.ROTATE_180)
        if debug:
            res.show()
            tile.show()
            input("Press any key")
        
        return tile

    def progress(self):
        return len(self.visited_tiles)
