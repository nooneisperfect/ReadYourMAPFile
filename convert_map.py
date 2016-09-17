import TileHandling
import SatmapFormat
import sys
import math
import osm_adaptor
import io
import re
import os
import os.path
import datetime
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def help():
    print("Usage: %s [-maxscale=<ms>] [-poi=<lat>,<long>] [-ignorescales=<sc1>,<sc2>,<sc3>,...] [-d] [-list_contents] [-select_content=<content>] <MAP_DIRECTORY> <TARGET_DIR>" % os.path.basename(sys.argv[0]))
    os._exit(1)
    
if __name__ == "__main__":
    maxscale = None
    ignorescales = set()
    poi = None
    debug = False
    list_contents = False
    content = None
    for a in sys.argv[1:]:
        M = re.match('-maxscale=([0-9]+)',  a)
        if not M is None:
            sys.argv.remove(a)
            maxscale = int(M.group(1))
            continue
        M = re.match('-poi=([0-9.]+),([0-9.]+)',  a)
        if not M is None:
            sys.argv.remove(a)
            poi = (float(M.group(1)),  float(M.group(2)))
            continue
        M = re.match('-ignorescales=([0-9,]+)',  a)
        if not M is None:
            sys.argv.remove(a)
            ignorescales = set([int(s) for s in M.group(1).split(',')])
        if a == "-d":
            debug = True
            sys.argv.remove(a)
        if a == "-list_contents":
            list_contents = True
            sys.argv.remove(a)
        M = re.match('-select_content=(.+)',  a)
        if not M is None:
            sys.argv.remove(a)
            content = M.group(1)
        if a in ['-h',  '--help',  '/?']:
            help()
    if debug and poi is None:
        print("Debugging is only possible with -poi. Setting debug=False.")
        debug = False
    if len(sys.argv) < 2 or (not list_contents and len(sys.argv) < 3):
        help()
           
    progress = SatmapFormat.NoProgress()
    mapRes, spotRes, routeRes = SatmapFormat.openMapDir(sys.argv[1], progress)
            
    def dictWalk(map, f, self):
        if not type(map) == type({}):
            f(map, self)
        else:
            for k in map.keys():
                dictWalk(map[k], f, self)
                
    def populateMapDisplay(m, self):
        assert len(m) == 3 and m[1] is None
        tc = TileHandling.TileContainer(m[0], progress)
        ig = self.append(tc)
        m[0] = tc
        m[1] = ig
    
    maps = []
    dictWalk(mapRes, populateMapDisplay, maps)
        
    maps.sort(key=lambda m: m.tiles[0].scale)

    if list_contents:
        print("Contents of map:")
        contents = {}
        for m in maps:
            name = os.path.basename(m.name).split('@')[0]
            if not name in contents:
                contents[name] = dict(extends=m.extends[m.tiles[0].scale],  scales=[])
            contents[name]['scales'].append(m.tiles[0].scale)
            contents[name]['extends']['lat_min'] = min(contents[name]['extends']['lat_min'],  m.extends[m.tiles[0].scale]['lat_min'])
            contents[name]['extends']['lat_max'] = max(contents[name]['extends']['lat_max'],  m.extends[m.tiles[0].scale]['lat_max'])
            contents[name]['extends']['long_min'] = min(contents[name]['extends']['long_min'],  m.extends[m.tiles[0].scale]['long_min'])
            contents[name]['extends']['long_max'] = max(contents[name]['extends']['long_max'],  m.extends[m.tiles[0].scale]['long_max'])
        for name in sorted(contents.keys()):
            print("  Name:",  name,  
                     "; Extend: (%.3f, %.3f) - (%.3f, %.3f)" %  (contents[name]['extends']['lat_min'],  contents[name]['extends']['long_min'] , 
                                                                                                  contents[name]['extends']['lat_max'] ,  contents[name]['extends']['long_max']), 
                    "; Scales:",  sorted(contents[name]['scales']))
        print("Scales 50 - 1:200000,  100 - 1:100000,  200 - 1:50000,  400 - 1:25000,  800 - 1:12500")
        os._exit(0)
   
    import pprint
    pprint.pprint([(m.extends,  m.name) for m in maps])

    targetDir = sys.argv[2]
    
    if not maxscale is None or not len(ignorescales) == 0:
        if maxscale is None:
            maxscale = max([mm.tiles[0].scale for mm in maps])
        nmaps = list(filter(lambda mm: mm.tiles[0].scale <= maxscale and not mm.tiles[0].scale in ignorescales,  maps))
        if len(nmaps) < len(maps):
            print("Filtered %d/%d maps maxscale=%d" % (len(maps) - len(nmaps),  len(maps),  maxscale) )
            maps = nmaps
        else:
            print("-maxscale or -ignorescale given but did not match maps. Scales of maps: ",  list([mm.tiles[0].scale for mm in maps]))
            sys.exit(1)
    
    if not content is None:
        nmaps = list(filter(lambda mm: os.path.basename(mm.name).split('@')[0] == content,  maps))
        if len(nmaps) < len(maps) and len(nmaps) > 0:
            print("filtered %d/%d maps content=%s" % (len(maps)-len(nmaps),  len(maps),  content))
            maps = nmaps
        else:
            print("-select_content filter wrong.")
    
    start = datetime.datetime.now()
    
    transform = osm_adaptor.GlobalMercator()
    # get reasonable zoom levels
    # for that, use the center of the map
    dbres = []
    with ThreadPoolExecutor(max_workers=1) as dbthread:
        created_dbs = {}
        for m in maps:
            dbname = m.name.replace('/',  '_')
            if '@' in dbname:
                dbname = dbname[:dbname.find('@')]
            if dbname in created_dbs:
                db = created_dbs[dbname]
            else:            
                db = dbthread.submit(osm_adaptor.SqliteTileStorage, 'TMS').result()
                dbthread.submit(db.create, targetDir + '/map_%s.sqlitedb' % dbname,  True).result()
                created_dbs[dbname] = db
            print ("\n\nConverting",  m.name,  "to",  dbname)
            assert len(m.extends.keys()) == 1
            scale = list(m.extends.keys())[0]
            latc = (m.extends[scale]['lat_min'] + m.extends[scale]['lat_max'])*0.5
            longc = (m.extends[scale]['long_min'] + m.extends[scale]['long_max'])*0.5
            
            tileinfo = set([(tt.zone,  tt.scale,  tt.code) for tt in m.tiles])
            assert (len(tileinfo) == 1)
            ii = m.tiles[0].load()
            assert(ii.size == (m.tiles[0].w,  m.tiles[0].h))
            
            #print("zone",  m.tiles[0].zone,  "scale",  m.tiles[0].scale,  "code",  m.tiles[0].code)
            # find tile containing center and adapt the longc, latc accordingly
            center_tiles = m.find_tiles_colliding(latc,  longc,  latc,  longc,  latc,  longc,  latc,  longc)
            assert(len(center_tiles) >= 1)
            ct = center_tiles[0]
            longc = ct.long_min
            latc = ct.lat_min
            smx = ct.x_mc
            smy = ct.y_mc
            smx2 = smx + ct.w
            smy2 = smy + ct.h
            long2 = ct.long_max
            lat2 = ct.lat_max
            
            #smx,  smy = TileHandling.deg2mapCoord(longc,  latc,  m.tiles[0].zone,  m.tiles[0].scale,  m.tiles[0].code)
            #smx2 = smx + m.tiles[0].w
            #smy2 = smy + m.tiles[0].h
            #long2,  lat2 = TileHandling.mapCoord2deg(smx2,  smy2,  m.tiles[0].zone,  m.tiles[0].scale,  m.tiles[0].code)
            sm_density = abs((smx2 - smx)*(smy2-smy))/abs((long2-longc)*(lat2-latc))
            bestZoom = None
            bestDensityDelta = 0
            for zoom in range(osm_adaptor.MAXZOOMLEVEL):
                mx,  my = transform.LatLonToMeters(latc,  longc)
                px,  py = transform.MetersToPixels(mx,  my,  zoom)
                px2 = px + transform.tileSize
                py2 = py + transform.tileSize
                mx,  my = transform.PixelsToMeters(px2,  py2,  zoom)
                lat3,  long3 = transform.MetersToLatLon(mx,  my)
                osm_density = abs((px2-px)*(py2-py))/abs((lat3-latc)*(long3-longc))
                #print ("zoom:",  zoom,  math.sqrt(osm_density/sm_density), "OSM:",  osm_density,  (px2-px), (py2-py),  (lat3-latc),  (long3-longc),  "SM:", sm_density,  (smx2 - smx),  (smy2-smy), (lat2-latc),  (long2-longc)   )
                if osm_density > sm_density and (bestZoom is None or abs(osm_density - sm_density) < bestDensityDelta):
                    bestZoom = zoom
                    bestDensityDelta = abs(osm_density - sm_density)
                    bestOsmDensity = osm_density
            print ("bestZoom:",  bestZoom,  math.sqrt(bestOsmDensity/sm_density), "OSM:",  bestOsmDensity,  (px2-px), (py2-py),  (lat3-latc),  (long3-longc),  "SM:", sm_density,  (smx2 - smx),  (smy2-smy), (lat2-latc),  (long2-longc)   )
            # now we are going to render new tiles with this zoom factor
            
            lat1 = m.extends[scale]['lat_min']
            lat2 = m.extends[scale]['lat_max']
            long1 = m.extends[scale]['long_min']
            long2 = m.extends[scale]['long_max']
            
            if not poi is None:
                long1 = poi[1]-0.005
                long2 = poi[1]+0.005
                lat1 = poi[0]-0.005
                lat2 = poi[1]+0.005
                workers = 1
            else:
                workers = 12
            
            mx,  my  = transform.LatLonToMeters(lat1,  long1)
            px,  py = transform.MetersToPixels(mx,  my,  bestZoom)
            tx1,  ty1 = transform.PixelsToTile(px,  py)
            tx = tx1
            ty = ty1
            n = 0
            with ThreadPoolExecutor(max_workers=workers) as executor:
                exres = []
                while 1:
                    tx = tx1
                    yCovered = False
                    while 1:
                        def work(tx,  ty,  bestZoom):
                            p1_lat,  p1_lon,  p2_lat,  p2_lon,  p3_lat,  p3_lon,  p4_lat, p4_lon = transform.TileLatLonCorners(tx,  ty,  bestZoom)
                            tile = m.render_tile(p1_lat,  p1_lon,  p2_lat,  p2_lon,  p3_lat,  p3_lon,  p4_lat, p4_lon,  transform.tileSize,  debug)
                            if not tile is None:
                                f = io.BytesIO(b'')
                                tile = tile.convert('P', colors= 256, palette=Image.ADAPTIVE)
                                tile.save(f,  'PNG',  optimize = True)
                                tile = f.getvalue()
                            return p1_lat,  p1_lon,  p2_lat,  p2_lon,  p3_lat,  p3_lon,  p4_lat, p4_lon,  tile,  tx,  ty,  bestZoom
                        exres.append(executor.submit(work,  tx,  ty,  bestZoom))
                        stopit = False
                        while len(exres) > workers or (stopit and len(exres) > 0):
                            p1_lat,  p1_lon,  p2_lat,  p2_lon,  p3_lat,  p3_lon,  p4_lat, p4_lon,  tile,  ctx,  cty,  cbestZoom = exres[0].result()
                            exres = exres[1:]
                            if not tile is None:
                                #print("%d %d minLat=%.5f maxLat=%.5f minLong=%.5f maxLong=%.5f" %( tx,  ty,  p1_lat,  p3_lat,  p1_lon,  p3_lon))
                                dbres.append(dbthread.submit(db.writeImage,  ctx, cty, cbestZoom, tile))
                                if len(dbres) > 10:
                                    while len(dbres) > 3:
                                        dbres[0].result()
                                        dbres = dbres[1:]
                            maxLat = max(p1_lat, p2_lat,  p3_lat,  p4_lat)
                            maxLong = max(p1_lon,  p2_lon,  p3_lon,  p4_lon)
                            if maxLat <= lat2:
                                yCovered = True
                            if maxLong > long2:
                                stopit = True
                        if stopit:
                            break
                        tx += 1
                        n += 1
                        
                        if n == 200:
                            wRatio = float(sum([mm.progress() for mm in maps]))/sum([len(mm.tiles) for mm in maps])
                            if wRatio <= 0.0:
                                wRatio = 0.000001
                            curr = datetime.datetime.now()
                            elapsed = curr - start
                            total = elapsed / wRatio
                            remaining = total - elapsed
                            print("\rProgress: %10.1f %% ETA: %s                                 " % (wRatio*100,  str(remaining)),  end="")
                            n = 0
                    if not yCovered:
                        break
                    ty -= 1
        for dbname in created_dbs:
            dbthread.submit(created_dbs[dbname].close).result()
        print("\n\nDone")
    
