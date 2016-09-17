 # -*- coding: utf-8 -*-

# python imports
import os
import mmap
import io
import sys
import xml.dom.minidom
import zipfile
# local imports
import mapcoord
from logging import *

magicRouteFile = "S1nclair Rul3z!.txt"

class NoProgress:
    def __init__(self): pass
    def setMaximum(self, n, x=None): pass
    def incValue(self, n): pass

def show_smt(smt):
    s = open(smt, "rb").read()
    s = s[:1] + "PNG" + s[4:]
    open("/tmp/__smt_show.png", "wb").write(s)
    os.execvp("gwenview", ("gwenview","/tmp/__smt_show.png",))

def load_smt(smt, openfct, imageConstructor):
    s = openfct(smt).read()
    f = mmap.mmap(-1, len(s))
    f.write(s)
    f[1:4] = "PNG".encode("ascii")
    f.seek(0)
    return imageConstructor(f)

def decodeSpotXml(f):
    content = {}
    doc= xml.dom.minidom.parse(f)
    if doc.childNodes[0].nodeName != "MapKit":
        log(ERROR, "Cannot read spot xml file, unknown tag:", 
            doc.childNodes[0].nodeName)
        raise KeyError
    for a in doc.childNodes[0].attributes.values():
        content[str(a.nodeName)] = str(a.nodeValue)
    return (content["Name"], 
            float(content["InitNorth"])/float(1 << 31)*90.,
            float(content["InitEast"])/float(1 << 31)*180.)

class Route:
    def __init__(self, name, samples):
        self.name = name
        self.samples = samples
        
    def info(self):
        length = 0.0
        ascent = 0.0
        descent = 0.0
        for i in range(len(self.samples)-1):
            s1 = self.samples[i]
            s2 = self.samples[i+1]
            phi1 = s1['North']
            lmb1 = s1['East']
            phi2 = s2['North']
            lmb2 = s2['East']
            h1 = s1['Elev']
            h2 = s2['Elev']
            length += mapcoord.distInMeter(lmb1, phi1, lmb2, phi2)
            if h1 < h2:
                ascent += (h2 - h1)
            else:
                descent += (h1 - h2)
        return length, ascent, descent
        
    def remove(self, indices):
        sr = [self.samples[i] for i in indices]
        for s in sr:
            self.samples.remove(s)
            
    def append(self, north, east, elev):
        self.samples.append({
                'North':north,
                'East':east,
                'Elev':elev,
            })

def merge_routes(routes, name):
    def dist(routes):
        res = 0
        for i in range(len(routes)-1):
            r1 = routes[i+1]
            s1 = r1.samples[0]
            r2 = routes[i]
            s2 = r2.samples[-1]
            res += mapcoord.distInMeter(s1['East'], s1['North'], s2['East'], s2['North'])
        return res
    changed = True
    while changed:
        changed = False
        for i in range(len(routes)):
            lastD = dist(routes)
            routes[i].samples = routes[i].samples[::-1]
            newD = dist(routes)
            if newD < lastD:
                changed = True
            else:
                routes[i].samples = routes[i].samples[::-1]
    samples = []
    for r in routes:
        samples.extend(r.samples)
    return Route(name, samples)
            


def encodeRoute(route):
    impl = xml.dom.minidom.getDOMImplementation()
    doc = impl.createDocument(None, "Route", None)
    doc.childNodes[0].setAttribute('Name', route.name)
    for sample in route.samples:
        c = doc.createElement("Point")
        for k in sample.keys():
            v = sample[k]
            if k == 'East':
                v = unicode(int(v * (1 << 31) / 180. + 0.5))
            elif k == 'North':
                v = unicode(int(v * (1 << 31) / 90. + 0.5))
            elif k == 'Elev':
                v = unicode(int(v))
            c.setAttribute(k, v)
        doc.childNodes[0].appendChild(c)
    o_str = io.BytesIO()
    new_f = zipfile.ZipFile(o_str, "w")
    xml_o = io.StringIO(u"")
    doc.writexml(xml_o)
    new_f.writestr(magicRouteFile, xml_o.getvalue().encode('ascii', 'xmlcharrefreplace'))
    new_f.close()
    convertBuf = o_str.getvalue()
    return convertBuf, doc, xml_o

def saveRoute(route, filename):
    buf, doc, xml = encodeRoute(route)
    open(filename, "wb").write(buf)
    
def decodeRoute(f, expectedName):
    content = {}
    doc = xml.dom.minidom.parse(f)
    if doc.childNodes[0].nodeName != "Route":
        log(ERROR, "Cannot read route xml file, unknown tag:", 
            doc.childNodes[0].nodeName)
        raise KeyError
    for a in doc.childNodes[0].attributes.values():
        content[a.nodeName] = a.nodeValue
    route = []
    for c in doc.childNodes[0].childNodes:
        node = {}
        for a in c.attributes.values():
            n = a.nodeName
            v = a.nodeValue
            if n == 'East': 
                v = int(v)*180. / (1 << 31)
            elif n == 'North':
                v = int(v)*90. / (1 << 31)
            elif n == 'Elev':
                v = float(v)
            node[n] = v
        route.append(node)
    convertBuf = None
    routeRes = Route(expectedName, route)
    if content['Name'].replace(":", "") != expectedName:
        content['Name'] = expectedName
        encodeBuf, encodeDoc, encodeXml = encodeRoute(routeRes)
        convertBuf = encodeBuf
    return ({content['Name'] : routeRes}, convertBuf)

def openMapDir(dirName, progress = NoProgress()):
    def addDirToRes(res, d, prefix):
        prefix = os.path.normpath(prefix)
        d = os.path.normpath(d)
        assert d[:len(prefix)] == prefix, "%s %s" % (d, prefix)
        if not prefix in res:
            res[prefix] = {}
        res = res[prefix]
        d = d[len(prefix)+1:]
        while d.find("/") != -1:
            pos = d.find("/")
            p = d[:pos]
            d = d[pos+1:]
            if not p in res:
                res[p] = {}
            res = res[p]
        return res
    maps = []
    if os.path.isdir(dirName):
        for root, dirs, files in os.walk(dirName):
            for f in files:
                if f[-4:].lower() == ".map":
                    maps.append(root + "/" + f)
    else:
        maps.append(dirName)
    log(INFO, "Number of maps found:", len(maps))
    prefs = {}
    progress.setMaximum(len(maps)*10)
    spot = None
    spotRes = {}
    mapRes = {}
    routeRes = {}
    n = 0
    for m in maps:
        log(INFO, "Inspecting", m)
        zf = zipfile.ZipFile(m, "r", allowZip64=True)
        il = zf.infolist()
        if len(il) == 1 and il[0].filename[-len(magicRouteFile):] == magicRouteFile:
            route, convertBuf = decodeRoute(zf.open(il[0]), os.path.split(m)[1][:-4])
            res = addDirToRes(routeRes, m, dirName)
            res.update(route)
            log(INFO, dirName, "route added", route.keys()[0], route[route.keys()[0]].name)
            if not convertBuf is None:
                zf.close()
                try:
                    print ("Converting route", m)
                    open(m, "wb").write(convertBuf)
                except IOError as err:
                    print ("Error while converting route %s: " % m, str(err))
            continue
        for i in il:
            fn = i.filename
            if fn[-3:].lower() == "smt" and fn[4:7] in ["326", "217"]:
                n += 1
                p = (os.path.dirname(m), fn[:24])
                cy = int(fn[-4-7:-4])
                cx = int(fn[-4-7-7:-4-7])
                try:
                    prefs[p][0].append((fn, zf))
                    prefs[p][1][cx] = None
                    prefs[p][2][cy] = None
                except KeyError:
                    prefs[p] = [[(fn, zf)], {cx:None}, {cy:None}]
            if fn[-3:].lower() == "xml":
                spot = decodeSpotXml(zf.open(fn))
                res = addDirToRes(spotRes, m, dirName)
                res[spot[0]] = [spot[2], spot[1]]
        progress.incValue(1)
        log(INFO, "Number of smt files", n)
    progress.setMaximum(n + n/10, n/10)
    for k in prefs.keys():
        m = k[0] + "/DUMMY"
        p = k[1]
        res = addDirToRes(mapRes, m, dirName)
        if not p[:4] in res:
            res[p[:4]] = {}
        res[p[:4]][p[4:]] = [prefs[k], None, True] # tilecontainer, itemgroup, UserSelect
    return mapRes, spotRes, routeRes

def loadKml(filename):
    f = open(filename, "r")
    doc = xml.dom.minidom.parse(f)
    res = []
    for tl in doc.childNodes:
      assert tl.nodeName == "kml"
      for cn in tl.childNodes:
          print (cn.nodeName)
          if cn.nodeName == "Folder":
            for pm in cn.getElementsByTagName("Placemark"):
              for ls in pm.getElementsByTagName("LineString"):
                coords = ls.childNodes[0].childNodes[0]
                coordText = coords.nodeValue
                coords = coordText.split()
                samples = []
                for pair in coords:
                    e,n = pair.split(",")
                    samples.append({'East':float(e), 'North':float(n), 'Elev' : 0})
                res.append(Route(os.path.basename(filename), samples))
    return res

def loadCsv(filename):
    f = open(filename, "r")
    samples = []
    for line in f.xreadlines():
        x = line.split(",")
        n = x[0]
        e = x[1]
        samples.append({'East':float(e), 'North':float(n), 'Elev' : 0})
    return Route(os.path.basename(filename), samples)

def writeKml(filename, route):
    f = open(filename, "w")
    f.write("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://earth.google.com/kml/2.0">
<Folder><name>OpenLayers export</name>
<description>Exported from satmap format</description>
<Placemark><name>OpenLayers.Feature.Vector_5474</name>
<description>No description available</description>
<LineString><coordinates>
""")
    c = []
    for s in route.samples:
        c.append("%f,%f" % (s['East'], s['North']))
    f.write(" ".join(c))
    f.write("""
</coordinates></LineString></Placemark></Folder></kml>""")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print ("Usage: %s <infile> <outfile>" % sys.argv[0])
        sys.exit(1)
    infile = sys.argv[1]
    outfile = sys.argv[2]
    if infile[-3:] == "kml" and outfile[-3:] == "map":
        route = loadKml(infile)
        for i in range(len(route)):
            saveRoute(route[i], outfile%i)
    elif infile[-3:] == "map" and outfile[-3:] == "kml":
        mapRes, spotRes, routeRes = openMapDir(infile)
        print (routeRes)
        assert len(routeRes.keys()) == 1, routeRes
        route = routeRes[routeRes.keys()[0]]
        assert len(route.keys()) == 1, route
        route = route[route.keys()[0]]
        writeKml(outfile, route)
    elif infile[-3:] == "csv" and outfile[-3:] == "map":
        route = loadCsv(infile)
        saveRoute(route,outfile)
    elif infile[-3:] == "txt" and outfile[-3:] == "map":
        routes = []
        for f in open(infile, "r").readlines():
            f = f.strip()
            if f != "":
                mapRes, spotRes, routeRes = openMapDir(f)
                assert len(routeRes.keys()) == 1, routeRes
                route = routeRes[routeRes.keys()[0]]
                assert len(route.keys()) == 1, route
                routes.append(route[route.keys()[0]])
        route = merge_routes(routes, outfile)
        saveRoute(route,outfile)
    else:
        print ("Unsupported format")
        sys.exit(1)
