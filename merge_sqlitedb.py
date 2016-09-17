import sys
import sqlite3
import os

if len(sys.argv) < 2:
    print("Usage: merge_sqlitedb.py <target> <source>")
    os._exit(1)

target,  source = sys.argv[1:]

tgt = sqlite3.connect(target)
src = sqlite3.connect(source)

tgtc = tgt.cursor()
srcc = src.cursor()

with tgt:
    srcc.execute("SELECT COUNT(*) FROM tiles")
    cntsrc = srcc.fetchone()[0]
    srcc.execute("SELECT z, x, y, s, image FROM tiles")
    tgtc.execute("SELECT z, x, y, s FROM tiles") 
    tgtcontent = set([tuple(x) for x in tgtc.fetchall()])
    print("%d datasets already in target."%  len(tgtcontent))

    cntins = 0
    for d in srcc:
        cntins += 1
        if cntins % 100 == 0:
            print("\rProgress: %5.1f %%" % (cntins*100./cntsrc),  end="")
        if not tuple(d[:-1]) in tgtcontent:
            tgtc.execute("""
                INSERT INTO tiles(z,x,y,s,image) 
                VALUES(?, ?, ?, ?, ?)
                """,  d)

    tgtc.execute("UPDATE info SET minzoom=(SELECT MIN(z) FROM tiles), maxzoom=(SELECT MAX(z) FROM tiles)")
    tgtc.execute("SELECT z, x, y, s FROM tiles")
    tgtcontent_new = set(tgtc.fetchall())
    print("%d datasets now in target."% len(tgtcontent_new))
