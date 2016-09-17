# ReadYourMAPFile
Convert Sat map A10's .MAP files to osmand compatible tile databases

The relevant scripts are 
- convert_map.py (for a description call with -h argument) to actually convert a directory containing .MAP files
- merge_sqlitedb.py to merge mutliple .sqlitedb files

The bundle uses python3, numpy and pillow.

These scripts are intended for people who want to use their legally buyed maps with open source software like osmand.
