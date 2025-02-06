from zipfile import ZipFile

def unzipFile(filepath,dest_dir):
    with ZipFile(filepath,'r') as zf:
        zf.extractall(path=dest_dir)