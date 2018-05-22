import argparse
import requests
import shutil
import pathlib

# Script command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("address", help="URL to test to download", nargs='?')
args = parser.parse_args()

# Pre-defined sources
sources = {
    'Network Repository': {
        'urlfile': 'networkrepository-urls.txt',
        'folder': 'networkrepository/in/'
    }
}

# Proper part
def download_one(url, folder):
    url = url.rstrip()
    print('Downloading from... {0} '.format(url), end='')
    
    local_filename = url[url.rfind('/')+1:] # TODO
    
    with requests.post(url, stream=True, allow_redirects=True) as r:
        local_filepath = folder + local_filename
        with open(local_filepath, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    print('to file ({0}) DONE.'.format(local_filename))
    return

def download(source_name):
    print('---- {0} ----'.format(source_name))
    # Extract source info
    (urlfile, folder) = (sources[source_name]['urlfile'], sources[source_name]['folder'])
    
    # mkdir for tests
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True) 

    # dl tests
    with open(urlfile) as file:
        for line in file:
            if not line.startswith('#'):
                download_one(line, folder)
    return

if __name__ == "__main__":
    if args.address is not None:
        # URL is specified
        download_one(args.address, '')
    else:
        # Use pre-defined sources
        for source in sources:
            download(source)
