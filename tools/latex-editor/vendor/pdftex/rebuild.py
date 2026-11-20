"""Rebuild the bundled TeX resources with Python 3. Source URLs in NOTICE.md."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import tarfile
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
ARCHIVE = 'https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2019/tlnet-final/archive/'
RELEASE = 'https://github.com/SwiftLaTeX/SwiftLaTeX/releases/download/v20022022/20-02-2022.zip'
FORMAT = 'https://raw.githubusercontent.com/SwiftLaTeX/Texlive-Ondemand/cc2146e220658f5ee0f8643355252ae0995fbc98/swiftlatexpdftex.fmt'


def fetch(url, algorithm, expected):
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    if hashlib.new(algorithm, data).hexdigest() != expected:
        raise ValueError('Source checksum mismatch for ' + url)
    return data


def main():
    release = fetch(RELEASE, 'sha256', 'cf5535ffe9cdf42f7c4522e96bba4d33db10c418a7ceb24e5d80504be7de7150')
    with zipfile.ZipFile(io.BytesIO(release)) as archive:
        for name in ['swiftlatexpdftex.js', 'swiftlatexpdftex.wasm']:
            (HERE / name).write_bytes(archive.read(name))

    files = {}
    for package in json.loads((HERE / 'packages.json').read_text()):
        blob = fetch(ARCHIVE + package['package'] + '.tar.xz', 'sha512', package['archive_sha512'])
        with tarfile.open(fileobj=io.BytesIO(blob), mode='r:xz') as archive:
            for path in package['paths']:
                files[Path(path).name] = archive.extractfile(path).read()
    files['swiftlatexpdftex.fmt'] = fetch(FORMAT, 'sha256', '3134ef8cbfa829780fc0b63b5320fb680e8f895f0c7f48ecb82af159098760ec')
    font_maps = ['cm.map', 'cmextra.map', 'symbols.map', 'euler.map', 'latxfont.map']
    files['pdftex.map'] = b'\n'.join(files[name] for name in font_maps if name in files)
    items = sorted(files.items())
    index = json.dumps([(name, len(content)) for name, content in items], separators=(',', ':')).encode()
    output = struct.pack('<I', len(index)) + index + b''.join(content for name, content in items)
    (HERE / 'texlive.bin').write_bytes(gzip.compress(output, compresslevel=9, mtime=0))


if __name__ == '__main__':
    main()
