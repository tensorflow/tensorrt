"""C4 dataset based on Common Crawl."""

import os
import gzip

import requests
from tqdm import tqdm

BASE_DOWNLOAD_PATH = "/tmp"

_VARIANTS = ["en", "realnewslike", "en.noblocklist", "en.noclean"]

_N_SHARDS_PER_SPLIT = {
    "en": {"train": 1024, "validation": 8},
    "realnewslike": {"train": 512, "validation": 1},
    "en.noblocklist": {"train": 1024, "validation": 8},
    "en.noclean": {"train": 7168, "validation": 64},
}

# _DATA_URL = "https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/{name}/c4-{split}.{index:05d}-of-{n_shards:05d}.json.gz"
_DATA_URL = "https://huggingface.co/datasets/allenai/c4/resolve/607bd4c8450a42878aa9ddc051a65a055450ef87/{name}/c4-{split}.{index:05d}-of-{n_shards:05d}.json.gz"


def download(url: str, fname: str):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    # Can also replace 'file' with a io.BytesIO object
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def decompress(infile, tofile):
    with open(infile, 'rb') as inf, open(tofile, 'w', encoding='utf8') as tof:
        decom_str = gzip.decompress(inf.read()).decode('utf-8')
        tof.write(decom_str)


if __name__ == "__main__":
    for variant in _VARIANTS:
        print('\n=============================================================')
        print(f'Processing Variant: {variant}')

        variant_dir = os.path.join(BASE_DOWNLOAD_PATH, variant)

        try:
            os.makedirs(variant_dir)
        except FileExistsError: pass

        for split in ["train", "validation"]:
            if split == "train":
                continue

            num_shards = _N_SHARDS_PER_SPLIT[variant][split]

            print(f"Split: {split}, Shards: {num_shards}")

            for index in range(num_shards):
                url = _DATA_URL.format(
                    name=variant,
                    split=split,
                    index=index,
                    n_shards=num_shards
                )

                filename = os.path.join(variant_dir, url.split("/")[-1])

                # Downloading the file in GZIP format

                if not os.path.isfile(filename):
                    print(f"Downloading: {url}...")
                    download(url, fname=filename)
                else:
                    print(f"Already exists: {filename}...")

                # Processing the file from GZIP to JSON

                target_file = filename.replace(".gz", "")

                if not os.path.isfile(target_file):
                    print(f"Decompressing: {filename}...")
                    decompress(filename, target_file)
                else:
                    print(f"Decompressed file already exists: {target_file}")
