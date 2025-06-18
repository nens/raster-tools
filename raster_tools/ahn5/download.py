# -*- coding: utf-8 -*-
"""
Download ahn5 files, using curl and the json file from the online viewer.
"""
from argparse import ArgumentParser
from json import load
from pathlib import Path
from shlex import split
from subprocess import call

KEY = {
    "dsm": "AHN5 DSM ½m",
    "dtm": "AHN5 maaiveldmodel (DTM) ½m",
}


def get_urls(json_path, dsm_or_dtm):
    data = load(json_path.open())
    key = KEY[dsm_or_dtm]
    urls = (
        feature["properties"][key]
        for feature in data["result"]["features"]
    )
    for url in urls:
        if url == "None":
            continue
        yield url


def download(json_path, target_dir, dsm_or_dtm):
    target_dir.mkdir(exist_ok=True)
    urls = list(get_urls(json_path=json_path, dsm_or_dtm=dsm_or_dtm))
    total = len(urls)
    processed, downloaded, notfound, skipped, failed = 0, 0, 0, 0, 0
    for url in urls:
        name = url.rsplit("/", 1)[1].lower()
        path = target_dir / name
        curl = f"curl --location --fail --output {path} --retry 3 --max-time 1800 {url}"
        if path.exists():
            skipped += 1
        else:
            print(curl)
            status = call(split(curl))
            if status == 22:
                notfound += 1
            elif status:
                failed += 1
                if path.exists():
                    path.unlink()
            else:
                downloaded += 1
        processed += 1

        # console logging
        template = ("{progress:.1%}: {processed} processed, "
                    "{downloaded} downloaded, {skipped} skipped, "
                    "{notfound} notfound, {failed} failed.")

        print(template.format(
            progress=processed / total,
            processed=processed,
            downloaded=downloaded,
            skipped=skipped,
            notfound=notfound,
            failed=failed,
        ))
        break


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("dsm_or_dtm", choices=["dsm", "dtm"])
    parser.add_argument("json_path", metavar="INFOJSON", type=Path)
    parser.add_argument("target_dir", metavar="TARGET", type=Path)
    kwargs = vars(parser.parse_args())
    download(**kwargs)
