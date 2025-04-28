"""
Upload a tiff file to a non-temporal Lizard raster.

If no path is supplied, print the number of uncommitted uploads.

Progress and debug ouutput goes to stderr.
"""
from argparse import ArgumentParser
from http import HTTPStatus
from os import environ
from pathlib import Path
from sys import stderr
from time import sleep
from uuid import UUID

from requests import get, post

URL = "https://nens.lizard.net/api/v4/rastersources/{uuid}"


def upload(uuid, path, name):
    # get API Key
    api_key = environ.get("API_KEY")
    if api_key is None:
        print("API_KEY is not defined.", file=stderr)
        return
    auth = ("__key__", api_key)

    # get raster
    rastersource_url = URL.format(uuid=uuid)
    get_response = get(url=rastersource_url, auth=auth)
    get_result = get_response.json()
    if not get_response.status_code == HTTPStatus.OK:
        print(get_result["detail"], file=stderr)

    # verify name
    if name is not None:
        remote_name = get_result["name"]
        if remote_name != name:
            print(f"Name does not match: '{remote_name}'", file=stderr)
            return

    # determine raster service busyness
    rastersource_uploads_url = f"{rastersource_url}/uploads/"
    uploads_data = get(url=rastersource_uploads_url, auth=auth).json()
    uncommitted = sum(
        e["state"] != "committed"
        for e in uploads_data["results"]
    )
    print(f"Uncommitted uploads for raster: {uncommitted}", file=stderr)
    if path is None:
        return

    # wait for some time if there are to many uncommitted uploads
    if uncommitted > 5:
        minutes = uncommitted - 5
        print(f"Waiting {minutes} minutes.", file=stderr, end="", flush=True)
        for i in range(minutes):
            for i in range(6):
                sleep(10)
                print(".", file=stderr, end="", flush=True)
        print(file=stderr)

    # upload
    print(f"Uploading '{path}'...", file=stderr, end="", flush=True)
    rastersource_data_url = f"{rastersource_url}/data/"
    post_response = post(
        url=rastersource_data_url,
        auth=auth,
        files={"file": path.open("rb")},
    )
    print("done.", file=stderr)
    post_result = post_response.json()

    # poll task status
    task_url = f"{post_result['url']}/"
    print("Polling task status", file=stderr, end="", flush=True)
    for t in (1, 2, 3, 5, 8):
        sleep(t)
        print(".", file=stderr, end="", flush=True)
        if get(url=task_url).json()["status"] == "SUCCESS":
            print(file=stderr)
            print(f"{path} imported.")
            print(f"{path} imported.", file=stderr)
            break
    else:
        print(f"{path} failed to import.")
        print(f"{path} failed to import.", file=stderr)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("uuid", type=UUID, help="raster UUID", metavar="UUID")
    parser.add_argument("--name", help="Name to verify correct raster")
    parser.add_argument(
        "path", nargs="?", type=Path, help="Path to tif", metavar="PATH",
    )
    args = parser.parse_args()
    upload(uuid=args.uuid, path=args.path, name=args.name)
