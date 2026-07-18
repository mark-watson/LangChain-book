"""Fetch .txt files from Google Drive using PyDrive2.

PyDrive2 is the maintained successor to PyDrive. Requires Google OAuth
credentials set up per the PyDrive2 documentation.
"""

import sys
from pathlib import Path

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

_SETUP_INSTRUCTIONS = """
Google OAuth credentials not found (client_secrets.json is missing).

To set up access to Google Drive:

  1. Go to the Google Cloud Console and create a project:
     https://console.cloud.google.com/

  2. Enable the Google Drive API for your project:
     https://console.cloud.google.com/apis/library/drive.googleapis.com

  3. Create OAuth 2.0 credentials (Desktop app type) and download
     the JSON file, saving it as 'client_secrets.json' in this directory.

  4. Full quickstart guide:
     https://developers.google.com/drive/api/quickstart/python

Then re-run this script — a browser window will open to complete sign-in.
"""


def get_txt_files(drive, dir_id="root"):
    """Get all plain text files with .txt extension in a Google Drive directory."""

    file_list = drive.ListFile(
        {"q": f"'{dir_id}' in parents and trashed=false"}
    ).GetList()
    for file1 in file_list:
        print("title: %s, id: %s" % (file1["title"], file1["id"]))
    return [
        [file1["title"], file1["id"], file1.GetContentString()]
        for file1 in file_list
        if file1["title"].endswith(".txt")
    ]


def test():
    if not Path("client_secrets.json").exists():
        print(_SETUP_INSTRUCTIONS)
        sys.exit(0)

    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    fl = get_txt_files(drive)
    for f in fl:
        print(f)
        with open("data/" + f[0], "w") as fh:
            fh.write(f[2])


if __name__ == "__main__":
    test()
