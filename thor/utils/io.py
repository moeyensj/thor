import datetime
import gzip
import io
import logging
import os
import shutil
import urllib

import dateutil
import requests
import yaml
from astropy.time import Time

logger = logging.getLogger(__name__)

__all__ = [
    "_readFileLog",
    "_writeFileLog",
    "_checkUpdate",
    "_downloadFile",
    "_removeDownloadedFiles",
]


def _readFileLog(log_file=None):
    """
    Read THOR file log. The file log tracks all supplemental data
    files that different components of THOR may need. Examples
    include SPICE kernels, the MPC observatory code file, the
    MPC orbit catalog etc...

    Parameters
    ----------
    log_file : str, optional
        Path to log file

    Returns
    -------
    None
    """
    if log_file is None:
        log_file = os.path.join(os.path.dirname(__file__), "..", "data/log.yaml")
    if not os.path.isfile(log_file):
        with open(log_file, "w") as f:
            yaml.dump({}, f)
    with open(log_file) as file:
        log = yaml.load(file, Loader=yaml.FullLoader)
    return log


def _writeFileLog(log, log_file=None):
    """
    Write THOR file log. The file log tracks all supplemental data
    files that different components of THOR may need. Examples
    include SPICE kernels, the MPC observatory code file, the
    MPC orbit catalog etc...

    Parameters
    ----------
    log : dict
        Dictionary with file names as keys and dictionaries of properties
        as values.
    log_file : str, optional
        Path to log file

    Returns
    -------
    None
    """
    if log_file is None:
        log_file = os.path.join(os.path.dirname(__file__), "..", "data/log.yaml")
    with open(log_file, "w") as f:
        yaml.dump(log, f)
    return


def _checkUpdate(url):
    """
    Query url for "Last-modified" argument in header. If "Last-modified" argument
    exists in the header this function will return the time, if it does
    not it will return None.

    Parameters
    ----------
    url : str
        URL to query for last modification.

    Returns
    -------
    {None, `~astropy.core.time.Time`}

    """
    response = requests.head(url)
    last_modified = response.headers.get("Last-Modified")
    if last_modified:
        last_modified = Time(dateutil.parser.parse(last_modified))
    return last_modified


def _downloadFile(to_dir, url, log_file=None):
    """
    Download file at given url to the passed directory.

    Parameters
    ----------
    to_dir : str
        Path to directory where file should be downloaded to.
    url : str
        URL of file.
    log_file : str, optional
        Path to THOR file log. The file log tracks all supplemental data
        files that different components of THOR may need. Examples
        include SPICE kernels, the MPC observatory code file, the
        MPC orbit catalog etc...

    Returns
    -------
    None
    """
    # Extract the file name (remove path) and
    # also grab the absolute path
    file_name = os.path.basename(urllib.parse.urlparse(url).path)
    file_path = os.path.join(os.path.abspath(to_dir), file_name)
    logger.info("Checking {}.".format(file_name))

    # Check if the file is compressed with gzip
    compressed = False
    if os.path.splitext(file_name)[1] == ".gz":
        compressed = True

    # Read file log
    if log_file is None:
        log_file = os.path.join(os.path.dirname(__file__), "..", "data/log.yaml")
    log = _readFileLog(log_file)

    # Has file been logged previously
    logged = False
    if file_name in log.keys():
        logged = True
    else:
        log[file_name] = {}

    download = False
    file_exists = os.path.isfile(file_path)
    if not file_exists:
        logger.info("File has not been downloaded previously.")
        download = True

        # Reset the log if the file needs to be downloaded
        # but the file was previously logged
        if logged:
            log[file_name] = {}

    else:
        # Check when the file was last modified online
        last_modified = _checkUpdate(url)
        logger.info("Last modified online: {}".format(last_modified.utc.isot))

        last_downloaded = Time(log[file_name]["downloaded"], format="isot", scale="utc")
        logger.info("Last downloaded: {}".format(last_downloaded.utc.isot))
        if last_downloaded < last_modified:
            download = True

    if download:
        logger.info("Downloading from {}...".format(url))

        response = urllib.request.urlopen(url)
        f = io.BytesIO(response.read())
        with open(file_path, "wb") as f_out:
            f_out.write(f.read())

        logger.info("Download complete.")

        # If the file is compressed with gzip, decompress it
        # and update the file path to reflect the change
        if compressed:
            logger.info("Downloaded file is gzipped. Decompressing...")
            log[file_name]["compressed_location"] = file_path
            uncompressed_file_path = os.path.splitext(file_path)[0]
            with gzip.open(file_path, "r") as f_in:
                with open(uncompressed_file_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            file_path = uncompressed_file_path
            logger.info("Decompression complete.")

        # Update log and save to file
        log[file_name]["url"] = url
        log[file_name]["location"] = file_path
        downloaded = Time(datetime.datetime.now(datetime.timezone.utc), scale="utc")
        log[file_name]["downloaded"] = downloaded.utc.isot

        _writeFileLog(log, log_file)
        logger.info("Log updated.")

    else:
        logger.info("No download needed.")

    return


def _removeDownloadedFiles(file_names=None):
    """
    Remove downloaded files.

    Parameters
    ----------
    file_names : list, optional
        Names of files to remove (should be keys listed
        in the THOR file log). Default behaviour is to remove
        all downloaded supplemental data files.

    Returns
    -------
    None
    """
    log_file = os.path.join(os.path.dirname(__file__), "..", "data/log.yaml")
    log = _readFileLog(log_file=log_file)
    if file_names is None:
        file_names = list(log.keys())

    if len(log) == 0:
        logger.info("No files to remove.")
    else:

        for file in file_names:

            logger.info("Removing {} ({}).".format(file, log[file]["location"]))
            os.remove(log[file]["location"])
            if "compressed_location" in log[file].keys():
                logger.info(
                    "Removing compressed {} ({}).".format(
                        file, log[file]["compressed_location"]
                    )
                )
                os.remove(log[file]["compressed_location"])

            del log[file]

        _writeFileLog(log, log_file=log_file)
        logger.info("Log updated.")
    return
