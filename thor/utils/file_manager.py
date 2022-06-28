import io
import os
import yaml
import gzip
import shutil
import urllib
import requests
import datetime
import dateutil
import logging
from typing import (
    List,
    Optional
)
from astropy.time import Time

__all__ = [
    "FileManager",
]

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.expanduser("~"), ".thor")

class FileManager:
    """
    FileManager

    This class maintains a file log in the passed directory.

    """
    def __init__(self, directory: str = DATA_DIR):

        self._directory = os.path.abspath(directory)
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        self._log_file = os.path.join(self.directory, "data_log.yaml")

        if not os.path.exists(self._log_file):
            self._log = {}
            self.write_log()
        else:
            self.read_log()

        return

    @property
    def directory(self):
        return self._directory

    @property
    def log(self):
        if not isinstance(self._log, dict):
            self._log = {}
        return self._log

    @property
    def log_file(self):
        return self._log_file

    def read_log(self):
        """
        Read file log. The file log tracks all supplemental data
        files that different components of THOR may need. Examples
        include SPICE kernels, the MPC observatory code file, the
        MPC orbit catalog etc...

        Returns
        -------
        None
        """
        logger.debug(f"Reading log file ({self.log_file}).")
        with open(self.log_file) as file:
            self._log = yaml.load(file, Loader=yaml.FullLoader)
        return

    def write_log(self):
        """
        Write file log. The file log tracks all supplemental data
        files that different components of THOR may need. Examples
        include SPICE kernels, the MPC observatory code file, the
        MPC orbit catalog etc...

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        logger.debug(f"Writing log file ({self.log_file}).")
        with open(self.log_file, "w") as f:
            yaml.dump(self.log, f)
        return

    def check_update(self, url: str):
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
        last_modified : {None, `~astropy.core.time.Time`}
        """
        response = requests.head(url)
        last_modified = response.headers.get('Last-Modified')
        if last_modified:
            last_modified = Time(dateutil.parser.parse(last_modified))
        return last_modified

    def download(self, url: str, sub_directory: Optional[str] = None):
        """
        Download file at given url.

        Parameters
        ----------
        url : str
            URL of file.
        sub_directory : str, optional
            Download file to sub directory within the FileManager's main
            directory.

        """
        # Extract the file name (remove path) and
        # also grab the absolute path
        file_name = os.path.basename(urllib.parse.urlparse(url).path)

        # Add sub_directory if defined
        if sub_directory is not None:
            out_dir = os.path.join(self.directory, sub_directory)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
        else:
            out_dir = self.directory

        file_path = os.path.join(out_dir, file_name)
        logger.info("Checking {}.".format(file_name))

        # Check if the file is compressed with gzip
        compressed = False
        if os.path.splitext(file_name)[1] == ".gz":
            compressed = True

        # Has file been logged previously
        logged = False
        if file_name in self.log.keys():
            logged = True
        else:
            self.log[file_name] = {}

        download = False
        file_exists = os.path.isfile(file_path)
        if not file_exists:
            logger.debug("File has not been downloaded previously.")
            download = True

            # Reset the log if the file needs to be downloaded
            # but the file was previously logged
            if logged:
                self.log[file_name] = {}

        else:
            # Check when the file was last modified online
            last_modified = self.check_update(url)
            logger.debug("Last modified online: {}".format(last_modified.utc.isot))

            last_downloaded = Time(self.log[file_name]["downloaded"], format="isot", scale="utc")
            logger.debug("Last downloaded: {}".format(last_downloaded.utc.isot))
            if last_downloaded < last_modified:
                download = True

        if download:
            logger.debug("Downloading from {}...".format(url))

            response = urllib.request.urlopen(url)
            f = io.BytesIO(response.read())
            with open(file_path, 'wb') as f_out:
                f_out.write(f.read())

            logger.debug("Download complete.")

            # If the file is compressed with gzip, decompress it
            # and update the file path to reflect the change
            if compressed:
                logger.debug("Downloaded file is gzipped. Decompressing...")
                self.log[file_name]["compressed_location"] = file_path
                uncompressed_file_path = os.path.splitext(file_path)[0]
                with gzip.open(file_path, 'r') as f_in:
                    with open(uncompressed_file_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                file_path = uncompressed_file_path
                logger.debug("Decompression complete.")

            # Update log and save to file
            self.log[file_name]["url"] = url
            self.log[file_name]["location"] = file_path
            downloaded = Time(datetime.datetime.now(datetime.timezone.utc), scale="utc")
            self.log[file_name]["downloaded"] = downloaded.utc.isot

            self.write_log()
        else:
            logger.debug("No download needed.")

        return

    def remove_downloads(self, file_names: Optional[List] = None):
        """
        Remove downloaded files.

        Parameters
        ----------
        file_names : list, optional
            Names of files to remove (should be keys listed
            in the file log). Default behaviour is to remove
            all downloaded supplemental data files.

        Returns
        -------
        None
        """
        if file_names is None:
            file_names = list(self.log.keys())

        if len(self.log) == 0:
            logger.debug("No files to remove.")
        else:

            for file in file_names:

                logger.debug("Removing {} ({}).".format(file, self.log[file]["location"]))
                os.remove(self.log[file]["location"])

                if "compressed_location" in self.log[file].keys():
                    logger.debug("Removing compressed {} ({}).".format(file, self.log[file]["compressed_location"]))
                    os.remove(self.log[file]["compressed_location"])

                del self.log[file]

            self.write_log()
        return

