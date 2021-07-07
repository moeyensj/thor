from typing import Any
import json
import logging
import requests
import time

import googleapiclient.discovery


logger = logging.getLogger("thor")


def terminate_self():
    """
    Terminate the Google Compute Engine VM that is running the current process.

    This will trigger shutdown of the current host, if it is on GCE..
    """

    logger.info("terminating running Google Compute Engine instance")
    logger.info("trying to infer the identity of the instance...")
    name = discover_instance_name()
    zone_url = discover_instance_zone()
    zone = zone_url.split("/")[-1]
    project = discover_project_id()
    logger.info(
        "identity inference done. name=%s  zone=%s  project=%s", name, zone, project
    )

    logger.info("sending DELETE request to terminate instance")
    compute_client = googleapiclient.discovery.build("compute", "v1")
    operation = compute_client.instances().delete(
        project=project, zone=zone, instance=name,
    )

    logger.info("Blocking to wait for the delete to complete")
    compute_client.zoneOperations().wait(
        project=project, zone=zone, operation=operation["name"]
    )


def _google_metadata_request(path: str) -> Any:
    retry_limit = 30
    retry_count = 0
    while retry_count < retry_limit:
        response = requests.get("http://metadata.google.internal" + path, timeout=1.0)
        if response.status_code == 503:
            # Indicates metadata server maintenance. Retry.
            time.sleep(1)
            retry_count += 1
        else:
            break
    response.raise_for_status()
    return response.content


def discover_running_on_compute_engine() -> bool:
    """
    Returns True if the current Python process is running on Google Compute
    Engine.

    Returns
    -------
    bool
        Whether the current process is running on GCE.
    """
    try:
        _google_metadata_request("")
        return True
    except (requests.ConnectionError, requests.HTTPError):
        return False


def discover_instance_name() -> bytes:
    """
    Query Google Compute Engine metadata to discover the host instance's name.

    If not running on Google Compute Engine, this function raises either a
    requests.HTTPError or a requests.ConnectionError.

    Returns
    -------
    bytes : The raw name of the running instance.

    Examples
    --------

    >>> discover_instance_name()
    b'asgard'
    """
    return _google_metadata_request("/computeMetadata/v1/instance/name")


def discover_instance_zone() -> bytes:
    """
    Query Google Compute Engine metadata to discover the host instance's zone.

    If not running on Google Compute Engine, this function raises either a
    requests.HTTPError or a requests.ConnectionError.

    Returns
    -------

    bytes : The URL of the running instance's zone.

    Examples
    --------

    >>> discover_instance_zone()
    b'projects/492788363398/zones/us-west1-a'
    """
    return _google_metadata_request("/computeMetadata/v1/instance/zone")


def discover_project_id() -> bytes:
    return _google_metadata_request("/computeMetadata/v1/project/project-id")


def _get_access_token() -> str:
    raw = _google_metadata_request(
        "/computeMetadata/v1/instance/service-accounts/default/token"
    )
    parsed = json.loads(raw)
    return parsed["access_token"]
