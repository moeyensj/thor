import colorama
import socket
from typing import List, Optional
import time
import logging

import paramiko.ssh_exception
import paramiko.client
from .worker_pool import WorkerPoolManager

logger = logging.getLogger("thorctl")

_COLORS = [
    colorama.Fore.GREEN,
    colorama.Fore.YELLOW,
    colorama.Fore.CYAN,
    colorama.Style.BRIGHT + colorama.Fore.BLUE,
    colorama.Style.DIM + colorama.Fore.YELLOW,
    colorama.Style.DIM + colorama.Fore.GREEN,
    colorama.Fore.RED,
    colorama.Style.BRIGHT + colorama.Fore.GREEN,
    colorama.Fore.WHITE,
    colorama.Fore.MAGENTA,
]


class WorkerPoolSSHConnection:
    def __init__(self, manager: WorkerPoolManager):
        self.manager = manager
        self.connections = {}  # name -> WorkerSSHConnection

        # counts all connections *ever*, not just currently active ones.
        self.connection_count = 0

    def _update_connections(self, conn_timeout: int = 3) -> List[str]:
        """Updates the WorkerPoolSSHConnection's internal connection state.

        First, we check if there are any new instances that we have not
        connected to. New connections are established to these. If a connection
        can't be established, a warning message is emitted.

        Second, we remove any disconnected connections from self.connections.

        Parameters
        ----------
        conn_timeout : int
            How long to wait, in seconds, to establish a connection to a single
            host.

        Returns
        -------
        List[str]
            A list of the names of instances that were newly connected to.

        """

        instances = self.manager.list_worker_instances()

        # Gather names of new instances that don't have a connection.
        to_add = []
        for instance in instances:
            if instance["name"] not in self.connections:
                to_add.append(instance)

        # Try to connect to all the new instances.
        added = []
        for instance in to_add:
            success = self._try_to_connect(instance, conn_timeout)
            if success:
                added.append(instance["name"])

        # Gather names of instances that we've disconnected from.
        to_remove = []
        for name, conn in self.connections.items():
            if not conn.connected:
                to_remove.append(name)

        # Remove all the disconnected instances.
        for name in to_remove:
            self.connections.pop(name)

        return added

    def _try_to_connect(self, instance: dict, conn_timeout: int) -> bool:
        """Attempt to connect to an instance. Return true if success, false if failed.

        Parameters
        ----------
        instance : dict
            A resource description of the instance.
        conn_timeout : int
            Time in seconds to wait when trying to connect.

        Returns
        -------
        bool
            Whether we connected successfully.

        """
        color = self._choose_color()
        name = instance["name"]
        logger.debug("attempting to connect to %s", name)
        ip = _get_external_ip(instance)
        if ip is None:
            logger.debug("instance %s does not have an IP yet", name)
            return False

        conn = WorkerSSHConnection(name, ip, color)
        try:
            conn.connect(timeout=conn_timeout)
            self.connections[name] = conn
            self.connection_count += 1
            logger.debug("connected to %s (%s)", name, ip)
            return True
        except Exception as e:
            logger.debug("failed to connect to %s (%s): %s", name, ip, e)
            return False

    def _choose_color(self):
        return _COLORS[self.connection_count % len(_COLORS)]

    def stream_logs(self):
        last_client_update = 0
        # Loop over the open connections, printing output as we get it.
        last_loop = 0
        while True:
            # Every 5 seconds, update our connections, attaching to new workers.
            if time.time() - last_client_update > 5:
                last_client_update = time.time()
                added = self._update_connections()

                for instance in added:
                    conn = self.connections[instance]
                    conn.start_tailing_logs()
                if len(self.connections) == 0:
                    logger.debug("not connected to any workers")

            # Main loop here: print up to 64 lines from each connection.
            for conn in self.connections.values():
                for line in conn.iter_available_lines(max_lines=64):
                    conn.print(line)

            # Poll at most every 0.01 seconds.
            since_last_loop = time.time() - last_loop
            if since_last_loop < 0.01:
                time.sleep(0.01 - since_last_loop)
            last_loop = time.time()


class WorkerSSHConnection:
    """
    Represents a persistent SSH connection to a Worker instance.

    The connection can be connected or disconnected.
    """
    def __init__(self, instance_name: str, instance_ip: str, print_color: str):
        self.instance_name = instance_name
        self.instance_ip = instance_ip
        self.print_color = print_color

        self._client = paramiko.client.SSHClient()
        self._client.set_missing_host_key_policy(_IgnoreMissingHostKeys)

        self._session = None
        self._session_stdout = None

        self._read_buffer = bytes()

        self.connected = False
        self.command_running = False
        self.exit_status = None

    def print(self, message: str):
        """
        Print a message, prefixed with the instance's hostname and using the
        WorkerSSHConnection's color.

        Parameters
        ----------
        message : str
            Message to print
        """

        reset = colorama.Style.RESET_ALL
        print(f"{self.print_color}{self.instance_name}{reset}: {message}")

    def connect(self, timeout: int = 1):
        """Establish a connection to the instance.

        Parameters
        ----------
        timeout : int
            Time, in seconds, to wait for the connection to be established.
        """

        self._client.connect(hostname=self.instance_ip, timeout=timeout)
        transport = self._client.get_transport()
        self._session = transport.open_session(timeout)
        self.connected = True

    def start_command(self, cmd: str):
        """Send a command over the connected SSH session.

        Parameters
        ----------
        cmd : str
            Command to send over ssh.

        Examples
        --------
        >>> conn.start_command("ls -l")
        """
        assert self.connected
        assert not self.command_running

        self._session.get_pty()
        self._session.exec_command(cmd)
        self._session_stdout = self._session.makefile("r", 4096)
        self.command_running = True

    def disconnect(self):
        """End any running session and connection. """

        if self.command_running:
            self._session_stdout.close()
            self.command_running = False
        if self.connected:
            self._session.close()
            self._client.close()
            self.connected = False

    def start_tailing_logs(self):
        """Start a session which will stream logs from a THOR worker.

        The log lines can be retrieved with iter_available_lines.

        Examples
        --------
        >>> conn = WorkerSSHConnection("asgard", "192.168.1.1", colorama.Fore.GREEN)
        >>> conn.connect()
        >>> conn.start_tailing_logs()
        >>> for line in conn.iter_available_lines(64):
        ...     conn.print(line)
        """

        self.start_command("journalctl -o cat -f -u thor-worker.service")
        self._session.settimeout(0.05)

    def iter_available_lines(self, max_lines: int):
        """
        Iterate over buffered lines from stdout of any running command.

        If not connected, or if no command is running, this returns without
        yielding any lines.

        If there is an incomplete line in the buffer, it is not returned.

        When the running command has exited, this automatically disconnects.

        Parameters
        ----------
        max_lines : int
            Maximum number of lines to yield.

        Examples
        --------
        >>> conn = WorkerSSHConnection("asgard", "192.168.1.1", colorama.Fore.GREEN)
        >>> conn.connect()
        >>> conn.start_tailing_logs()
        >>> while conn.connected:
        ...     for line in conn.iter_available_lines(64):
        ...         conn.print(line)
        """

        if not self.connected or not self.command_running:
            return

        lines_read = 0
        while True:
            # If we have any lines already buffered, hand them out until we hit
            # max_lines.
            for line in self._iterate_buffered_lines():
                yield line
                lines_read += 1
                if lines_read >= max_lines:
                    return

            # If we can add to the buffer, do so, and then go back to yielding
            # from the buffer.
            if self._session.recv_ready():
                data = self._session.recv(4096)
                self._read_buffer = self._read_buffer + data
                continue

            # We couldn't add to the buffer. Maybe it's because the command exited?
            if self._session.exit_status_ready():
                # Yes, the command has completed.
                #
                # But there's a rare race possible: we could have gotten some
                # output since we last checked, but before the command exited.
                # If this is the case, we should return to the top of the loop,
                # and read data until the recv buffer is exhausted.
                #
                # Once recv_ready returns False consistently *and* the exit
                # status is ready, we can be sure we have read all output.
                if self._session.recv_ready():
                    continue

                self.exit_status = self._session.recv_exit_status()
                self.command_running = False
                yield f"command exited with status {self.exit_status}"
                return

            # Otherwise, there are just no available lines right now, so return.
            return

    def _iterate_buffered_lines(self):
        next_linebreak = self._read_buffer.find(b"\n")
        while next_linebreak > 0:
            line = self._read_buffer[:next_linebreak]
            self._read_buffer = self._read_buffer[(next_linebreak+1):]
            yield line.decode()
            next_linebreak = self._read_buffer.find(b"\n")


def _get_external_ip(instance_description: dict) -> Optional[str]:
    networks = instance_description.get("networkInterfaces", [])
    for net in networks:
        access_configs = net.get("accessConfigs", [])
        for ac in access_configs:
            if ac.get("natIP", None) is not None:
                return ac["natIP"]
    return None


class _IgnoreMissingHostKeys(paramiko.client.MissingHostKeyPolicy):
    def missing_host_key(self, client, hostname, key):
        return
