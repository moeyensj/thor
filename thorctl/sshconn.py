import colorama
import socket
from typing import List

import paramiko.client
from .worker_pool import WorkerPoolManager


class WorkerPoolSSHConnection:
    def __init__(self, manager: WorkerPoolManager):
        self.manager = manager

    def connect(self):
        instances = self.manager.list_worker_instances()
        addrs = [_get_external_ip(instance) for instance in instances]
        hosts = {ip: instance["name"] for ip, instance in zip(addrs, instances)}
        clients = {}
        for ip, name in hosts.items():
            client = paramiko.client.SSHClient()
            client.set_missing_host_key_policy(_IgnoreMissingHostKeys)
            client.connect(hostname=ip)
            clients[name] = client
        return clients

    def stream_logs(self, colorize: bool = True):
        clients = self.connect()
        # First, establish connections to the workers.
        channels = {}
        for name, client in clients.items():
            transport = client.get_transport()
            channel = transport.open_session(timeout=1)
            channel.get_pty()
            channel.exec_command("journalctl -o cat -f -u thor-worker.service")
            channel.settimeout(0.05)
            stdout = channel.makefile("r", 4096)
            channels[name] = stdout

        # Set up pretty colors, if requested.
        if colorize:
            printer = ColorizedPrinter(list(clients.keys()))
        else:
            printer = PlainPrinter()

        # Loop over the open connections, printing output as we get it.
        while True:
            # Keep track of any connections that appear to be closed. We should
            # remove them from the list that we loop over.
            closed = set()

            for instance, stdout in channels.items():
                # If any channel greedily emits at least 1024 lines, then pause
                # and move on to other connections to give them a chance to spam
                # us too.
                i = 0
                while i < 1024:
                    try:
                        line = stdout.readline()[:-1]
                        i += 1
                        printer.print(instance, line)
                    except socket.timeout:
                        # Wait for more input - exit the loop.
                        break
                    except OSError:
                        # Closed - exit the loop.
                        closed.add(instance)
                        break

            # Clean up and close channels to any commands that exited.
            for closed_instance in closed:
                client = clients[closed_instance]
                client.close()
                clients.pop(closed_instance)

            if len(clients) == 0:
                return


class ColorizedPrinter:
    def __init__(self, hosts: List[str]):
        colors = [
            colorama.Fore.GREEN,
            colorama.Fore.YELLOW,
            colorama.Fore.CYAN,
            colorama.Style.BRIGHT + colorama.Fore.BLUE,
            colorama.Style.DIM + colorama.Fore.YELLOW,
            colorama.Style.DIM + colorama.Fore.RED,
            colorama.Style.DIM + colorama.Fore.GREEN,
            colorama.Fore.RED,
            colorama.Style.BRIGHT + colorama.Fore.GREEN,
            colorama.Fore.WHITE,
            colorama.Fore.MAGENTA,
        ]
        self.colors_by_name = {}
        for i, name in enumerate(hosts):
            color = colors[i % len(colors)]
            self.colors_by_name[name] = color

    def print(self, hostname, message):
        color = self.colors_by_name.get(hostname, "")
        reset = colorama.Style.RESET_ALL
        print(f"{color}{hostname}{reset}: {message}")


class PlainPrinter:
    def __init__(self):
        pass

    def print(self, hostname, message):
        print(f"{hostname}: {message}")


def _get_external_ip(instance_description: dict) -> str:
    networks = instance_description.get("networkInterfaces", [])
    for net in networks:
        access_configs = net.get("accessConfigs", [])
        for ac in access_configs:
            if ac.get("natIP", None) is not None:
                return ac["natIP"]
    raise ValueError(
        f"no external IP address found for instance {instance_description['name']}"
    )


class _IgnoreMissingHostKeys(paramiko.client.MissingHostKeyPolicy):
    def missing_host_key(self, client, hostname, key):
        return
