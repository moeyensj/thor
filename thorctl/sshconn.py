import colorama
from pssh.clients import ParallelSSHClient
from pssh.exceptions import Timeout


from .worker_pool import WorkerPoolManager


class WorkerPoolSSHConnection:
    def __init__(self, manager: WorkerPoolManager):
        self.manager = manager
        self.pssh_client = ParallelSSHClient(hosts=self._remote_addresses(),)

    def _remote_addresses(self):
        instances = self.manager.list_worker_instances()
        addrs = [_get_external_ip(instance) for instance in instances]
        self.hostnames = {
            ip: instance["name"] for ip, instance in zip(addrs, instances)
        }
        return addrs

    def stream_logs(self, colorize: bool = True):
        output = self.pssh_client.run_command(
            "journalctl -o cat -f -u thor-worker.service", use_pty=True, read_timeout=0.2,
        )

        if colorize:
            colorama.init()
            colors_per_host = {}
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
            for i, host_out in enumerate(output):
                colors_per_host[host_out.host] = colors[i % len(colors)]
        while True:
            try:
                for host_out in output:
                    try:
                        for line in host_out.stdout:
                            hostname = self.hostnames[host_out.host]
                            color = colors_per_host[host_out.host]
                            reset = colorama.Style.RESET_ALL
                            print(f"{color}{hostname}: {line}{reset}")
                    except Timeout:
                        pass
            except KeyboardInterrupt:
                for host_out in output:
                    host_out.client.close_channel(host_out.channel)
                self.pssh_client.join(output)
                raise


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
