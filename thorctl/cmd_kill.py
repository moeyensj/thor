from .main import cli


@cli.command()
def kill():
    print("kill")
    raise NotImplementedError()
