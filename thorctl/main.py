import click


@click.group()
def main():
    print("OK")


@main.command()
def list():
    print("list")
    raise NotImplementedError()


@main.command()
def new():
    print("new")
    raise NotImplementedError()


@main.command()
def status():
    print("status")
    raise NotImplementedError()


@main.command()
def results():
    print("results")
    raise NotImplementedError()
