import click
import pybamm

@click.command()
def pybamm2diffsl(count, name):
    model = pybamm.lithium_ion.SPM()
    for x in range(count):
        click.echo(f"Hello {name}!")


if __name__ == '__main__':
    pybamm2diffsl()