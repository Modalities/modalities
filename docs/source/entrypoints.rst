.. role:: python(code)
   :language: python

.. role:: bash(code)
   :language: bash


Entrypoints
=======================================================

We use `click <https://click.palletsprojects.com/en/>`_ as a tool to add new entry points and their CLI arguments.
For this we have a main entry point from which all other entry points are started. 

The main entry point is :file:`src/modalities/__main__.py:main()`. 
We register other sub-entrypoints by using our main :python:`click.group`, called :python:`main`, as follows:

.. code-block:: python

  @main.command(name="my_new_entry_point")


See the following full example:

.. code-block:: python

  
  import click
  import click_pathlib
  
  
  @click.group()
  def main() -> None:
      pass
  
  
  config_option = click.option(
      "--config_file_path",
      type=click_pathlib.Path(exists=False),
      required=True,
      help="Path to a file with the YAML config file.",
  )
  
  
  @main.command(name="do_stuff")
  @config_option
  @click.option(
      "--my_cli_argument",
      type=int,
      required=True,
      help="New integer argument",
  )
  def entry_point_do_stuff(config_file_path: Path, my_cli_argument: int):
      print(f"Do stuff with {config_file_path} and {my_cli_argument}...)
      ...
  
  if __name__ == "__main__":
      main()

With 
    
.. code-block:: python
    
  [project.scripts]
  modalities = "modalities.__main__:main"

in our :file:`pyproject.toml`, we can start only main with :python:`modalities` (which does nothing), or a specific sub-entrypoint e.g. :bash:`modalities do_stuff --config_file_path config_files/config.yaml --my_cli_argument 3537`.

Alternatively, directly use :bash:`src/modalities/__main__.py do_stuff --config_file_path config_files/config.yaml --my_cli_argument 3537`.
