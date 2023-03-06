### ABSL-Extra

A collection of utils I commonly use for running my experiments.

Minimal example
```python
from absl_extra import App

def main(*args):
    ...

if __name__ == "__main__":
    app = App(name="bla")
    app.run(main)
```