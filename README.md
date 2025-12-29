# Python VM
This is a Python implementation of a virtual machine that executes CPython bytecode.

> I completed this project as part of Python course by Yandex School of Data Analysis

## Implementation
This VM supports a major subset of [CPython](https://github.com/python/cpython) 3.12.5 bytecode instructions (as specified by https://docs.python.org/3/library/dis.html). However, some advanced features (such as classes, context managers, exceptions and type annotations) are not supported, as well as some uncommon instructions. Since CPython bytecode is considered an implementation detail of CPython and lacks stable specification, a significant part of the effort in working on the project was put into examining CPython source code. Therefore, the general execution model and the design of certain features are quite similar to CPython.

`tests` directory contains a test suite that covers most of the target instructions and language features. Since this interpreter does not implement all possible instructions, some tests are expected to fail. Note that tests are designed to forbid direct interfacing with the actual CPython interpreter and using certain Python introspection hacks (which would otherwise make the "VM implementation" trivial).

## Usage
You will need Python 3.12.5 with some libraries (`requirements.txt`) to run and test this software. To set up the environment with [pyenv](https://github.com/pyenv/pyenv):
```shell
$ pyenv install 3.12.5
$ ~/.pyenv/versions/3.12.5/bin/python -m venv myenv
$ source myenv/bin/activate
(myenv)$ pip install --upgrade -r requirements.txt
```

Run the interpreter on a Python source file:
```shell
(myenv)$ python3 src/vm.py examples/hello_world.py
Hello, world!
```
Run tests:
```shell
$ pytest tests/test_public.py -vvv -tb=no
```

You can find tested code samples in [cases.py](tests/cases.py).
