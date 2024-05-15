#!/usr/bin/env bash

bash ./install.sh --no-build-isolation --editable   "$@"


if [ $? -eq 0 ]; then
echo "Generating stubs (useful for e.g. VSCode full code completion)"
if which pybind11-stubgen &>/dev/null; then 
 if ! pybind11-stubgen -o _pyoomph-stubs  _pyoomph --no-setup-py 2>/dev/null; then
    pybind11-stubgen -o _pyoomph-stubs  _pyoomph  || echo "Error in stub generation" >&2
 fi
 python3 src/pybind/patch_stubs.py || exit 1
 if ! diff -q _pyoomph-stubs/_pyoomph.pyi _pyoomph-stubs/__init__.pyi &>/dev/null ; then
  cp _pyoomph-stubs/_pyoomph.pyi _pyoomph-stubs/__init__.pyi
 fi
 rm -f _pyoomph-stubs/_pyoomph.pyi
 
else
echo "To generate stubs (useful for e.g. VSCode full code completion), ensure to install pybind11-stubgen"
fi
fi
