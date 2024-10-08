#!/usr/bin/env bash

COLOR_OUTPUT=-fdiagnostics-color=always

if git help &>/dev/null ; then
PYOOMPH_GIT_HASH=$(git rev-parse HEAD)
PYOOMPH_BRANCH=$(git branch --show-current)
export PYOOMPH_VERSION_INFO=-DPYOOMPH_GIT_HASH="\"${PYOOMPH_GIT_HASH}\""\ -DPYOOMPH_BRANCH="\"${PYOOMPH_BRANCH}\""
else
PYOOMPH_VERSION_INFO=
fi

#PYVERS=$(python3 -c 'import sys; print(sys.version_info>=(3,10,12));')

PYVERS=$(python3 -c 'import pip; print(int(pip.__version__.split(".")[0])>=24);')


if [[ "$PYVERS" == "True" ]]; then
 EDITABLE_MODE="--config-settings editable_mode=strict"
else
 echo "ERROR: Please upgrade pip to at least version 24.0!"
 echo "pyoomph -m pip install --upgrade pip"
 exit
 EDITABLE_MODE=
fi

if [[ "$PYOOMPH_CONFIG_FILE" == "" ]]; then
cd $(readlink -f $(dirname $0))
echo "Sourcing default config pyoomph_config.env"
source pyoomph_config.env ||  exit 1
PYOOMPH_CONFIG_FILE=pyoomph_config.env
else
echo "Sourcing custom config file $PYOOMPH_CONFIG_FILE"
source "$PYOOMPH_CONFIG_FILE" ||  exit 1
fi

if $PYOOMPH_USE_MPI; then
 MPICCBINARY=mpic++
else
 MPICCBINARY=$CXX
fi

if [[ $MPICCBINARY == "" ]]; then
MPICCBINARY=g++
fi


if [[ "$OSTYPE" != "darwin"* ]]; then
cd $(readlink -f $(dirname $0))
if which ccache &>/dev/null; then 
NPY_NUM_BUILD_JOBS=4 CXX="$MPICCBINARY -pthread -g3 -shared -Wl,-O1 -Wl,-Bsymbolic-functions  -Wl,-z,relro $PYOOMPH_VERSION_INFO" CC="ccache $MPICCBINARY $COLOR_OUTPUT" LDSHARED="$MPICCBINARY -g3 -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-z,relro"  python3 -m pip install  -v "$@"  .  $EDITABLE_MODE || exit 1
 else
 NPY_NUM_BUILD_JOBS=4 CXX="$MPICCBINARY -pthread -g3 -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-z,relro $PYOOMPH_VERSION_INFO" CC="$MPICCBINARY $COLOR_OUTPUT" LDSHARED="$MPICCBINARY -g3 -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-z,relro"  python3 -m pip install  -v "$@" . $EDITABLE_MODE || exit 1
 fi
else
 if which ccache &>/dev/null; then 
  NPY_NUM_BUILD_JOBS=4 CXX="$MPICCBINARY -L/usr/local/lib $PYOOMPH_VERSION_INFO" CC="ccache $MPICCBINARY -I/usr/local/include $COLOR_OUTPUT" LD="$MPICCBINARY"  python3 -m pip install -v  "$@" . $EDITABLE_MODE || exit 1
 else
   NPY_NUM_BUILD_JOBS=4 CXX="$MPICCBINARY -L/usr/local/lib $PYOOMPH_VERSION_INFO" CC="$MPICCBINARY -I/usr/local/include $COLOR_OUTPUT" LD="$MPICCBINARY"  python3 -m pip install -v  "$@" .  $EDITABLE_MODE || exit 1
 fi
fi

#CC="ccache g++"  python3 -m pip install -e . -v "$@"

#
