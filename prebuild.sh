if [[ "$PYOOMPH_CONFIG_FILE" == "" ]]; then
cd $(readlink -f $(dirname $0))
echo "Sourcing default config pyoomph_config.env"
source pyoomph_config.env ||  exit 1
PYOOMPH_CONFIG_FILE=pyoomph_config.env
else
echo "Sourcing custom config file $PYOOMPH_CONFIG_FILE"
source "$PYOOMPH_CONFIG_FILE" ||  exit 1
fi


PYOOMPH_CONFIG_FILE=$(readlink -f $PYOOMPH_CONFIG_FILE)

export PYOOMPH_CONFIG_FILE

cd src/thirdparty
cd oomph-lib
make -j 4 || exit 1
cd ..
if ! $PYOOMPH_NO_TCC; then
bash ./compile_tcc_for_pyoomph.sh || exit
fi

