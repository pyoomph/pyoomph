if [ $# -eq 0 ]; then
rm -rf build/ _pyoomph.cpython*.so pyoomph.egg-info/ tmp/ 
ccache -C
make -C src clean
exit
fi

for k in $@; do
if [[ "$k" == "all" ]]; then
  rm -rf build/ _pyoomph.cpython*.so pyoomph.egg-info/ tmp/ 
  ccache -C
  make -C src/thirdparty/oomph-lib clean
  make -C src/thirdparty/tinycc clean  
  make -C src clean
elif [[ "$k" == "pyoomph" ]]; then
  rm -r build/ _pyoomph.cpython*.so pyoomph.egg-info/
elif [[ "$k" == "ccache" ]]; then
  ccache -C      
elif [[ "$k" == "tcc" ]]; then
  make -C src/thirdparty/tinycc clean
elif [[ "$k" == "oomph-lib" ]]; then
  make -C src/thirdparty/oomph-lib clean
fi
done
