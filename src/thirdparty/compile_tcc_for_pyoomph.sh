# You can include your own version of the TinyC compiler by putting it into a subfolder tinycc and compile with PYOOMPH_NO_TCC=false

if [[ "$PYOOMPH_CONFIG_FILE" == "" ]]; then
 PYOOMPH_CONFIG_FILE=../../pyoomph_config.env
else
 PYOOMPH_CONFIG_FILE=$(readlink -f $PYOOMPH_CONFIG_FILE)
fi

source $PYOOMPH_CONFIG_FILE

if [[ "$CC" == "" ]]; then
 CC=gcc
fi

cd tinycc

echo 0.9.27 >VERSION
./configure --extra-ldflags=-fPIC --enable-static $PYOOMPH_TCC_CONFIGURE_OPTIONS $PYOOMPH_TCC_CONFIGURE_EXTRA
cat > config-extra.mak << EOCAT
CC=$CC -fPIC
EOCAT

if [[ "$PYOOMPH_CROSS_COMPILE_MINGW" == "true" ]]; then
echo "CONFIG_WIN32=yes" >> config.mak
echo "CONFIG_ldl=no"  >> config.mak
fi

make -j 4 CONFIG_bcheck=no CONFIG_backtrace=no 

if [[ "$PYOOMPH_CROSS_COMPILE_MINGW" == "true" ]]; then
 echo "Using Wine for: c2str.exe include/tccdefs.h tccdefs_.h"
 wine ./c2str.exe include/tccdefs.h tccdefs_.h
 make -j 4 CONFIG_bcheck=no CONFIG_backtrace=no  libtcc1.a
 echo "Using Wine for: tcc.exe -c libtcc1.c -o libtcc1.o -B../win32 -I../include -I.."
 ( 
 cd lib
 wine ../tcc.exe -c libtcc1.c -o libtcc1.o -B../win32 -I../include -I..
 wine ../tcc.exe -c alloca.S -o alloca.o -B../win32 -I../include -I..
 wine ../tcc.exe -c alloca-bt.S -o alloca-bt.o -B../win32 -I../include -I..
 wine ../tcc -c tcov.c -o tcov.o -B../win32 -I../include -I..
 wine ../tcc -c stdatomic.c -o stdatomic.o -B../win32 -I../include -I..
 wine ../tcc -c ../win32/lib/chkstk.S -o chkstk.o -B../win32 -I../include -I..
 wine ../tcc -c ../win32/lib/crt1.c -o crt1.o -B../win32 -I../include -I..
 wine ../tcc -c ../win32/lib/crt1w.c -o crt1w.o -B../win32 -I../include -I..
 wine ../tcc -c ../win32/lib/wincrt1.c -o wincrt1.o -B../win32 -I../include -I..
 wine ../tcc -c ../win32/lib/wincrt1w.c -o wincrt1w.o -B../win32 -I../include -I..
 wine ../tcc -c ../win32/lib/dllcrt1.c -o dllcrt1.o -B../win32 -I../include -I..
 wine ../tcc -c ../win32/lib/dllmain.c -o dllmain.o -B../win32 -I../include -I..
 wine ../tcc -ar rcs ../libtcc1.a libtcc1.o alloca.o alloca-bt.o tcov.o stdatomic.o chkstk.o crt1.o crt1w.o wincrt1.o wincrt1w.o dllcrt1.o dllmain.o
 )
 make -j 4 CONFIG_bcheck=no CONFIG_backtrace=no   libtcc1.a
fi


cp libtcc1.a ../../../pyoomph/jitbridge/
rm VERSION

