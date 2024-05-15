

(
cd /
echo Install pkg config
git clone https://gitlab.freedesktop.org/pkg-config/pkg-config 
cd pkg-config
set
./autogen.sh 
make
make install 
)

if [[ "$AUDITWHEEL_POLICY" == "manylinux2014" ]];
then

echo Install wget
ulimit -n 1024 
yum install wget pkgconfig -y || exit 1
bash ./clean.sh all

fi
