cd /tmp
git clone https://github.com/Boolector/btor2tools.git
cd btor2tools
./configure.sh
cd build
make
sudo make install
cd ../..
git clone https://github.com/soarlab/pysmt.git
cd pysmt
python setup.py install
cd ..
pysmt-install --btor
