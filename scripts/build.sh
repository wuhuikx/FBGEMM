#gcc 7.0 (in cluster, use this command to switch to "scl enable devtoolset-7 bash")
cd ..
git submodule sync
git submodule update --init --recursive
mkdir build && cd build
cmake3 ..
make -j `nproc`

