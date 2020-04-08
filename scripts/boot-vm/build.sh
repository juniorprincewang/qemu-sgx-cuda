#!/bin/bash
set -e
set -x


#./configure --prefix=/opt/qemu-2.4.0 --target-list=x86_64-softmmu --enable-kvm --disable-xen --enable-debug --enable-debug-info --enable-cuda
#./configure --target-list=x86_64-softmmu --enable-kvm --disable-xen --enable-debug --enable-debug-info --enable-cuda
../../configure --cxx=/usr/local/cuda/bin/nvcc --target-list="x86_64-softmmu" --enable-debug --enable-vnc \
	--enable-vnc-jpeg --enable-vnc-png --enable-kvm --enable-spice --enable-curl --enable-tools --enable-sdl \
	--enable-gtk --enable-cuda --disable-git-update --disable-capstone
make -j 16
