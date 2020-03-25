#!/usr/bin/sudo /bin/bash

# set up environment
. $(dirname $0)/scripts/settings

set -x

gdb -q --args \
${QEMU_BIN} ${MEM_OPT} ${IMAGE} ${KVM} ${SGX} ${SGX_CONFIG} ${SMP}\
    ${VGA} ${!GRAPHICS} ${!AUTOBALLOON} ${NET} ${!QMP} \
    ${!SNAPSHOT} ${!DEBUG} ${MONITOR} \
	-device virtio-cuda-pci \
    -device virtcudaport