#ifndef VIRTCR_IOC_H
#define VIRTCR_IOC_H

// #define VIRTIO_CUDA_DEBUG
// #define KMALLOC_SHIFT 22 // 4MB
#define KMALLOC_SHIFT 13
#define KMALLOC_SIZE (1UL<<KMALLOC_SHIFT)
#define PAGE_SHIFT 	12
#define PAGE_SIZE 	(1UL<<PAGE_SHIFT)
#define VIRTIO_ENC 


#ifndef __KERNEL__
#define __user

#include <stdint.h>
#include <sys/ioctl.h>

#define VIRTIO_CUDA_HELLO 					0
/** module control	**/
#define VIRTIO_CUDA_REGISTERFATBINARY 		1
#define VIRTIO_CUDA_UNREGISTERFATBINARY		2
#define VIRTIO_CUDA_REGISTERFUNCTION 		3
#define VIRTIO_CUDA_LAUNCH 					4
/* memory management */
#define VIRTIO_CUDA_MALLOC 					5
#define VIRTIO_CUDA_MEMCPY 					6
#define VIRTIO_CUDA_MEMCPY_ASYNC 			24
#define VIRTIO_CUDA_MEMSET 					25
#define VIRTIO_CUDA_FREE 					7
/**	device management	**/
#define VIRTIO_CUDA_GETDEVICE 				8
#define VIRTIO_CUDA_GETDEVICECOUNT 			12
#define VIRTIO_CUDA_GETDEVICEPROPERTIES 	9
#define VIRTIO_CUDA_SETDEVICE 				13
#define VIRTIO_CUDA_DEVICERESET 			14
#define VIRTIO_CUDA_DEVICESYNCHRONIZE 		26

/*stream management*/
#define VIRTIO_CUDA_STREAMCREATE 			15
#define VIRTIO_CUDA_STREAMDESTROY 			16
/*event management*/
#define VIRTIO_CUDA_EVENTCREATE 			17
#define VIRTIO_CUDA_EVENTDESTROY 			18
#define VIRTIO_CUDA_EVENTRECORD 			19
#define VIRTIO_CUDA_EVENTSYNCHRONIZE 		20
#define VIRTIO_CUDA_EVENTELAPSEDTIME 		21
#define VIRTIO_CUDA_EVENTCREATEWITHFLAGS 	27
/*Thread management*/
#define VIRTIO_CUDA_THREADSYNCHRONIZE 		22
/*Error Handling*/
#define VIRTIO_CUDA_GETLASTERROR 			23

/*zero-copy*/
#define VIRTIO_CUDA_HOSTREGISTER 			28
#define VIRTIO_CUDA_HOSTGETDEVICEPOINTER 	29
#define VIRTIO_CUDA_HOSTUNREGISTER 			30
#define VIRTIO_CUDA_SETDEVICEFLAGS 			31
#define VIRTIO_CUDA_MEMGETINFO 				32
#define VIRTIO_CUDA_MALLOCHOST 				33
#define VIRTIO_CUDA_FREEHOST 				34
#define VIRTIO_CUDA_MEMCPYTOSYMBOL 			35
#define VIRTIO_CUDA_MEMCPYFROMSYMBOL 		36
#define VIRTIO_CUDA_REGISTERVAR 			37
#define VIRTIO_CUDA_STREAMWAITEVENT 		38
#define VIRTIO_CUDA_STREAMSYNCHRONIZE 		39
#define VIRTIO_CUDA_STREAMCREATEWITHFLAGS 	40

#define VIRTIO_CUDA_PEEKATLASTERROR 		41
#define VIRTIO_CUDA_EVENTQUERY 				42
#define VIRTIO_CUDA_PRIMARYCONTEXT			43
#define VIRTIO_CUDA_DEVICESETCACHECONFIG 	44

#define VIRTIO_CUDA_MMAPCTL 				10
#define VIRTIO_CUDA_MUNMAPCTL 				11
//sgx----------------------------------
#define VIRTIO_SGX_MSG0 					80
#define VIRTIO_SGX_MSG1 					81
#define VIRTIO_SGX_MSG3 					82
#define VIRTIO_SGX_MEMCPY 					83
//sgx----------------------------------
#define VIRTIO_CUBLAS_CREATE 				100
#define VIRTIO_CUBLAS_DESTROY 				101
#define VIRTIO_CUBLAS_SETVECTOR 			102
#define VIRTIO_CUBLAS_GETVECTOR 			103
#define VIRTIO_CUBLAS_SGEMM 				104
#define VIRTIO_CUBLAS_DGEMM 				105
#define VIRTIO_CUBLAS_SETSTREAM 			106
#define VIRTIO_CUBLAS_GETSTREAM 			107
#define VIRTIO_CUBLAS_SASUM 				108
#define VIRTIO_CUBLAS_DASUM 				109
#define VIRTIO_CUBLAS_SAXPY 				110
#define VIRTIO_CUBLAS_DAXPY 				111
#define VIRTIO_CUBLAS_SCOPY 				112
#define VIRTIO_CUBLAS_DCOPY 				113
#define VIRTIO_CUBLAS_SGEMV 				114
#define VIRTIO_CUBLAS_DGEMV 				115
#define VIRTIO_CUBLAS_SDOT 					116
#define VIRTIO_CUBLAS_DDOT 					117
#define VIRTIO_CUBLAS_SSCAL 				118
#define VIRTIO_CUBLAS_DSCAL 				119
#define VIRTIO_CUBLAS_SETMATRIX 			120
#define VIRTIO_CUBLAS_GETMATRIX 			121
//cublas----------------------------------
#define VIRTIO_CURAND_CREATEGENERATOR 			200
#define VIRTIO_CURAND_GENERATE 					201
#define VIRTIO_CURAND_GENERATENORMAL			202
#define VIRTIO_CURAND_GENERATENORMALDOUBLE		203
#define VIRTIO_CURAND_GENERATEUNIFORM			204
#define VIRTIO_CURAND_GENERATEUNIFORMDOUBLE		205
#define VIRTIO_CURAND_DESTROYGENERATOR			206
#define VIRTIO_CURAND_SETGENERATOROFFSET		207
#define VIRTIO_CURAND_SETPSEUDORANDOMSEED		208
#define VIRTIO_CURAND_CREATEGENERATORHOST		209

#else

#include <linux/ioctl.h>


#endif //KERNEL



//for crypto_data_header, if these is no openssl header
#ifndef RSA_PKCS1_PADDING

#define RSA_PKCS1_PADDING	1
#define RSA_SSLV23_PADDING	2
#define RSA_NO_PADDING		3
#define RSA_PKCS1_OAEP_PADDING	4

#endif

/*
 * function arguments
*/
typedef struct VirtIOArg
{
	uint32_t cmd;
	uint32_t tid;
	uint64_t src;
	uint32_t srcSize;
	uint64_t src2;
	uint32_t srcSize2;
	uint64_t dst;
	uint32_t dstSize;
	uint64_t flag;
	uint64_t param;
	uint32_t paramSize;
	uint64_t param2;
	uint8_t mac[16];
} VirtIOArg;
/* see ioctl-number in https://github.com/torvalds/
	linux/blob/master/Documentation/ioctl/ioctl-number.txt
*/
#define VIRTIO_IOC_ID 0xBB

#define VIRTIO_IOC_HELLO \
	_IOWR(VIRTIO_IOC_ID,0,int)
/** module control	**/
#define VIRTIO_IOC_REGISTERFATBINARY \
	_IOWR(VIRTIO_IOC_ID,1, VirtIOArg)
#define VIRTIO_IOC_UNREGISTERFATBINARY	\
	_IOWR(VIRTIO_IOC_ID,2,VirtIOArg)
#define VIRTIO_IOC_REGISTERFUNCTION \
	_IOWR(VIRTIO_IOC_ID,3,VirtIOArg)
#define VIRTIO_IOC_LAUNCH \
	_IOWR(VIRTIO_IOC_ID,4,VirtIOArg)
/* memory management */
#define VIRTIO_IOC_MALLOC\
	_IOWR(VIRTIO_IOC_ID,5,VirtIOArg)
#define VIRTIO_IOC_MEMCPY \
	_IOWR(VIRTIO_IOC_ID,6,VirtIOArg)
#define VIRTIO_IOC_FREE \
	_IOWR(VIRTIO_IOC_ID,7,VirtIOArg)
/**	device management	**/
#define VIRTIO_IOC_GETDEVICE \
	_IOWR(VIRTIO_IOC_ID,8,VirtIOArg)
#define VIRTIO_IOC_GETDEVICEPROPERTIES \
	_IOWR(VIRTIO_IOC_ID,9,VirtIOArg)
#define VIRTIO_IOC_MMAPCTL \
	_IOWR(VIRTIO_IOC_ID,10,VirtIOArg)

#define VIRTIO_IOC_MUNMAPCTL \
	_IOWR(VIRTIO_IOC_ID,11,VirtIOArg)
#define VIRTIO_IOC_GETDEVICECOUNT \
	_IOWR(VIRTIO_IOC_ID,12,VirtIOArg)
#define VIRTIO_IOC_SETDEVICE \
	_IOWR(VIRTIO_IOC_ID,13,VirtIOArg)
#define VIRTIO_IOC_DEVICERESET \
	_IOWR(VIRTIO_IOC_ID,14,VirtIOArg)
#define VIRTIO_IOC_STREAMCREATE \
	_IOWR(VIRTIO_IOC_ID,15,VirtIOArg)

#define VIRTIO_IOC_STREAMDESTROY \
	_IOWR(VIRTIO_IOC_ID,16,VirtIOArg)
#define VIRTIO_IOC_EVENTCREATE \
	_IOWR(VIRTIO_IOC_ID,17,VirtIOArg)
#define VIRTIO_IOC_EVENTDESTROY \
	_IOWR(VIRTIO_IOC_ID,18,VirtIOArg)
#define VIRTIO_IOC_EVENTRECORD \
	_IOWR(VIRTIO_IOC_ID,19,VirtIOArg)
#define VIRTIO_IOC_EVENTSYNCHRONIZE \
	_IOWR(VIRTIO_IOC_ID,20,VirtIOArg)

#define VIRTIO_IOC_EVENTELAPSEDTIME \
	_IOWR(VIRTIO_IOC_ID,21,VirtIOArg)
#define VIRTIO_IOC_THREADSYNCHRONIZE \
	_IOWR(VIRTIO_IOC_ID,22,VirtIOArg)
#define VIRTIO_IOC_GETLASTERROR \
	_IOWR(VIRTIO_IOC_ID,23,VirtIOArg)

#define VIRTIO_IOC_MEMCPY_ASYNC \
	_IOWR(VIRTIO_IOC_ID,24,VirtIOArg)
#define VIRTIO_IOC_MEMSET \
	_IOWR(VIRTIO_IOC_ID,25,VirtIOArg)
#define VIRTIO_IOC_DEVICESYNCHRONIZE \
	_IOWR(VIRTIO_IOC_ID,26,VirtIOArg)

#define VIRTIO_IOC_EVENTCREATEWITHFLAGS \
	_IOWR(VIRTIO_IOC_ID,27,VirtIOArg)
#define VIRTIO_IOC_HOSTREGISTER \
	_IOWR(VIRTIO_IOC_ID,28,VirtIOArg)
#define VIRTIO_IOC_HOSTGETDEVICEPOINTER \
	_IOWR(VIRTIO_IOC_ID,29,VirtIOArg)
#define VIRTIO_IOC_HOSTUNREGISTER \
	_IOWR(VIRTIO_IOC_ID,30,VirtIOArg)
#define VIRTIO_IOC_SETDEVICEFLAGS \
	_IOWR(VIRTIO_IOC_ID,31,VirtIOArg)


#define VIRTIO_IOC_MEMGETINFO \
	_IOWR(VIRTIO_IOC_ID,32,VirtIOArg)
#define VIRTIO_IOC_MALLOCHOST \
	_IOWR(VIRTIO_IOC_ID,33,VirtIOArg)
#define VIRTIO_IOC_FREEHOST \
	_IOWR(VIRTIO_IOC_ID,34,VirtIOArg)

#define VIRTIO_IOC_MEMCPYTOSYMBOL \
	_IOWR(VIRTIO_IOC_ID,35,VirtIOArg)
#define VIRTIO_IOC_MEMCPYFROMSYMBOL \
	_IOWR(VIRTIO_IOC_ID,36,VirtIOArg)
#define VIRTIO_IOC_REGISTERVAR \
	_IOWR(VIRTIO_IOC_ID,37,VirtIOArg)
#define VIRTIO_IOC_STREAMWAITEVENT \
	_IOWR(VIRTIO_IOC_ID,38,VirtIOArg)
#define VIRTIO_IOC_STREAMSYNCHRONIZE \
	_IOWR(VIRTIO_IOC_ID,39,VirtIOArg)
#define VIRTIO_IOC_STREAMCREATEWITHFLAGS \
	_IOWR(VIRTIO_IOC_ID,40,VirtIOArg)

#define VIRTIO_IOC_PEEKATLASTERROR \
	_IOWR(VIRTIO_IOC_ID,41,VirtIOArg)
#define VIRTIO_IOC_EVENTQUERY \
	_IOWR(VIRTIO_IOC_ID,42,VirtIOArg)
#define VIRTIO_IOC_PRIMARYCONTEXT \
	_IOWR(VIRTIO_IOC_ID,43,VirtIOArg)
#define VIRTIO_IOC_DEVICESETCACHECONFIG \
	_IOWR(VIRTIO_IOC_ID,44,VirtIOArg)

// sgx ------------------------------
#define VIRTIO_IOC_SGX_MSG0 \
	_IOWR(VIRTIO_IOC_ID,80,VirtIOArg)
#define VIRTIO_IOC_SGX_MSG1 \
	_IOWR(VIRTIO_IOC_ID,81,VirtIOArg)
#define VIRTIO_IOC_SGX_MSG3 \
	_IOWR(VIRTIO_IOC_ID,82,VirtIOArg)
#define VIRTIO_IOC_SGX_MEMCPY \
	_IOWR(VIRTIO_IOC_ID,83,VirtIOArg)
// cublas ------------------------------
#define VIRTIO_IOC_CUBLAS_CREATE \
	_IOWR(VIRTIO_IOC_ID,100,VirtIOArg)
#define VIRTIO_IOC_CUBLAS_DESTROY \
	_IOWR(VIRTIO_IOC_ID,101,VirtIOArg)
#define VIRTIO_IOC_CUBLAS_SETVECTOR \
	_IOWR(VIRTIO_IOC_ID,102,VirtIOArg)
#define VIRTIO_IOC_CUBLAS_GETVECTOR \
	_IOWR(VIRTIO_IOC_ID,103,VirtIOArg)
#define VIRTIO_IOC_CUBLAS_SGEMM \
	_IOWR(VIRTIO_IOC_ID,104,VirtIOArg)
#define VIRTIO_IOC_CUBLAS_DGEMM \
	_IOWR(VIRTIO_IOC_ID,105,VirtIOArg)
#define VIRTIO_IOC_CUBLAS_SETSTREAM \
	_IOWR(VIRTIO_IOC_ID,106,VirtIOArg)
#define VIRTIO_IOC_CUBLAS_GETSTREAM \
	_IOWR(VIRTIO_IOC_ID,107,VirtIOArg)
#define VIRTIO_IOC_CUBLAS_SASUM \
	_IOWR(VIRTIO_IOC_ID,108,VirtIOArg)
#define VIRTIO_IOC_CUBLAS_DASUM \
	_IOWR(VIRTIO_IOC_ID,109,VirtIOArg)

#define VIRTIO_IOC_CUBLAS_SAXPY \
	_IOWR(VIRTIO_IOC_ID,110,VirtIOArg)
#define VIRTIO_IOC_CUBLAS_DAXPY \
	_IOWR(VIRTIO_IOC_ID,111,VirtIOArg)
#define VIRTIO_IOC_CUBLAS_SCOPY \
	_IOWR(VIRTIO_IOC_ID,112,VirtIOArg)
#define VIRTIO_IOC_CUBLAS_DCOPY \
	_IOWR(VIRTIO_IOC_ID,113,VirtIOArg)
#define VIRTIO_IOC_CUBLAS_SGEMV \
	_IOWR(VIRTIO_IOC_ID,114,VirtIOArg)
#define VIRTIO_IOC_CUBLAS_DGEMV \
	_IOWR(VIRTIO_IOC_ID,115,VirtIOArg)
#define VIRTIO_IOC_CUBLAS_SDOT \
	_IOWR(VIRTIO_IOC_ID,116,VirtIOArg)
#define VIRTIO_IOC_CUBLAS_DDOT \
	_IOWR(VIRTIO_IOC_ID,117,VirtIOArg)
#define VIRTIO_IOC_CUBLAS_SSCAL \
	_IOWR(VIRTIO_IOC_ID,118,VirtIOArg)
#define VIRTIO_IOC_CUBLAS_DSCAL \
	_IOWR(VIRTIO_IOC_ID,119,VirtIOArg)
#define VIRTIO_IOC_CUBLAS_SETMATRIX \
	_IOWR(VIRTIO_IOC_ID,120,VirtIOArg)

#define VIRTIO_IOC_CUBLAS_GETMATRIX \
	_IOWR(VIRTIO_IOC_ID,121,VirtIOArg)
// curand-------------------------------
#define VIRTIO_IOC_CURAND_CREATEGENERATOR \
	_IOWR(VIRTIO_IOC_ID,200,VirtIOArg)
#define VIRTIO_IOC_CURAND_GENERATE \
	_IOWR(VIRTIO_IOC_ID,201,VirtIOArg)
#define VIRTIO_IOC_CURAND_GENERATENORMAL \
	_IOWR(VIRTIO_IOC_ID,202,VirtIOArg)
#define VIRTIO_IOC_CURAND_GENERATENORMALDOUBLE \
	_IOWR(VIRTIO_IOC_ID,203,VirtIOArg)
#define VIRTIO_IOC_CURAND_GENERATEUNIFORM \
	_IOWR(VIRTIO_IOC_ID,204,VirtIOArg)
#define VIRTIO_IOC_CURAND_GENERATEUNIFORMDOUBLE \
	_IOWR(VIRTIO_IOC_ID,205,VirtIOArg)
#define VIRTIO_IOC_CURAND_DESTROYGENERATOR \
	_IOWR(VIRTIO_IOC_ID,206,VirtIOArg)
#define VIRTIO_IOC_CURAND_SETGENERATOROFFSET \
	_IOWR(VIRTIO_IOC_ID,207,VirtIOArg)
#define VIRTIO_IOC_CURAND_SETPSEUDORANDOMSEED \
	_IOWR(VIRTIO_IOC_ID,208,VirtIOArg)
#define VIRTIO_IOC_CURAND_CREATEGENERATORHOST \
	_IOWR(VIRTIO_IOC_ID,209,VirtIOArg)
#endif

