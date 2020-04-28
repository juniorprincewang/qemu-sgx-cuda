/*
 * Virtio Console and Generic Serial Port Devices
 *
 * Copyright Red Hat, Inc. 2009, 2010
 *
 * Authors:
 *  Amit Shah <amit.shah@redhat.com>
 *
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "chardev/char-fe.h"
#include "qemu/error-report.h"
#include "trace.h"
#include "hw/virtio/virtio-serial.h"
#include "hw/virtio/virtio-access.h"
#include "qapi/error.h"
#include "qapi-event.h"
#include "exec/cpu-common.h"    // cpu_physical_memory_rw

#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h> //PATH: /usr/local/cuda/include/builtin_types.h
// #include <driver_types.h>   // enum cudaDeviceProp, enum cudaFuncCache
#include <cublas_v2.h>
#include <curand.h>
#include "virtio-ioc.h"
#include <openssl/hmac.h> // hmac EVP_MAX_MD_SIZE
/*Encodes Base64 */
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/buffer.h>
#include <stdint.h> // uint32_t ...
#include <limits.h> // CHAR_BIT , usually 8
/* SGX */
#include "hw/virtio/service_provider.h"
#include <sgx_key_exchange.h>
// #include <sgx_report.h>
// #include "hexutil.h"
// #include "crypto.h"
// #include "settings.h"
// #include "iasrequest.h"
// #include <openssl/pem.h>

#ifndef min
#define min(a,b)    (((a)<(b)) ? (a) : (b))
#endif

// #define VIRTIO_CUDA_DEBUG
#ifdef VIRTIO_CUDA_DEBUG
    #define func() printf("[FUNC]%s\n",__FUNCTION__)
    #define debug(fmt, arg...) printf("[DEBUG] " fmt, ##arg)
    #define print(fmt, arg...) printf("" fmt, ##arg)
    #define DPRINT_TAGS(tags)  do{  \
                                    printf("payload_tag\n"); \
                                    for (int idx=0; idx<SAMPLE_SP_TAG_SIZE; idx++) { \
                                        printf("%x ", tags[idx]); \
                                    } \
                                    printf("\n\n"); \
                                }while(0);
#else
    #define func()   
    #define debug(fmt, arg...)   
    #define print(fmt, arg...) 
    #define DPRINT_TAGS(tags)   
#endif

#define error(fmt, arg...) printf("[ERROR]In file %s, line %d, "fmt, \
            __FILE__, __LINE__, ##arg)


#define TYPE_VIRTIO_CONSOLE_SERIAL_PORT "virtcudaport"
#define VIRTIO_CONSOLE(obj) \
    OBJECT_CHECK(VirtConsole, (obj), TYPE_VIRTIO_CONSOLE_SERIAL_PORT)

#ifndef WORKER_THREADS
#define WORKER_THREADS 32+1
#endif

typedef struct VirtConsole {
    VirtIOSerialPort parent_obj;
    char *privatekeypath;    
    char *hmac_path;
    CharBackend chr;
    guint watch;
} VirtConsole;

typedef struct KernelConf {
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem;
    cudaStream_t stream;
} KernelConf_t ;

static int total_device;   // total GPU device
static int total_port;     // port count
static QemuMutex total_port_mutex;

static int global_initialized = 0;
static int global_deinitialized = 0;


#define cuErrorExit(call) { \
    cudaError_t err; \
    if ( (err = (call)) != cudaSuccess) { \
        char *str; \
        cuGetErrorName(err, (const char**)&str); \
        fprintf(stderr, "Got error %s at %s:%d\n", str, \
                __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define cudaError(err) __cudaErrorCheck(err, __LINE__)
static inline void __cudaErrorCheck(cudaError_t err, const int line)
{
    if (err != cudaSuccess) {
        char *str = (char*)cudaGetErrorString(err);
        error("CUDA Runtime API Error = %04d \"%s\" line %d\n", err, str, line);
    }
}

#define cuError(err) __cuErrorCheck(err, __LINE__)
static inline void __cuErrorCheck(cudaError_t err, const int line)
{
    char *str;
    if (err != cudaSuccess) {
        cuGetErrorName(err, (const char**)&str);
        error("CUDA Driver API Error = %04d \"%s\" line %d\n", err, str, line);
    }
}

#define cublasCheck(fn) { \
        cublasStatus_t __err = fn; \
        if (__err != CUBLAS_STATUS_SUCCESS) { \
            error("Fatal cublas error: %d (at %s:%d)\n", \
                (int)(__err), __FILE__, __LINE__); \
        } \
}

#define curandCheck(fn) { \
        curandStatus_t __err = fn; \
        if (__err != CURAND_STATUS_SUCCESS) { \
            error("Fatal curand error: %d (at %s:%d)\n", \
                (int)(__err), __FILE__, __LINE__); \
        } \
}

#define execute_with_context(call, context) {\
    cuError(cuCtxSetCurrent(context));\
    cudaError( (call) );\
}

    // cuError(cuCtxPushCurrent(context));
    // cuError(cuCtxPopCurrent(&context));

#define execute_with_cu_context(call, context) {\
    cuError(cuCtxSetCurrent(context));\
    cuError( (call) );\
}

#define HMAC_SHA256_SIZE 32 // hmac-sha256 output size is 32 bytes
#define HMAC_SHA256_COUNT (1<<10)
#define HMAC_SHA256_BASE64_SIZE 44 //HMAC_SHA256_SIZE*4/3

/*
static const char key[] = {
    0x30, 0x81, 0x9f, 0x30, 0x0d, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 
    0x0d, 0x01, 0x01, 0x01, 0x05, 0xa0, 0x03, 0x81, 0x8d, 0x44, 0x30, 0x81, 
    0x89, 0x02, 0x81, 0x81, 0x0b, 0xbb, 0xbd, 0xba, 0x9a, 0x8c, 0x3c, 0x38, 
    0xa3, 0xa8, 0x09, 0xcb, 0xc5, 0x2d, 0x98, 0x86, 0xe4, 0x72, 0x99, 0xe4,
    0x3b, 0x72, 0xb0, 0x73, 0x8a, 0xac, 0x12, 0x74, 0x99, 0xa7, 0xf4, 0xd1, 
    0xf9, 0xf4, 0x22, 0xeb, 0x61, 0x7b, 0xf5, 0x11, 0xd6, 0x9b, 0x02, 0x8e, 
    0xb4, 0x59, 0xb0, 0xb5, 0xe5, 0x11, 0x80, 0xb6, 0xe3, 0xec, 0x3f, 0xd6,
    0x1a, 0xe3, 0x4b, 0x18, 0xe7, 0xda, 0xff, 0x6b, 0xec, 0x7b, 0x71, 0xb6,
    0x78, 0x79, 0xc7, 0x97, 0x90, 0x81, 0xf2, 0xbb, 0x91, 0x5f, 0xd7, 0xc1, 
    0x97, 0xf2, 0xa0, 0xc0, 0x25, 0x6b, 0xd8, 0x96, 0x84, 0xb9, 0x49, 0xba, 
    0xe9, 0xb0, 0x50, 0x78, 0xfe, 0x57, 0x78, 0x1a, 0x2d, 0x75, 0x1e, 0x1c, 
    0xbd, 0x7d, 0xfc, 0xb8, 0xf6, 0x22, 0xbc, 0x20, 0xdd, 0x3e, 0x32, 0x75, 
    0x41, 0x63, 0xdd, 0xb2, 0x94, 0xc1, 0x29, 0xcc, 0x5e, 0x15, 0xb7, 0x1c, 
    0x0f, 0x02, 0x03, 0x01, 0x80, 0x01, 0x00
};

*/

int key_len =  162;
unsigned int result_len;
unsigned char result[EVP_MAX_MD_SIZE];
char *hmac_list = NULL;
unsigned long long int hmac_count = 0;
time_t oldMTime; // 判断hmac.txt是否更新

/*
static void hmac_encode( const char * keys, unsigned int key_length,  
                const char * input, unsigned int input_length,  
                unsigned char *output, unsigned int *output_length) 
{
    const EVP_MD * engine = EVP_sha256();  
    HMAC_CTX ctx;  
    func();

    HMAC_CTX_init(&ctx);  
    HMAC_Init_ex(&ctx, keys, key_length, engine, NULL);  
    HMAC_Update(&ctx, (unsigned char*)input, input_length); 

    HMAC_Final(&ctx, output, output_length);  
    HMAC_CTX_cleanup(&ctx);  
}
*/

// from https://gist.github.com/barrysteyn/7308212
//Encodes a binary safe base 64 string
/*
static int base64_encode(const unsigned char* buffer, size_t length, char** b64text) 
{
    func();
    BIO *bio, *b64;
    BUF_MEM *bufferPtr;

    b64 = BIO_new(BIO_f_base64());
    bio = BIO_new(BIO_s_mem());
    bio = BIO_push(b64, bio);

    //Ignore newlines - write everything in one line
    BIO_set_flags(bio, BIO_FLAGS_BASE64_NO_NL); 

    BIO_write(bio, buffer, length);
    BIO_flush(bio);
    BIO_get_mem_ptr(bio, &bufferPtr);
    BIO_set_close(bio, BIO_NOCLOSE);
    BIO_free_all(bio);

    *b64text=(*bufferPtr).data;
    return 0; //success
}
*/


/*
* 检查传入代码经过base64编码后的hmac值是否合法
*/
/*
static int check_hmac(const char * buffer, unsigned int buffer_len)
{
    int i =0;
    char* base64_encode_output;
    unsigned char * hmac = NULL;  
    unsigned int hmac_len = 0;  

    func();
    hmac = (unsigned char*)malloc(EVP_MAX_MD_SIZE);// EVP_MAX_MD_SIZE=64
    if (NULL == hmac) {
        error("Can't malloc %d\n", EVP_MAX_MD_SIZE);
        return 0;
    }
    hmac_encode(key, key_len, buffer, buffer_len, hmac, &hmac_len);

    if(0 == hmac_len) {
        error("HMAC encode failed!\n");
        return 0;
    } else {
        debug("HMAC encode succeeded!\n");
        debug("hmac length is %d\n", hmac_len);
        debug("hmac is :");
        
        // for(int i = 0; i < hmac_len; i++)
        // {
        //     printf("\\x%-02x", (unsigned int)hmac[i]);
        // }
        // printf("\n");
        
    }
    base64_encode(hmac, hmac_len, &base64_encode_output);
    printf("Output (base64): %s\n", base64_encode_output);
    if(hmac)
        free(hmac);
    // compare with the list
    for(i=0; i < (hmac_count/(HMAC_SHA256_BASE64_SIZE+1)); i++) {
        if ( 0 == memcmp((void *)base64_encode_output, 
            (void *)(hmac_list+i*(HMAC_SHA256_BASE64_SIZE+1)), 
            HMAC_SHA256_BASE64_SIZE) ) {
            debug("Found '%s' in hmac list.\n", base64_encode_output);
            return 1;
        }
    }
    debug("Not Found '%s' in hmac list.\n", base64_encode_output);
    return 0;
}
*/


static int get_active_ports_nr(VirtIOSerial *vser)
{
    VirtIOSerialPort *port;
    uint32_t nr_active_ports = 0;
    QTAILQ_FOREACH(port, &vser->ports, next) {
        nr_active_ports++;
    }
    return nr_active_ports;
}

static VOL *find_vol_by_vaddr(uint64_t vaddr, struct list_head *header);
static uint64_t map_device_addr_by_vaddr(uint64_t vaddr, struct list_head *header);
static int set_shm(size_t size, char *file_path);
/*
* bitmap
#include <limits.h> // for CHAR_BIT
*/
static void __set_bit(word_t *word, int n)
{
    word[WORD_OFFSET(n)] |= (1 << BIT_OFFSET(n));
}

static void __clear_bit(word_t *word, int n)
{
    word[WORD_OFFSET(n)] &= ~(1 << BIT_OFFSET(n));
}

static int __get_bit(word_t *word, int n)
{
    word_t bit = word[WORD_OFFSET(n)] & (1 << BIT_OFFSET(n));
    return bit != 0;
}

static void *gpa_to_hva(hwaddr gpa, int len)
{
    hwaddr size = (hwaddr)len;
    void *hva = cpu_physical_memory_map(gpa, &size, 0);
    if (!hva || len != size) {
        error("Failed to map MMIO memory for"
                          " gpa 0x%lx element size %u\n",
                            gpa, len);
        return NULL;
    }
    return hva;
}

static void deinit_primary_context(CudaContext *ctx)
{
    ctx->dev         = 0;
    ctx->moduleCount = 0;
    ctx->initialized = 0;
    memset(ctx->modules, 0, sizeof(ctx->modules));
}

static int cmp_mac(uint8_t *a, uint8_t *b, int n) {
    for (int i=0; i< n; i++) {
        if(a[i] != b[i]) {
            error("binary mac does not match!"
                " where a[%d]=%x != b[%d]=%x\n", i, a[i], i, b[i]);
            return 1;
        }
    }
    return 0;
}

typedef struct fatbin_buf
{
    uint32_t block_size;
    uint32_t total_size;
    uint32_t size;
    uint32_t nr_binary;
    uint8_t buf[];
} fatbin_buf_t;

typedef struct binary_buf
{
    uint32_t size;
    uint32_t nr_var;
    uint32_t nr_func;
#ifdef ENABLE_MAC
    uint8_t payload_tag[SAMPLE_SP_TAG_SIZE];
#endif
    uint8_t buf[];
} binary_buf_t;

typedef struct function_buf
{
    uint64_t hostFun;
    uint32_t size;
    uint8_t buf[];
} function_buf_t;

typedef struct var_buf
{
    uint64_t hostVar;
    int constant;
    int global;
    uint32_t size;
    uint8_t buf[];
} var_buf_t;

static void primarycontext(VirtQueueElement *elem, ThreadContext *tctx)
{
    fatbin_buf_t *fat_bin;
    VirtIOArg *arg;
    int nr, j;
    uint32_t offset = 0;
    CudaContext *ctx        = &tctx->contexts[DEFAULT_DEVICE];

    func();
#ifndef PRE_INIT_CTX
    cuError(cuCtxCreate(&ctx->context, 0, ctx->dev));
#endif
    arg = elem->out_sg[0].iov_base;
    fat_bin = (fatbin_buf_t *)(elem->out_sg[1].iov_base);
    debug("fat_bin addr %p\n", fat_bin);
    debug("nr_binary %x\n", fat_bin->nr_binary);
#ifdef ENABLE_MAC
    sp_aes_gcm_data_t *decrypt_msg;
    uint8_t *mac;
    /*arg mac*/
    sp_ra_mac_req(&tctx->g_sp_db, (uint8_t *)arg, sizeof(VirtIOArg)-SAMPLE_SP_TAG_SIZE, &decrypt_msg);
    DPRINT_TAGS(decrypt_msg->payload_tag);
    DPRINT_TAGS(arg->mac);
    if (cmp_mac(decrypt_msg->payload_tag, arg->mac, SAMPLE_SP_TAG_SIZE)) {
        free(decrypt_msg);
        arg->cmd = cudaErrorInitializationError;
        return ;
    }
    free(decrypt_msg);
    /*binary buffer*/
    sp_ra_mac_req(&tctx->g_sp_db, (uint8_t *)fat_bin, arg->srcSize-SAMPLE_SP_TAG_SIZE, &decrypt_msg);
    mac = (uint8_t *)fat_bin + arg->srcSize-SAMPLE_SP_TAG_SIZE;
    DPRINT_TAGS(decrypt_msg->payload_tag);
    DPRINT_TAGS(mac);
    if (cmp_mac(decrypt_msg->payload_tag, mac, SAMPLE_SP_TAG_SIZE)) {
        free(decrypt_msg);
        arg->cmd = cudaErrorInitializationError;
        return ;
    }
    free(decrypt_msg);
#endif
    cuError(cuCtxSetCurrent(ctx->context));
    for(nr=0; nr<fat_bin->nr_binary; nr++) {
        binary_buf_t *p_binary = (void*)fat_bin->buf + offset;
        ctx->moduleCount++;
        if (ctx->moduleCount > CudaModuleMaxNum-1) {
            error("Fatbinary number is overflow.\n");
            arg->cmd = cudaErrorUnknown;
            return;
        }
        if(p_binary->nr_func >= CudaFunctionMaxNum) {
            error("kernel number is overflow.\n");
            arg->cmd = cudaErrorUnknown;
            return;
        }
        if(p_binary->nr_var >= CudaVariableMaxNum) {
            error("var number is overflow.\n");
            arg->cmd = cudaErrorUnknown;
            return;
        }
        CudaModule *cuda_module = &ctx->modules[nr];
        cuda_module->handle              = (size_t)arg->src;
        cuda_module->fatbin_size         = p_binary->size;
        cuda_module->cudaKernelsCount    = p_binary->nr_func;
        cuda_module->cudaVarsCount       = p_binary->nr_var;
// #ifdef ENABLE_ENC
//         cuda_module->fatbin              = malloc(p_binary->size);
//         debug("arg payload_tag\n");
//         for (i=0; i<SAMPLE_SP_TAG_SIZE; i++) {
//             print("%x ", p_binary->payload_tag[i]);
//         }
//         print("\n\n");
//         if(sp_ra_decrypt_req(&tctx->g_sp_db, p_binary->buf, p_binary->size, cuda_module->fatbin, p_binary->payload_tag)) {
//             error("decrypt failed.\n");
//             arg->cmd = cudaErrorUnknown;
//             return;
//         }
// #else
        cuda_module->fatbin              = p_binary->buf;
// #endif
        offset += p_binary->size + sizeof(binary_buf_t);
        cuError(cuModuleLoadData(&cuda_module->module, cuda_module->fatbin));
        debug("Loading module... #%d, fatbin = 0x%lx, fatbin size=0x%x\n", 
            nr, cuda_module->handle, cuda_module->fatbin_size);
        // functions
        debug("nr_func %x\n", p_binary->nr_func);
        for(j=0; j<p_binary->nr_func; j++) {
            function_buf_t *p_func  = (void*)fat_bin->buf + offset;
            CudaKernel *kernel      = &cuda_module->cudaKernels[j];
            // kernel->func_name       = p_func->buf;
            kernel->func_name       = malloc(p_func->size);
            memcpy(kernel->func_name, p_func->buf, p_func->size);
            kernel->func_name_size  = p_func->size;
            kernel->func_id         = p_func->hostFun;
            offset += p_func->size + sizeof(function_buf_t);
            cuError(cuModuleGetFunction(&kernel->kernel_func, cuda_module->module, kernel->func_name));
            debug("func %d, name='%s', name size=0x%x, func_id=0x%lx\n",
                j, kernel->func_name, kernel->func_name_size, kernel->func_id);
        }
        // var
        debug("nr_var %x\n", p_binary->nr_var);
        for(j=0; j<p_binary->nr_var; j++) {
            var_buf_t *p_var    = (void*)fat_bin->buf + offset;
            CudaMemVar *var     = &cuda_module->cudaVars[j];
            // var->addr_name      = p_var->buf;
            var->addr_name_size = p_var->size;
            var->addr_name = malloc(p_var->size);
            memcpy(var->addr_name, p_var->buf, p_var->size);
            var->global         = p_var->global ? 1 : 0;
            var->host_var       = p_var->hostVar;
            offset += p_var->size + sizeof(var_buf_t);
            cuError(cuModuleGetGlobal(&var->device_ptr, &var->mem_size, cuda_module->module, var->addr_name));
            debug("var %d, name='%s', name size=0x%x, host_var=0x%lx, global =%d\n", 
                j, var->addr_name, var->addr_name_size, var->host_var, var->global);
        }
    }
    ctx->initialized = 1;
    ctx->tctx->deviceBitmap |= 1<< ctx->tctx->cur_dev;
    arg->cmd = cudaSuccess;
}

static void cuda_register_fatbinary(VirtIOArg *arg, ThreadContext *tctx)
{
}

/*
 * unused
 */
static void cuda_unregister_fatbinary(VirtIOArg *arg, ThreadContext *tctx)
{
}

static void cuda_register_function(VirtIOArg *arg, ThreadContext *tctx)
{
}

static void cuda_register_var(VirtIOArg *arg, ThreadContext *tctx)
{
}

static void cuda_set_device(VirtIOArg *arg, ThreadContext *tctx)
{
}

static void cu_set_device(VirtQueueElement *elem, ThreadContext *tctx)
{
    CudaContext *ctx = NULL;
    cudaError_t err = -1;
    VirtIOArg *arg = NULL;
    arg = elem->out_sg[0].iov_base;

    func();
    int dev_id = (int)(arg->flag);
    debug("set devices=%d\n", dev_id);
    if (dev_id < 0 || dev_id > tctx->deviceCount-1) {
        error("setting error device = %d\n", dev_id);
        arg->cmd = cudaErrorInvalidDevice;
        return ;
    }
    if (! (tctx->deviceBitmap & 1<<dev_id)) {
        tctx->cur_dev = dev_id;
        ctx = &tctx->contexts[dev_id];
        memcpy(ctx->modules, &tctx->contexts[DEFAULT_DEVICE].modules,
                sizeof(ctx->modules));
        cuError(cuDeviceGet(&ctx->dev, dev_id));
        // init_device_module(ctx);
    }
    cudaError(err = cudaSetDevice(dev_id));
    arg->cmd = err;
    /* clear kernel function addr in parent process,
    because cudaSetDevice will clear all resources related with host thread.
    */
}

static void init_device_module(CudaContext *ctx)
{
    int i=0, j=0;
    CudaModule *module = NULL;
    CudaKernel *kernel = NULL;
    CudaMemVar *var = NULL;
    debug("sub module number = %d\n", ctx->moduleCount);
    for (i=0; i < ctx->moduleCount; i++) {
        module = &ctx->modules[i];
        if (!module)
            return;
        cuError(cuModuleLoadData(&module->module, module->fatbin));
        debug("kernel count = %d\n", module->cudaKernelsCount);
        for(j=0; j<module->cudaKernelsCount; j++) {
            kernel = &module->cudaKernels[j];
            cuError(cuModuleGetFunction(&kernel->kernel_func, module->module, kernel->func_name));
        }
        debug("var count = %d\n", module->cudaVarsCount);
        for(j=0; j<module->cudaVarsCount; j++) {
            var = &module->cudaVars[j];
            cuError(cuModuleGetGlobal(&var->device_ptr, &var->mem_size, module->module, var->addr_name));
        }
    }
}

static void init_primary_context(CudaContext *ctx)
{
    if(!ctx->initialized) {
        // cuErrorExit(cuDeviceGet(&ctx->dev, ctx->tctx->cur_dev));
        // cuErrorExit(cuDevicePrimaryCtxRetain(&ctx->context, ctx->dev));
        // cuErrorExit(cuCtxCreate(&ctx->context, 0, ctx->dev));
#ifndef PRE_INIT_CTX
        cuError(cuCtxCreate(&ctx->context, 0, ctx->dev));
#endif
        cuError(cuCtxSetCurrent(ctx->context));
        ctx->initialized = 1;
        ctx->tctx->deviceBitmap |= 1<< ctx->tctx->cur_dev;
        init_device_module(ctx);
    }
}

static void cuda_set_device_flags(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err = -1;
    // CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    func();
    unsigned int flags = (unsigned int)arg->flag;
    debug("set devices flags=%d\n", flags);
    cudaError(err = cudaSetDeviceFlags(flags));
    if (err == cudaErrorSetOnActiveProcess) 
        err = cudaSuccess;
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("set device flags error.\n");
    }
}

static void cuda_device_set_cache_config(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err = -1;
    // CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    func();
    enum cudaFuncCache cacheConfig = (enum cudaFuncCache)arg->flag;
    debug("set devices cache config=%d\n", cacheConfig);
    cudaError(err = cudaDeviceSetCacheConfig(cacheConfig));
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("set device flags error.\n");
    }
}

static void cuda_launch(VirtIOArg *arg, VirtIOSerialPort *port)
{
}

static void cu_launch_kernel(VirtQueueElement *elem, VirtIOSerialPort *port)
{
    cudaError_t err=0;
    uint32_t para_num=0;
    int i = 0;
    int j = 0;
    ThreadContext *tctx = port->thread_context;
    CudaContext *ctx    = &tctx->contexts[tctx->cur_dev];
    int m_num           = ctx->moduleCount;
    CudaModule *cuda_module = NULL;
    CudaKernel *kernel  = NULL;
    size_t func_handle  = 0;
    void **para_buf     = NULL;
    uint32_t *offset_buf = NULL;
    VirtIOArg *arg = NULL;
    CUkernel_st *para;

    
    func();
    init_primary_context(ctx);
    arg = elem->out_sg[0].iov_base;
#ifdef ENABLE_ENC
    DPRINT_TAGS(arg->mac);
    para = malloc(elem->out_sg[1].iov_len);
    if(sp_ra_decrypt_req(&tctx->g_sp_db, elem->out_sg[1].iov_base, 
                        elem->out_sg[1].iov_len, 
                        (uint8_t *)para,
                        (uint8_t *)arg, elem->out_sg[0].iov_len - SAMPLE_SP_TAG_SIZE,
                        arg->mac)) {
        error("decrypt failed.\n");
        arg->cmd = cudaErrorUnknown;
        return;
    }
#else
    para = (CUkernel_st *)elem->out_sg[1].iov_base;
    if (!para ) {
        arg->cmd = cudaErrorInvalidConfiguration;
        error("Invalid para configure.\n");
        return ;
    }
#endif

    func_handle = (size_t)arg->flag;
    debug(" func_id = 0x%lx\n", func_handle);

    for (i=0; i < m_num; i++) {
        cuda_module = &ctx->modules[i];
        for (j=0; j < cuda_module->cudaKernelsCount; j++) {
            if (cuda_module->cudaKernels[j].func_id == func_handle) {
                kernel = &cuda_module->cudaKernels[j];
                debug("Found func_id\n");
                break;
            }
        }
    }
    if (!kernel) {
        error("Failed to find func id %lx.\n", func_handle);
        arg->cmd = cudaErrorInvalidDeviceFunction;
        return;
    }


    debug("gridDim=%u %u %u\n", para->grid_x, 
          para->grid_y, para->grid_z);
    debug("blockDim=%u %u %u\n", para->block_x,
          para->block_y, para->block_z);
    debug("sharedMem=%d\n", para->smem_size);
    debug("stream=0x%lx\n", (uint64_t)(para->stream));
    debug("param_nr=0x%x\n", para->param_nr);
    debug("param_size=0x%x\n", para->param_size);

    para_num = para->param_nr;
    para_buf = malloc(para_num * sizeof(void*));
    offset_buf = (uint32_t *)(para->param_buf+para->param_size);
    for(i=0; i<para_num; i++) {
        para_buf[i] = para->param_buf+offset_buf[i];
        debug("arg %d = %p, offset %x\n", i, para_buf[i], offset_buf[i]);
    }

    cuError(cuCtxSetCurrent(ctx->context));
    cuError(cuLaunchKernel( kernel->kernel_func,
                            para->grid_x, para->grid_y, para->grid_z,
                            para->block_x, para->block_y, para->block_z,
                            para->smem_size,
                            para->stream,
                            para_buf, NULL));
    cudaError(err = cudaGetLastError());
    arg->cmd = err;
#ifdef ENABLE_ENC
    free(para);
#endif
    free(para_buf);
}


static VOL *find_vol_by_vaddr(uint64_t vaddr, struct list_head *header)
{
    VOL *vol;
    list_for_each_entry(vol, header, list) {
        if(vol->v_addr <= vaddr && vaddr < (vol->v_addr+vol->size) )
            goto out;
    }
    vol = NULL;
out:
    return vol;
}

static HVOL *find_hvol_by_vaddr(uint64_t vaddr, struct list_head *header)
{
    HVOL *hvol;
    list_for_each_entry(hvol, header, list) {
        if(hvol->virtual_addr <= vaddr && vaddr < (hvol->virtual_addr+hvol->size) )
            goto out;
    }
    hvol = NULL;
out:
    return hvol;
}

static uint64_t map_device_addr_by_vaddr(uint64_t vaddr, struct list_head *header)
{
    VOL *vol = find_vol_by_vaddr(vaddr, header);
    if(vol != NULL)
        return vol->addr + (vaddr - vol->v_addr);
    return 0;
}

static uint64_t map_host_addr_by_vaddr(uint64_t vaddr, struct list_head *header)
{
    HVOL *hvol = find_hvol_by_vaddr(vaddr, header);
    if(hvol != NULL)
        return hvol->addr + (vaddr - hvol->virtual_addr);
    return 0;
}

static void remove_hvol_by_vaddr(uint64_t vaddr, struct list_head *header)
{
    HVOL *hvol, *hvol2;
    list_for_each_entry_safe(hvol, hvol2, header, list) {
        if (hvol->virtual_addr == vaddr) {
            debug("Found memory maped ptr=0x%lx\n", (uint64_t)hvol->addr);
            munmap((void*)hvol->addr, hvol->size);
            close(hvol->fd);
            list_del(&hvol->list);
            free(hvol);
            return;
        }
    }
    error("Found no memory maped ptr=0x%lx\n", vaddr);
}

static void remove_plvol_by_vaddr(uint64_t vaddr, struct list_head *header, CudaContext * ctx)
{
    HPLOL *plvol, *plvol2;
    list_for_each_entry_safe(plvol, plvol2, header, list) {
        if (plvol->virtual_addr == vaddr) {
            debug("Found memory(page) maped ptr=0x%lx\n", (uint64_t)plvol->virtual_addr);
            for(int i=0; i<plvol->blocks; i++) {
                execute_with_context(cudaHostUnregister((void *)plvol->addr[i]), ctx->context);
            }
            free(plvol->addr);
            list_del(&plvol->list);
            free(plvol);
            return;
        }
    }
    error("Found no memory(page) maped ptr=0x%lx\n", vaddr);
}

static void cuda_memcpy_safe(VirtIOArg *arg, ThreadContext *tctx)
{
    return;
}

static void cuda_memcpy(VirtIOArg *arg, ThreadContext *tctx)
{
    return;
}

static void cu_memcpy_htod(VirtQueueElement *elem, ThreadContext *tctx)
{
    CUresult err=0;
    size_t size = 0;
    CUdeviceptr dev;
    unsigned long offset = 0;
    int i=0;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    init_primary_context(ctx);
#ifdef ENABLE_ENC
    /*arg mac*/
    sp_aes_gcm_data_t *decrypt_msg;
    sp_ra_mac_req(&tctx->g_sp_db, (uint8_t *)arg, sizeof(VirtIOArg)-SAMPLE_SP_TAG_SIZE, &decrypt_msg);
    DPRINT_TAGS(decrypt_msg->payload_tag);
    DPRINT_TAGS(arg->mac);
    if (cmp_mac(decrypt_msg->payload_tag, arg->mac, SAMPLE_SP_TAG_SIZE)) {
        free(decrypt_msg);
        arg->cmd = cudaErrorInitializationError;
        return ;
    }
    free(decrypt_msg);
    /* mac list*/
    uint8_t *mac_list = elem->out_sg[1].iov_base;
    // uint32_t mac_size = elem->out_sg[1].iov_len;
    // debug("mac nr 0x%x, elem->out_num 0x%x\n", mac_size / SAMPLE_SP_TAG_SIZE, elem->out_num);
    size = arg->srcSize;
    uint8_t *p_host = malloc(size);
    offset=0;
    unsigned long in_offset = 0;
    unsigned long size_left = 0, block_size = 0;
    int idx=0;
    for(i=2; i < elem->out_num; i++) {
        /*determine if continued chunk*/
        size_left = elem->out_sg[i].iov_len;
        debug("elem 0x%x : size 0x%lx\n", i, size_left);
        in_offset = 0;
        while(size_left) {
            block_size = (size_left > CHUNK_SIZE) ? CHUNK_SIZE : size_left;
            if(sp_ra_decrypt_req(&tctx->g_sp_db, elem->out_sg[i].iov_base + in_offset, block_size,
                                p_host+offset+in_offset,
                                NULL, 0,
                                mac_list+idx*SAMPLE_SP_TAG_SIZE )) {
                error("decrypt failed.\n");
                arg->cmd = cudaErrorUnknown;
                return;
            }
            idx += 1;
            in_offset += block_size;
            size_left -= block_size;
        }
        offset += in_offset;
    }
    debug("mac nr 0x%x\n", idx);
#endif
    size = arg->srcSize;
    dev = arg->dst;
    cuError(cuCtxSetCurrent(ctx->context));
#ifdef ENABLE_ENC
    cuError((err=cuMemcpyHtoD(dev, p_host, size)));
    if(err != CUDA_SUCCESS) {
        error("Failed to memcpy htod\n");
    }
    free(p_host);
#else
    void *va = NULL;
    for(i=1; i<elem->out_num; i++) {
        size = elem->out_sg[i].iov_len;
        va = elem->out_sg[i].iov_base;
        cuError((err=cuMemcpyHtoD(dev+offset, va, size)));
        if(err != CUDA_SUCCESS) {
            error("Failed to memcpy htod\n");
            break;
        }
        offset += size;
    }
#endif
    debug("tid = %d, src=0x%lx, srcSize=0x%x, dst=0x%lx, "
          "dstSize=0x%x, kind=0x%lx\n",
          arg->tid,  arg->src, arg->srcSize, arg->dst, 
          arg->dstSize, arg->flag);
    arg->cmd = err;
}

static void cu_memcpy_dtoh(VirtQueueElement *elem, ThreadContext *tctx)
{
    CUresult err=0;
    size_t size = 0;
    CUdeviceptr dev;
    unsigned long offset = 0;
    int i=0;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    VirtIOArg *arg = NULL;
    arg = elem->out_sg[0].iov_base;

    func();
    init_primary_context(ctx);
#ifdef ENABLE_ENC
    sp_aes_gcm_data_t *decrypt_msg;
    /*arg mac*/
    sp_ra_mac_req(&tctx->g_sp_db, (uint8_t *)arg, sizeof(VirtIOArg)-SAMPLE_SP_TAG_SIZE, &decrypt_msg);
    DPRINT_TAGS(decrypt_msg->payload_tag);
    DPRINT_TAGS(arg->mac);
    if (cmp_mac(decrypt_msg->payload_tag, arg->mac, SAMPLE_SP_TAG_SIZE)) {
        free(decrypt_msg);
        arg->cmd = cudaErrorInitializationError;
        return ;
    }
    free(decrypt_msg);
#endif
    debug("tid = %d, src=0x%lx, srcSize=0x%x, dst=0x%lx, "
          "dstSize=0x%x, kind=0x%lx, block_nr=0x%lx\n",
          arg->tid,  arg->src, arg->srcSize, arg->dst, 
          arg->dstSize, arg->flag, arg->param);
    dev = arg->src;
    offset = 0;
    cuError(cuCtxSetCurrent(ctx->context));
#ifdef ENABLE_ENC
    size = arg->srcSize;
    uint8_t *p_host = malloc(size);
    cuError((err=cuMemcpyDtoH(p_host, dev, size)));
    if(err != CUDA_SUCCESS) {
        error("Failed to memcpy dtoh\n");
        arg->cmd = err;
        memset(arg->mac, 0, SAMPLE_SP_TAG_SIZE);
        return;
    }
    arg->cmd = 0;
    uint8_t *mac_list = elem->in_sg[0].iov_base;
    // uint32_t mac_size = elem->in_sg[0].iov_len;
    // debug("mac nr 0x%x, elem->in_num 0x%x\n", mac_size / SAMPLE_SP_TAG_SIZE, elem->in_num);
    offset=0;
    unsigned long in_offset = 0;
    unsigned long size_left = 0, block_size = 0;
    int idx=0;
    for(i=1; i < elem->in_num; i++) {
        /*determine if continued chunk*/
        size_left = elem->in_sg[i].iov_len;
        debug("elem 0x%x : size 0x%lx\n", i, size_left);
        in_offset = 0;
        while(size_left) {
            block_size = (size_left > CHUNK_SIZE) ? CHUNK_SIZE : size_left;
            if(sp_ra_encrypt_req(&tctx->g_sp_db, p_host+offset+in_offset, block_size,
                                elem->in_sg[i].iov_base + in_offset, 
                                NULL, 0,
                                mac_list+idx*SAMPLE_SP_TAG_SIZE )) {
                error("decrypt failed.\n");
                arg->cmd = cudaErrorUnknown;
                return;
            }
            idx += 1;
            in_offset += block_size;
            size_left -= block_size;
        }
        offset += in_offset;
    }
    debug("mac nr 0x%x\n", idx);
    free(p_host);
#else
    void *va = NULL;
    for(i=0; i<elem->in_num; i++) {
        size = elem->in_sg[i].iov_len;
        va = elem->in_sg[i].iov_base;
        cuError((err=cuMemcpyDtoH(va, dev+offset, size)));
        if(err != CUDA_SUCCESS) {
            error("Failed to memcpy dtoh\n");
            break;
        }
        offset += size;
    }
    arg->cmd = err;
#endif
}

static void cu_memcpy_dtod(VirtQueueElement *elem, ThreadContext *tctx)
{
    CUresult err=0;
    CUdeviceptr src, dst;
    size_t size;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    VirtIOArg *arg = NULL;
    arg = elem->out_sg[0].iov_base;
    
    func();
#ifdef ENABLE_ENC
    sp_aes_gcm_data_t *decrypt_msg;
    /*arg mac*/
    sp_ra_mac_req(&tctx->g_sp_db, (uint8_t *)arg, sizeof(VirtIOArg)-SAMPLE_SP_TAG_SIZE, &decrypt_msg);
    DPRINT_TAGS(decrypt_msg->payload_tag);
    DPRINT_TAGS(arg->mac);
    if (cmp_mac(decrypt_msg->payload_tag, arg->mac, SAMPLE_SP_TAG_SIZE)) {
        free(decrypt_msg);
        arg->cmd = cudaErrorInitializationError;
        return ;
    }
    free(decrypt_msg);
#endif
    debug("tid = %d, src=0x%lx, srcSize=0x%x, dst=0x%lx, "
          "dstSize=0x%x, kind=0x%lx\n",
          arg->tid,  arg->src, arg->srcSize, arg->dst, 
          arg->dstSize, arg->flag);
    src = (CUdeviceptr)arg->src;
    dst = (CUdeviceptr)arg->dst;
    size = arg->srcSize;
    cuError(cuCtxSetCurrent(ctx->context));
    cuError((err=cuMemcpyDtoD(dst, src, size)));
    if(err != CUDA_SUCCESS) {
        error("Failed to memcpy dtod\n");
    }
    arg->cmd = err;
}

static void cu_memcpy_htod_async(VirtQueueElement *elem, ThreadContext *tctx)
{
    CUresult err=0;
    size_t size = 0;
    CUdeviceptr dev;
    unsigned long offset = 0;
    int i=0;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    VirtIOArg *arg = elem->out_sg[0].iov_base;
    CUstream stream;
    
    func();
#ifdef ENABLE_ENC
    /*arg mac*/
    sp_aes_gcm_data_t *decrypt_msg;
    sp_ra_mac_req(&tctx->g_sp_db, (uint8_t *)arg, sizeof(VirtIOArg)-SAMPLE_SP_TAG_SIZE, &decrypt_msg);
    DPRINT_TAGS(decrypt_msg->payload_tag);
    DPRINT_TAGS(arg->mac);
    if (cmp_mac(decrypt_msg->payload_tag, arg->mac, SAMPLE_SP_TAG_SIZE)) {
        free(decrypt_msg);
        arg->cmd = cudaErrorInitializationError;
        return ;
    }
    free(decrypt_msg);
    /* mac list*/
    uint8_t *mac_list = elem->out_sg[1].iov_base;
    // uint32_t mac_size = elem->out_sg[1].iov_len;
    // debug("mac nr 0x%x, elem->out_num 0x%x\n", mac_size / SAMPLE_SP_TAG_SIZE, elem->out_num);
    size = arg->srcSize;
    uint8_t *p_host = malloc(size);
    offset=0;
    unsigned long in_offset = 0;
    unsigned long size_left = 0, block_size = 0;
    int idx=0;
    for(i=2; i < elem->out_num; i++) {
        /*determine if continued chunk*/
        size_left = elem->out_sg[i].iov_len;
        debug("elem 0x%x : size 0x%lx\n", i, size_left);
        in_offset = 0;
        while(size_left) {
            block_size = (size_left > CHUNK_SIZE) ? CHUNK_SIZE : size_left;
            if(sp_ra_decrypt_req(&tctx->g_sp_db, elem->out_sg[i].iov_base + in_offset, block_size,
                                p_host+offset+in_offset,
                                NULL, 0,
                                mac_list+idx*SAMPLE_SP_TAG_SIZE )) {
                error("decrypt failed.\n");
                arg->cmd = cudaErrorUnknown;
                return;
            }
            idx += 1;
            in_offset += block_size;
            size_left -= block_size;
        }
        offset += in_offset;
    }
    debug("mac nr 0x%x\n", idx);
#endif
    debug("tid = %d, src=0x%lx, srcSize=0x%x, dst=0x%lx, "
          "dstSize=0x%x, kind=0x%lx, stream=0x%lx\n",
          arg->tid,  arg->src, arg->srcSize, arg->dst, 
          arg->dstSize, arg->flag, (uint64_t)arg->stream);
    size = arg->srcSize;
    dev = arg->dst;
    stream = (CUstream)arg->stream;
    cuError(cuCtxSetCurrent(ctx->context));
#ifdef ENABLE_ENC
    cuError((err=cuMemcpyHtoDAsync(dev, p_host, size, stream)));
    if(err != CUDA_SUCCESS) {
        error("Failed to memcpy htod\n");
    }
    free(p_host);
#else
    void *va = NULL;
    for(i=1; i<elem->out_num; i++) {
        size = elem->out_sg[i].iov_len;
        va = elem->out_sg[i].iov_base;
        cuError((err=cuMemcpyHtoDAsync(dev+offset, va, size, stream)));
        if(err != CUDA_SUCCESS) {
            error("Failed to memcpy htod async\n");
            break;
        }
        offset += size;
    }
#endif
    arg->cmd = err;
}

static void cu_memcpy_dtoh_async(VirtQueueElement *elem, ThreadContext *tctx)
{
    CUresult err=0;
    size_t size = 0;
    CUdeviceptr dev;
    unsigned long offset = 0;
    int i=0;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    VirtIOArg *arg = elem->out_sg[0].iov_base;
    CUstream stream;

    func();
#ifdef ENABLE_ENC
    sp_aes_gcm_data_t *decrypt_msg;
    /*arg mac*/
    sp_ra_mac_req(&tctx->g_sp_db, (uint8_t *)arg, sizeof(VirtIOArg)-SAMPLE_SP_TAG_SIZE, &decrypt_msg);
    DPRINT_TAGS(decrypt_msg->payload_tag);
    DPRINT_TAGS(arg->mac);
    if (cmp_mac(decrypt_msg->payload_tag, arg->mac, SAMPLE_SP_TAG_SIZE)) {
        free(decrypt_msg);
        arg->cmd = cudaErrorInitializationError;
        return ;
    }
    free(decrypt_msg);
#endif
    debug("tid = %d, src=0x%lx, srcSize=0x%x, dst=0x%lx, "
          "dstSize=0x%x, kind=0x%lx, stream=0x%lx\n",
          arg->tid,  arg->src, arg->srcSize, arg->dst, 
          arg->dstSize, arg->flag, (uint64_t)arg->stream);
    dev = (CUdeviceptr)arg->src;
    stream = (CUstream)arg->stream;
    cuError(cuCtxSetCurrent(ctx->context));
#ifdef ENABLE_ENC
    size = arg->srcSize;
    uint8_t *p_host = malloc(size);
    cuError((err=cuMemcpyDtoHAsync(p_host, dev, size, stream)));
    if(err != CUDA_SUCCESS) {
        error("Failed to memcpy dtoh\n");
        arg->cmd = err;
        memset(arg->mac, 0, SAMPLE_SP_TAG_SIZE);
        return;
    }
    arg->cmd = 0;
    uint8_t *mac_list = elem->in_sg[0].iov_base;
    // uint32_t mac_size = elem->in_sg[0].iov_len;
    // debug("mac nr 0x%x, elem->in_num 0x%x\n", mac_size / SAMPLE_SP_TAG_SIZE, elem->in_num);
    offset=0;
    unsigned long in_offset = 0;
    unsigned long size_left = 0, block_size = 0;
    int idx=0;
    for(i=1; i < elem->in_num; i++) {
        /*determine if continued chunk*/
        size_left = elem->in_sg[i].iov_len;
        debug("elem 0x%x : size 0x%lx\n", i, size_left);
        in_offset = 0;
        while(size_left) {
            block_size = (size_left > CHUNK_SIZE) ? CHUNK_SIZE : size_left;
            if(sp_ra_encrypt_req(&tctx->g_sp_db, p_host+offset+in_offset, block_size,
                                elem->in_sg[i].iov_base + in_offset, 
                                NULL, 0,
                                mac_list+idx*SAMPLE_SP_TAG_SIZE )) {
                error("decrypt failed.\n");
                arg->cmd = cudaErrorUnknown;
                return;
            }
            idx += 1;
            in_offset += block_size;
            size_left -= block_size;
        }
        offset += in_offset;
    }
    debug("mac nr 0x%x\n", idx);
    free(p_host);
#else
    void *va = NULL;
    offset=0;
    for(i=0; i<elem->in_num; i++) {
        size = elem->in_sg[i].iov_len;
        va = elem->in_sg[i].iov_base;
        // debug("in_sg len 0x%lx, data %p\n", size, va);
        cuError((err=cuMemcpyDtoHAsync(va, dev+offset, size, stream)));
        if(err != CUDA_SUCCESS) {
            error("Failed to memcpy dtoh async\n");
            break;
        }
        offset += size;
    }
    arg->cmd = err;
#endif
}

static void cu_memcpy_dtod_async(VirtQueueElement *elem, ThreadContext *tctx)
{
    CUresult err=0;
    CUdeviceptr src, dst;
    size_t size;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    VirtIOArg *arg = elem->out_sg[0].iov_base;
    CUstream stream;

    func();
#ifdef ENABLE_ENC
    sp_aes_gcm_data_t *decrypt_msg;
    /*arg mac*/
    sp_ra_mac_req(&tctx->g_sp_db, (uint8_t *)arg, sizeof(VirtIOArg)-SAMPLE_SP_TAG_SIZE, &decrypt_msg);
    DPRINT_TAGS(decrypt_msg->payload_tag);
    DPRINT_TAGS(arg->mac);
    if (cmp_mac(decrypt_msg->payload_tag, arg->mac, SAMPLE_SP_TAG_SIZE)) {
        free(decrypt_msg);
        arg->cmd = cudaErrorInitializationError;
        return ;
    }
    free(decrypt_msg);
#endif
    debug("tid = %d, src=0x%lx, srcSize=0x%x, dst=0x%lx, "
          "dstSize=0x%x, kind=0x%lx, stream=0x%lx\n",
          arg->tid,  arg->src, arg->srcSize, arg->dst, 
          arg->dstSize, arg->flag, (uint64_t)arg->stream);
    src = (CUdeviceptr)arg->src;
    dst = (CUdeviceptr)arg->dst;
    size = arg->srcSize;
    stream = (CUstream)arg->stream;
    cuError(cuCtxSetCurrent(ctx->context));
    cuError((err=cuMemcpyDtoDAsync(dst, src, size, stream)));
    if(err != CUDA_SUCCESS) {
        error("Failed to memcpy dtod async\n");
    }
    arg->cmd = err;
}

static void cuda_memcpy_async(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err=-1;
    uint32_t size;
    void *src, *dst, *addr;
    int pos = 0;
    cudaStream_t stream=0;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];

    func();
    init_primary_context(ctx);
    debug("src=0x%lx, srcSize=0x%x, dst=0x%lx, dstSize=0x%x, kind=%lu, "
        "stream=0x%lx , src2=0x%lx\n",
        arg->src, arg->srcSize, arg->dst, arg->dstSize, arg->flag, 
        arg->param, arg->src2);
    pos = arg->src2;
    if (pos==0) {
        stream=0;
    } else if (!__get_bit(&ctx->cudaStreamBitmap, pos-1)) {
        stream = ctx->cudaStream[pos-1];
    } else {
        error("No such stream, pos=%d\n", pos);
        arg->cmd=cudaErrorInvalidValue;
        return;
    }
    debug("stream 0x%lx\n", (uint64_t)stream);
    size = arg->srcSize;
    if (arg->flag == cudaMemcpyHostToDevice) {
        if((dst=(void*)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==0) {
            error("Failed to find dst virtual addr %p in vol\n",
                  (void *)arg->dst);
            arg->cmd=cudaErrorInvalidValue;
            return;
        }
        if(arg->param) {
            if( (src = (void*)map_host_addr_by_vaddr(arg->src, &ctx->host_vol))==NULL) {
                error("Failed to find virtual addr %p in host vol\n", (void *)arg->src);
                arg->cmd = cudaErrorInvalidValue;
                return;
            }
            execute_with_context((err= cudaMemcpyAsync(dst, src, size, 
                                cudaMemcpyHostToDevice, stream)), ctx->context);
            debug("src = %p\n", src);
            debug("dst = %p\n", dst);
            arg->cmd = err;
            if(err != cudaSuccess) {
                error("memcpy async HtoD error!\n");
            }
            return;
        } else {
            uint32_t rsize = size;
            int blocks = arg->paramSize;
            hwaddr *gpa_array = (hwaddr*)gpa_to_hva((hwaddr)arg->param2, blocks*sizeof(unsigned long));
            if(!gpa_array) {
                error("Failed to get gpa_array.\n");
                arg->cmd = cudaErrorInvalidValue;
                return;
            }
            uint32_t page_offset = arg->src & (BIT(PAGE_SHIFT) - 1);
            debug("page_offset %x\n", page_offset);
            if((addr = gpa_to_hva(gpa_array[0], PAGE_SIZE))==NULL) {
                error("No such physical address 0x%lx.\n", gpa_array[0]);
                arg->cmd = cudaErrorInvalidValue;
                return;
            }
            addr = addr + page_offset;
            uint32_t len = min(rsize, PAGE_SIZE - page_offset);
            execute_with_context((err= cudaMemcpyAsync(dst, addr, len, 
                        cudaMemcpyHostToDevice, stream)), ctx->context);
            if(err != cudaSuccess) {
                error("memcpy cudaMemcpyHostToDevice error!\n");
                arg->cmd = err;
                return;
            }
            dst     += len;
            rsize   -= len;
            for(int i=1; i<blocks; i++) {
                addr = gpa_to_hva(gpa_array[i], PAGE_SIZE);
                len = min(rsize, PAGE_SIZE);
                execute_with_context((err= cudaMemcpyAsync(dst, addr, len, 
                        cudaMemcpyHostToDevice, stream)), ctx->context);
                if(err != cudaSuccess) {
                    error("memcpy cudaMemcpyHostToDevice error!\n");
                    arg->cmd = err;
                    return;
                }
                dst     += len;
                rsize   -= len;
            }
            arg->cmd = err;
            return;
        }
    } else if (arg->flag == cudaMemcpyDeviceToHost) {
        if((src=(void*)map_device_addr_by_vaddr(arg->src, &ctx->vol))==0) {
            error("Failed to find virtual addr %p in vol or hvol\n",
                  (void *)arg->src);
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
        if(arg->param) {
            if( (dst = (void*)map_host_addr_by_vaddr(arg->dst, &ctx->host_vol))==0) {
                error("Failed to find virtual addr %p in host vol\n", (void *)arg->dst);
                arg->cmd = cudaErrorInvalidValue;
                return;
            }
            execute_with_context( (err=cudaMemcpyAsync(dst, src, size, 
                            cudaMemcpyDeviceToHost, stream)), ctx->context );
            debug("src = %p\n", src);
            debug("dst = %p\n", dst);
            debug("size = 0x%x\n", size);
            debug("(int)dst[0] = 0x%x\n", *(int*)dst);
            debug("(int)dst[1] = 0x%x\n", *((int*)dst+1));
            arg->cmd = err;
            if (err != cudaSuccess) {
                error("memcpy async DtoH error!\n");
            }
            return;
        } else {
            uint32_t rsize = size;
            int blocks = arg->paramSize;
            hwaddr *gpa_array = (hwaddr*)gpa_to_hva((hwaddr)arg->param2, blocks*sizeof(unsigned long));
            if(!gpa_array) {
                error("Failed to get gpa_array.\n");
                arg->cmd = cudaErrorInvalidValue;
                return;
            }
            uint32_t page_offset = arg->dst & (BIT(PAGE_SHIFT) - 1);
            debug("page_offset %x\n", page_offset);
            if((addr = gpa_to_hva(gpa_array[0], PAGE_SIZE))==NULL) {
                error("No such physical address 0x%lx.\n", gpa_array[0]);
                arg->cmd = cudaErrorInvalidValue;
                return;
            }
            addr = addr + page_offset;
            uint32_t len = min(rsize, PAGE_SIZE - page_offset);
            execute_with_context( (err=cudaMemcpyAsync(addr, src, len, 
                    cudaMemcpyDeviceToHost, stream)), ctx->context );
            if(err != cudaSuccess) {
                error("memcpy cudaMemcpyHostToDevice error!\n");
                arg->cmd = err;
                return;
            }
            src     += len;
            rsize   -= len;
            for(int i=1; i<blocks; i++) {
                addr = gpa_to_hva(gpa_array[i], PAGE_SIZE);
                len = min(rsize, PAGE_SIZE);
                execute_with_context( (err=cudaMemcpyAsync(addr, src, len, 
                    cudaMemcpyDeviceToHost, stream)), ctx->context );
                if(err != cudaSuccess) {
                    error("memcpy cudaMemcpyHostToDevice error!\n");
                    arg->cmd = err;
                    return;
                }
                src     += len;
                rsize   -= len;
            }
            arg->cmd = err;
            return;
        }
    } else if (arg->flag == cudaMemcpyDeviceToDevice) {
        if((src = (void*)map_device_addr_by_vaddr(arg->src, &ctx->vol))==0) {
            error("Failed to find virtual addr %p in vol\n",
                  (void *)arg->src);
            arg->cmd=cudaErrorInvalidValue;
            return;
        }
        if((dst = (void*)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==0) {
            error("Failed to find virtual addr %p in vol\n",
                  (void *)arg->dst);
            arg->cmd=cudaErrorInvalidValue;
            return;
        }
        execute_with_context( (err=cudaMemcpyAsync(dst, src, size, 
                            cudaMemcpyDeviceToDevice, stream)), ctx->context );
        arg->cmd = err;
        if (err != cudaSuccess) {
            error("memcpy async DtoD error!\n");
        }
        return;
    } else {
        error("No such memcpy direction.\n");
        arg->cmd= cudaErrorInvalidMemcpyDirection;
        return;
    }
}

static void cuda_memcpy_to_symbol(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err=-1;
    size_t size;
    void *src, *dst;
    size_t var_handle=0;
    size_t var_offset=0;
    int found       = 0;
    int i           = 0;
    int j           = 0;
    int var_count   = 0;
    int m_num       = 0;
    CudaMemVar      *var;
    CudaModule      *cuda_module;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    
    func();
    init_primary_context(ctx);
    debug("src=0x%lx, srcSize=0x%x, dst=0x%lx, dstSize=0x%x, kind=%lu, "
        "param=0x%lx, param2=0x%lx \n",
        arg->src, arg->srcSize, arg->dst, arg->dstSize, arg->flag, 
        arg->param, arg->param2);
    var_handle = (size_t)arg->dst;
    m_num = ctx->moduleCount;
    for (i=0; i < m_num; i++) {
        cuda_module = &ctx->modules[i];
        var_count = cuda_module->cudaVarsCount;
        for (j=0; j < var_count; j++) {
            var = &cuda_module->cudaVars[j];
            if (var->host_var == var_handle) {
                found = 1;
                break;
            }
        }
    }
    if (found == 0) {
        error("Failed to find such var handle 0x%lx\n", var_handle);
        arg->cmd = cudaErrorNoKernelImageForDevice;
        return;
    }

    size = arg->srcSize;
    if (arg->flag != 0) {
        if((src = (void*)map_host_addr_by_vaddr(arg->src, &ctx->host_vol))==NULL) {
            error("Failed to find virtual addr %p in host vol\n", (void *)arg->src);
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
    } else {
        if((src= gpa_to_hva((hwaddr)arg->param2, size)) == NULL) {
            error("No such src physical address 0x%lx.\n", arg->param2);
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
    }
    var_offset = (size_t)arg->param;
    dst = (void *)(var->device_ptr+var_offset);
    execute_with_context(err=cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice), ctx->context);
    arg->cmd = err;
    if(err != cudaSuccess) {
        error("memcpy to symbol HtoD error!\n");
    }
}

static void cu_memcpy_to_symbol(VirtQueueElement *elem, ThreadContext *tctx)
{
    CUresult err=0;
    size_t size;
    void *src;
    CUdeviceptr dst;
    size_t var_handle=0;
    size_t var_offset=0;
    int found       = 0;
    int i           = 0;
    int j           = 0;
    int var_count   = 0;
    int m_num       = 0;
    CudaMemVar      *var;
    CudaModule      *cuda_module;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    VirtIOArg *arg = elem->out_sg[0].iov_base;
    
    func();
    init_primary_context(ctx);
    debug("src=0x%lx, srcSize=0x%x, dst=0x%lx, dstSize=0x%x, kind=%lu, "
        "param=0x%lx, param2=0x%lx \n",
        arg->src, arg->srcSize, arg->dst, arg->dstSize, arg->flag, 
        arg->param, arg->param2);
    var_handle = (size_t)arg->dst;
    m_num = ctx->moduleCount;
    for (i=0; i < m_num; i++) {
        cuda_module = &ctx->modules[i];
        var_count = cuda_module->cudaVarsCount;
        for (j=0; j < var_count; j++) {
            var = &cuda_module->cudaVars[j];
            if (var->host_var == var_handle) {
                found = 1;
                break;
            }
        }
    }
    if (found == 0) {
        error("Failed to find such var handle 0x%lx\n", var_handle);
        arg->cmd = cudaErrorNoKernelImageForDevice;
        return;
    }

    size = arg->srcSize;
    src = elem->out_sg[1].iov_base;
    var_offset = (size_t)arg->param;
    dst = (CUdeviceptr)(var->device_ptr+var_offset);
    execute_with_cu_context(err=cuMemcpyHtoD(dst, src, size), ctx->context);
    arg->cmd = err;
    if(err != CUDA_SUCCESS) {
        error("memcpy to symbol HtoD error!\n");
    }
}

static void cuda_memcpy_from_symbol(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err=-1;
    size_t  size;
    void    *dst, *src;
    size_t  var_handle=0;
    size_t  var_offset=0;
    int found       = 0;
    int i           = 0;
    int j           = 0;
    int var_count   = 0;
    CudaMemVar      *var;
    CudaModule      *cuda_module;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    int m_num       = 0;

    func();
    init_primary_context(ctx);
    debug(  "src=0x%lx, srcSize=0x%x, dst=0x%lx, dstSize=0x%x, kind=0x%lx, "
            "param=0x%lx, param2=0x%lx,\n",
            arg->src, arg->srcSize, arg->dst, arg->dstSize, arg->flag, 
            arg->param, arg->param2);
 
    var_handle = (size_t)arg->src;
    m_num = ctx->moduleCount;
    for (i=0; i < m_num; i++) {
        cuda_module = &ctx->modules[i];
        var_count = cuda_module->cudaVarsCount;
        for (j=0; j < var_count; j++) {
            var = &cuda_module->cudaVars[j];
            if (var->host_var == var_handle) {
                found = 1;
                break;
            }
        }
    }
    if (found == 0) {
        error("Failed to find such var handle 0x%lx\n", var_handle);
        arg->cmd = cudaErrorNoKernelImageForDevice;
        return;
    }
    size = arg->dstSize;
    var_offset = (size_t)arg->param;
    src = (void*)(var->device_ptr+var_offset);
    if (arg->flag != 0) {
        if((dst = (void*)map_host_addr_by_vaddr(arg->dst, &ctx->host_vol))==NULL) {
            error("Failed to find virtual addr %p in host vol\n", (void *)arg->dst);
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
    } else {
        if((dst= gpa_to_hva((hwaddr)arg->param2, size)) == NULL) {
            error("No such dst physical address 0x%lx.\n", arg->param2);
            arg->cmd = cudaErrorInvalidValue;
            return;
        }
    }
    execute_with_context(err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost), ctx->context);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("memcpy from symbol DtoH error!\n");
    }
}

static void cu_memcpy_from_symbol(VirtQueueElement *elem, ThreadContext *tctx)
{
    CUresult err=-1;
    size_t  size;
    void    *dst;
    CUdeviceptr src;
    size_t  var_handle=0;
    size_t  var_offset=0;
    int found       = 0;
    int i           = 0;
    int j           = 0;
    int var_count   = 0;
    int m_num       = 0;
    CudaMemVar      *var;
    CudaModule      *cuda_module;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    init_primary_context(ctx);
    debug(  "src=0x%lx, srcSize=0x%x, dst=0x%lx, dstSize=0x%x, kind=0x%lx, "
            "param=0x%lx, param2=0x%lx,\n",
            arg->src, arg->srcSize, arg->dst, arg->dstSize, arg->flag, 
            arg->param, arg->param2);
 
    var_handle = (size_t)arg->src;
    m_num = ctx->moduleCount;
    for (i=0; i < m_num; i++) {
        cuda_module = &ctx->modules[i];
        var_count = cuda_module->cudaVarsCount;
        for (j=0; j < var_count; j++) {
            var = &cuda_module->cudaVars[j];
            if (var->host_var == var_handle) {
                found = 1;
                break;
            }
        }
    }
    if (found == 0) {
        error("Failed to find such var handle 0x%lx\n", var_handle);
        arg->cmd = cudaErrorNoKernelImageForDevice;
        return;
    }
    size = arg->dstSize;
    var_offset = (size_t)arg->param;
    src = (CUdeviceptr)(var->device_ptr+var_offset);
    dst = elem->in_sg[0].iov_base;
    execute_with_cu_context(err = cuMemcpyDtoH(dst, src, size), ctx->context);
    arg->cmd = err;
    if (err != CUDA_SUCCESS) {
        error("memcpy from symbol DtoH error!\n");
    }
}


static void cuda_memset(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err=-1;
    size_t count;
    int value;
    uint64_t dst;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    
    func();
    init_primary_context(ctx);
    count = (size_t)(arg->param);
    value = (int)(arg->param2);
    debug("dst=0x%lx, value=0x%x, count=0x%lx\n", arg->dst, value, count);
    if((dst = map_device_addr_by_vaddr(arg->dst, &ctx->vol))==0) {
        error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
        arg->cmd = cudaErrorInvalidValue;
        return;
    }
    debug("dst=0x%lx\n", dst);
    execute_with_context( (err= cudaMemset((void*)dst, value, count)), ctx->context);
    arg->cmd = err;
    if (err != cudaSuccess)
        error("memset memory error!\n");
}

static void cu_memset(VirtQueueElement *elem, ThreadContext *tctx)
{
    cudaError_t err=-1;
    size_t count;
    int value;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    VirtIOArg *arg = elem->out_sg[0].iov_base;
    
    func();
    init_primary_context(ctx);
    count = (size_t)(arg->param);
    value = (int)(arg->param2);
    debug("dst=0x%lx, value=0x%x, count=0x%lx\n", arg->dst, value, count);

    execute_with_context( (err= cudaMemset((void*)arg->dst, value, count)), ctx->context);
    arg->cmd = err;
    if (err != cudaSuccess)
        error("memset memory error!\n");
}

static void cu_malloc(VirtQueueElement *elem, ThreadContext *tctx)
{
    CUresult err=-1;
    CUdeviceptr devPtr;
    size_t size = 0;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    init_primary_context(ctx);
    size = arg->srcSize;    
    execute_with_cu_context(err= cuMemAlloc(&devPtr, size), ctx->context);
    arg->cmd = err;
    if (err != CUDA_SUCCESS) {
        error("Alloc memory error!\n");
        return;
    }
    arg->dst    = (uint64_t)devPtr;
    debug("devptr = 0x%llx\n", devPtr);
}

static int set_shm(size_t size, char *file_path)
{
    int res=0;
    int mmap_fd = shm_open(file_path, O_CREAT|O_RDWR, S_IRUSR|S_IWUSR);
    if (mmap_fd == -1) {
        error("Failed to open with errno %s\n", strerror(errno));
        return 0;
    }
    shm_unlink(file_path);
    // extend
    res = ftruncate(mmap_fd, size);
    if (res == -1) {
        error("Failed to ftruncate.\n");
        return 0;
    }
    return mmap_fd;
}

static void cu_host_register(VirtQueueElement *elem, ThreadContext *tctx)
{
    CUresult err=-1;
    long size;
    void *ptr;
    int i = 0;
    unsigned int flags = 0;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    debug("src=0x%lx, srcSize=0x%x, dst=0x%lx, "
          "dstSize=0x%x, kind=0x%lx, elem nr=0x%lx\n",
          arg->src, arg->srcSize, arg->dst, 
          arg->dstSize, arg->flag, arg->param);
    flags = (unsigned int)arg->flag;
    debug("memory chunk is %d\n", elem->out_num-1);
    cuError(cuCtxSetCurrent(ctx->context));
    for(i=1; i < elem->out_num; i++) {
        size = elem->out_sg[i].iov_len;
        ptr = elem->out_sg[i].iov_base;
        cuError((err=cuMemHostRegister(ptr, size, flags)));
        if(err != CUDA_SUCCESS) {
            error("Failed to register memory\n");
            break;
        }
    }
    arg->cmd = err;
}

static void cu_host_unregister(VirtQueueElement *elem, ThreadContext *tctx)
{
    CUresult err=-1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    VirtIOArg *arg = elem->out_sg[0].iov_base;
    int i=0;
    void *ptr=NULL;

    func();
    debug("src=0x%lx, srcSize=0x%x, dst=0x%lx, "
          "dstSize=0x%x, kind=0x%lx, elem nr=0x%lx\n",
          arg->src, arg->srcSize, arg->dst, 
          arg->dstSize, arg->flag, arg->param);
    cuError(cuCtxSetCurrent(ctx->context));
    for(i=1; i < elem->out_num; i++) {
        ptr = elem->out_sg[i].iov_base;
        cuError(err=cuMemHostUnregister(ptr));
        if(err != CUDA_SUCCESS) {
            error("Failed to unregister memory\n");
            break;
        }
    }
    arg->cmd = err;
}

static void cuda_free(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err=-1;
    VOL *vol, *vol2;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    
    func();
    init_primary_context(ctx);
    debug(" ptr = 0x%lx\n", arg->src);
    list_for_each_entry_safe(vol, vol2, &ctx->vol, list) {
        if (vol->v_addr == arg->src) {
            debug(  "actual devPtr=0x%lx, virtual ptr=0x%lx\n", 
                    (uint64_t)vol->addr, arg->src);
            execute_with_context( (err= cudaFree((void*)(vol->addr))), ctx->context);
            arg->cmd = err;
            if (err != cudaSuccess) {
                error("free error.\n");
                return;
            }
            list_del(&vol->list);
            free(vol);
            return;
        }
    }
    arg->cmd = cudaErrorInvalidValue;
    error("Failed to free ptr. Not found it!\n");
}

static void cu_free(VirtQueueElement *elem, ThreadContext *tctx)
{
    CUresult err=-1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    VirtIOArg *arg = elem->out_sg[0].iov_base;
    
    func();
    init_primary_context(ctx);
    debug(" ptr = 0x%lx\n", arg->src);
    execute_with_cu_context( (err= cuMemFree((CUdeviceptr)arg->src)), ctx->context);
    arg->cmd = err;
}

/*
 * Let vm user see which card he actually uses
*/
static void cuda_get_device(VirtIOArg *arg)
{
    return;
}

/*
 * done by the vgpu in vm
 * this function is useless
*/
static void cuda_get_device_properties(VirtIOArg *arg)
{
    return;
}

static void cuda_get_device_count(VirtIOArg *arg)
{
    func();
    debug("Device count=%d.\n", total_device);
    arg->cmd = (int32_t)cudaSuccess;
    arg->flag = (uint64_t)total_device;
}

static void cuda_device_reset(VirtQueueElement *elem, ThreadContext *tctx)
{
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    VirtIOArg *arg = elem->out_sg[0].iov_base;
    func();
    cuError(cuDeviceGet(&ctx->dev, tctx->cur_dev));
    if(ctx->initialized)
        cuError(cuCtxDestroy(ctx->context));
    deinit_primary_context(ctx);
    tctx->deviceBitmap &= ~(1 << tctx->cur_dev);
    arg->cmd = 0;
    debug("reset devices\n");
}

static void cu_stream_create(VirtQueueElement *elem, ThreadContext *tctx)
{
    CUresult err = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    CUstream stream;
    VirtIOArg *arg = elem->out_sg[0].iov_base;
    
    func();
    execute_with_cu_context(err = cuStreamCreate(&stream, CU_STREAM_DEFAULT), ctx->context);
    arg->cmd = err;
    if (err != CUDA_SUCCESS) {
        error("create stream error.\n");
        return;
    }
    debug("create stream 0x%lx\n", (uint64_t)stream);
    arg->stream = (uint64_t)stream;
}

static void cu_stream_create_with_flags(VirtQueueElement *elem, ThreadContext *tctx)
{
    CUresult err = -1;
    unsigned int flag=0;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    CUstream stream;
    VirtIOArg *arg = elem->out_sg[0].iov_base;
    
    func();
    flag = (unsigned int)arg->flag;
    execute_with_cu_context( (err=cuStreamCreate(&stream, flag)), ctx->context);
    arg->cmd = err;
    if (err != CUDA_SUCCESS) {
        error("create stream with flags error.\n");
        return;
    }
    arg->stream = (uint64_t)stream;
    debug("create stream 0x%lx with flag %u\n", (uint64_t)stream, flag);
}

static void cu_stream_destroy(VirtQueueElement *elem, ThreadContext *tctx)
{
    CUresult err = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    CUstream stream;
    VirtIOArg *arg = elem->out_sg[0].iov_base;
    
    func();
    stream = (CUstream)arg->stream;
    debug("destroy stream 0x%lx\n", (uint64_t)stream);
    execute_with_cu_context((err=cuStreamDestroy(stream)), ctx->context);
    arg->cmd = err;
    if (err != CUDA_SUCCESS) {
        error("destroy stream error.\n");
    }
}

static void cu_stream_synchronize(VirtQueueElement *elem, ThreadContext *tctx)
{
    CUresult err = -1;
    CUstream stream;
    VirtIOArg *arg = elem->out_sg[0].iov_base;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];

    func();
    stream = (CUstream)arg->stream;
    debug("synchrone stream 0x%lx\n", (uint64_t)stream);
    execute_with_cu_context((err=cuStreamSynchronize(stream)), ctx->context);
    arg->cmd = err;
    if (err != CUDA_SUCCESS) {
        error("synchronize stream error.\n");
    }
}

static void cu_stream_wait_event(VirtQueueElement *elem, ThreadContext *tctx)
{
    CUresult err = -1;
    CUstream    stream = 0;
    CUevent     event = 0;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    stream = (CUstream)arg->stream;
    debug("stream 0x%lx\n", (uint64_t)stream);
    event = (CUevent)arg->event;
    debug("wait for event 0x%lx\n", (uint64_t)event);    
    execute_with_cu_context(err = cuStreamWaitEvent(stream, event, 0), ctx->context);
    arg->cmd = err;
    if (err != CUDA_SUCCESS) {
        error("failed to wait event for stream.\n");
    }
}

static void cu_event_create(VirtQueueElement *elem, ThreadContext *tctx)
{
    CUresult err = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    CUevent event;
    VirtIOArg *arg = elem->out_sg[0].iov_base;
    
    func();
    execute_with_cu_context(err = cuEventCreate(&event, CU_EVENT_DEFAULT), ctx->context);
    arg->cmd = err;
    if (err != CUDA_SUCCESS) {
        error("create event error.\n");
        return;
    }
    debug("create event 0x%lx\n", (uint64_t)event);
    arg->event = (uint64_t)event;
}

static void cu_event_create_with_flags(VirtQueueElement *elem, ThreadContext *tctx)
{
    CUresult err = -1;
    unsigned int flag=0;
    CUevent event;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    flag = (unsigned int)arg->flag;
    execute_with_cu_context((err=cuEventCreate(&event, flag)), ctx->context);    
    arg->cmd = err;
    if (err != CUDA_SUCCESS) {
        error("create event with flags error.\n");
        return;
    }
    arg->event = (uint64_t)event;
    debug("create event 0x%lx with flag %u\n", (uint64_t)event, flag);
}

static void cu_event_destroy(VirtQueueElement *elem, ThreadContext *tctx)
{
    CUresult err = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    CUevent event;
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    event = (CUevent)arg->event;
    debug("tid %d destroy event 0x%lx\n", arg->tid, (uint64_t)event);
    execute_with_cu_context( (err=cuEventDestroy(event)), ctx->context);
    arg->cmd = err;
    if (err != CUDA_SUCCESS) {
        error("destroy event error.\n");
    }    
}

static void cu_event_record(VirtQueueElement *elem, ThreadContext *tctx)
{
    CUresult err = -1;
    CUstream stream;
    CUevent event;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    event = (CUevent)arg->event;
    stream = (CUstream)arg->stream;
    debug("record event 0x%lx, stream=0x%lx\n", 
        (uint64_t)event, (uint64_t)stream);
    execute_with_cu_context((err=cuEventRecord(event, stream)), ctx->context);
    arg->cmd = err;
    if (err != CUDA_SUCCESS) {
        error("record event error.\n");
    }
}

static void cu_event_synchronize(VirtQueueElement *elem, ThreadContext *tctx)
{
    CUresult err = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    VirtIOArg *arg = elem->out_sg[0].iov_base;
    CUevent event;

    func();
    event = (CUevent)arg->event;
    debug("sync event 0x%lx\n", (uint64_t)event);
    execute_with_cu_context( (err=cuEventSynchronize(event)), ctx->context);
    
    arg->cmd = err;
    if (err != CUDA_SUCCESS) {
        error("synchronize event error.\n");
    }
}

static void cu_event_query(VirtQueueElement *elem, ThreadContext *tctx)
{
    CUresult err = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    CUevent event;
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    event = (CUevent)arg->event;
    debug("query event 0x%lx\n", (uint64_t)event);
    execute_with_cu_context( (err=cuEventQuery(event)), ctx->context);
    arg->cmd = err;
}



static void cu_event_elapsedtime(VirtQueueElement *elem, ThreadContext *tctx)
{
    CUresult err = -1;
    float time = 0;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    CUevent start, stop;
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    start = (CUevent)arg->event;
    stop = (CUevent)arg->event2;
    debug("start event 0x%lx\n", (uint64_t)start);
    debug("stop event 0x%lx\n", (uint64_t)stop);
    execute_with_cu_context( (err=cuEventElapsedTime(&time, 
                                 start, stop)), ctx->context);    
    arg->cmd = err;
    if (err != CUDA_SUCCESS) {
        error("event calc elapsed time error.\n");
        return;
    }
    debug("elapsed time=%g\n", time);
    memcpy((void*)elem->in_sg[0].iov_base, &time, sizeof(float));
}

static void cuda_device_synchronize(VirtQueueElement *elem, ThreadContext *tctx)
{
    cudaError_t err = 0;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    VirtIOArg *arg = elem->out_sg[0].iov_base;
    func();
    if (ctx->initialized)
        execute_with_cu_context(err = cuCtxSynchronize(), ctx->context);
    arg->cmd = err;
}

static void cuda_thread_synchronize(VirtQueueElement *elem, ThreadContext *tctx)
{
    func();
    /*
    * cudaThreadSynchronize is deprecated
    * cudaError( (err=cudaThreadSynchronize()) );
    */
    cuda_device_synchronize(elem, tctx);
}
/* unused */
static void cuda_get_last_error(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    func();
    init_primary_context(ctx);
    execute_with_context(err = cudaGetLastError(), ctx->context);
    arg->cmd = err;
}
/* unused */
static void cuda_peek_at_last_error(VirtIOArg *arg, ThreadContext *tctx)
{
    cudaError_t err = -1;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    func();
    init_primary_context(ctx);
    execute_with_context(err = cudaPeekAtLastError(), ctx->context);
    arg->cmd = err;
}

static void cuda_mem_get_info(VirtQueueElement *elem, ThreadContext *tctx)
{
    cudaError_t err = -1;
    size_t freeMem, totalMem;
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    VirtIOArg *arg = elem->out_sg[0].iov_base;
    
    func();
    execute_with_context(err = cudaMemGetInfo(&freeMem, &totalMem), ctx->context);
    arg->cmd = err;
    if (err != cudaSuccess) {
        error("get mem info error!\n");
        return;
    }
    arg->srcSize = freeMem;
    arg->dstSize = totalMem;
    debug("free memory = %lu, total memory = %lu.\n", freeMem, totalMem);
}

static void cublas_create(VirtQueueElement *elem, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    cublasHandle_t handle;
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    cublasCheck(status = cublasCreate_v2(&handle));
    memcpy(elem->in_sg[0].iov_base, &handle, sizeof(cublasHandle_t));
    arg->cmd = status;
    debug("cublas handle 0x%lx\n", (uint64_t)handle);
}

static void cublas_destroy(VirtQueueElement *elem, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    VirtIOArg *arg = elem->out_sg[0].iov_base;
    cublasHandle_t handle = (cublasHandle_t)arg->flag;

    func();
    cublasCheck(status = cublasDestroy_v2(handle));
    arg->cmd = status;
    debug("cublas handle 0x%lx\n", (uint64_t)handle);
}

static void cublas_set_vector(VirtQueueElement *elem, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    // CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    int n;
    int elemSize;
    int incx, incy;
    void *buf = NULL;
    void *src, *dst;
    int int_size = sizeof(int);
    int idx = 0;
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    dst = (void*)arg->dst;
    src = elem->out_sg[2].iov_base;
    buf = elem->out_sg[1].iov_base;
    memcpy(&n, buf+idx, int_size);
    idx += int_size;
    memcpy(&elemSize, buf+idx, int_size);
    idx += int_size;
    memcpy(&incx, buf+idx, int_size);
    idx += int_size;
    memcpy(&incy, buf+idx, int_size);
    cublasCheck(status = cublasSetVector(n, elemSize, src, incx, dst, incy));
    debug("n=%d, elemSize=%d, src=%p, incx=%d, dst=%p, incy=%d\n",
        n, elemSize, src, incx, dst, incy);
    arg->cmd = status;
}

static void cublas_get_vector(VirtQueueElement *elem, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    // CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    int n;
    int elemSize;
    int incx, incy;
    void *buf = NULL;
    void *src, *dst;
    int int_size = sizeof(int);
    int idx = 0;
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    buf = elem->out_sg[1].iov_base;
    src = (void*)arg->src;
    dst = (void*)elem->in_sg[0].iov_base;
    memcpy(&n, buf+idx, int_size);
    idx += int_size;
    memcpy(&elemSize, buf+idx, int_size);
    idx += int_size;
    memcpy(&incx, buf+idx, int_size);
    idx += int_size;
    memcpy(&incy, buf+idx, int_size);
    cublasCheck(status = cublasGetVector(n, elemSize, src, incx, dst, incy));
    debug("n=%d, elemSize=%d, src=%p, incx=%d, dst=%p, incy=%d\n",
        n, elemSize, src, incx, dst, incy);
    arg->cmd = status;
}

static void cublas_set_matrix(VirtQueueElement *elem, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    // CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    int rows, cols;
    int elemSize;
    int lda, ldb;
    void *buf = NULL;
    void *src, *dst;
    int int_size = sizeof(int);
    int idx = 0;
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    dst = (void*)arg->dst;
    buf = elem->out_sg[1].iov_base;
    src = elem->out_sg[2].iov_base;
    memcpy(&rows, buf+idx, int_size);
    idx += int_size;
    memcpy(&cols, buf+idx, int_size);
    idx += int_size;
    memcpy(&elemSize, buf+idx, int_size);
    idx += int_size;
    memcpy(&lda, buf+idx, int_size);
    idx += int_size;
    memcpy(&ldb, buf+idx, int_size);
    cublasCheck(status = cublasSetMatrix(rows, cols, elemSize, 
                                            src, lda, 
                                            dst, ldb));
    arg->cmd = status;
}

static void cublas_get_matrix(VirtQueueElement *elem, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    // CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    int rows, cols;
    int elemSize;
    int lda, ldb;
    void *buf = NULL;
    void *src, *dst;
    int int_size = sizeof(int);
    int idx = 0;
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    // device address
    src = (void*)arg->src;
    dst = (void*)elem->in_sg[0].iov_base;
    buf = elem->out_sg[1].iov_base;
    memcpy(&rows, buf+idx, int_size);
    idx += int_size;
    memcpy(&cols, buf+idx, int_size);
    idx += int_size;
    memcpy(&elemSize, buf+idx, int_size);
    idx += int_size;
    memcpy(&lda, buf+idx, int_size);
    idx += int_size;
    memcpy(&ldb, buf+idx, int_size);
    cublasCheck(status = cublasGetMatrix (rows, cols, elemSize, 
                                             src, lda, 
                                             dst, ldb));
    arg->cmd = status;
}

static void cublas_sgemm(VirtQueueElement *elem, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    // CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    void *buf = NULL;
    int int_size = sizeof(int);
    int idx = 0;
    cublasHandle_t handle;
    cublasOperation_t transa;
    cublasOperation_t transb;
    int m, n, k;
    int lda, ldb, ldc;
    float *A, *B, *C;
    float alpha; /* host or device pointer */
    float beta; /* host or device pointer */
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    // device address
    A = (void*)arg->src;
    B = (void*)arg->src2;
    C = (void*)arg->dst;
    buf = (uint8_t *)elem->out_sg[1].iov_base;
    memcpy(&handle, buf+idx, sizeof(cublasHandle_t));
    idx += sizeof(cublasHandle_t);
    memcpy(&transa, buf+idx, sizeof(cublasOperation_t));
    idx += sizeof(cublasOperation_t);
    memcpy(&transb, buf+idx, sizeof(cublasOperation_t));
    idx += sizeof(cublasOperation_t);
    memcpy(&m, buf+idx, int_size);
    idx += int_size;
    memcpy(&n, buf+idx, int_size);
    idx += int_size;
    memcpy(&k, buf+idx, int_size);
    idx += int_size;
    memcpy(&lda, buf+idx, int_size);
    idx += int_size;
    memcpy(&ldb, buf+idx, int_size);
    idx += int_size;
    memcpy(&ldc, buf+idx, int_size);
    idx += int_size;
    memcpy(&alpha, buf+idx, sizeof(float));
    idx += sizeof(float);
    memcpy(&beta, buf+idx, sizeof(float));
    cublasCheck(status = cublasSgemm_v2(handle, transa, transb,
                                        m, n, k,
                                        &alpha, A, lda,
                                        B, ldb,
                                        &beta, C, ldc));
    debug("handle=%lx, transa=0x%lx, transb=0x%lx, m=%d, n=%d, k=%d,"
            " alpha=%g, A=%p, lda=%d,"
            " B=%p, ldb=%d, beta=%g, C=%p, ldc=%d\n",
            (uint64_t)handle, (uint64_t)transa, (uint64_t)transb, 
            m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    arg->cmd = status;
}

static void cublas_dgemm(VirtQueueElement *elem, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    // CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    void *buf = NULL;
    int int_size = sizeof(int);
    int idx = 0;
    cublasHandle_t handle;
    cublasOperation_t transa;
    cublasOperation_t transb;
    int m, n, k;
    int lda, ldb, ldc;
    double *A, *B, *C;
    double alpha; /* host or device pointer */
    double beta; /* host or device pointer */
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    // device address
    A = (void*)arg->src;
    B = (void*)arg->src2;
    C = (void*)arg->dst;
    buf = (uint8_t *)elem->out_sg[1].iov_base;
    memcpy(&handle, buf+idx, sizeof(cublasHandle_t));
    idx += sizeof(cublasHandle_t);
    memcpy(&transa, buf+idx, sizeof(cublasOperation_t));
    idx += sizeof(cublasOperation_t);
    memcpy(&transb, buf+idx, sizeof(cublasOperation_t));
    idx += sizeof(cublasOperation_t);
    memcpy(&m, buf+idx, int_size);
    idx += int_size;
    memcpy(&n, buf+idx, int_size);
    idx += int_size;
    memcpy(&k, buf+idx, int_size);
    idx += int_size;
    memcpy(&lda, buf+idx, int_size);
    idx += int_size;
    memcpy(&ldb, buf+idx, int_size);
    idx += int_size;
    memcpy(&ldc, buf+idx, int_size);
    idx += int_size;
    memcpy(&alpha, buf+idx, sizeof(double));
    idx += sizeof(double);
    memcpy(&beta, buf+idx, sizeof(double));
    cublasCheck(status = cublasDgemm_v2(handle, transa, transb,
                                        m, n, k,
                                        &alpha, A, lda,
                                        B, ldb,
                                        &beta, C, ldc));
    debug("handle=%lx, transa=0x%lx, transb=0x%lx, m=%d, n=%d, k=%d,"
            " alpha=%g, A=%p, lda=%d,"
            " B=%p, ldb=%d, beta=%g, C=%p, ldc=%d\n",
            (uint64_t)handle, (uint64_t)transa, (uint64_t)transb, 
            m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    arg->cmd = status;
}

static void cublas_set_stream(VirtQueueElement *elem, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    cublasHandle_t handle;
    cudaStream_t stream = 0;
    // CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    handle = (cublasHandle_t)arg->src;
    stream = (cudaStream_t)arg->stream;
    debug("handle 0x%lx, stream %lx\n", (uint64_t)handle, (uint64_t)stream);
    cublasCheck(status = cublasSetStream_v2(handle, stream));
    arg->cmd = status;
}

static void cublas_get_stream(VirtQueueElement *elem, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    cublasHandle_t handle;
    cudaStream_t stream = 0;
    // CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    handle = (cublasHandle_t)arg->src;
    cublasCheck(status = cublasGetStream_v2(handle, &stream));
    arg->cmd = status;
    debug("handle 0x%lx, stream %lx\n", (uint64_t)handle, (uint64_t)stream);
    arg->stream = (uint64_t)stream;
}

static void cublas_sasum(VirtQueueElement *elem, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    // CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    cublasHandle_t handle;
    int n;
    int incx;
    float *x;
    float result; /* host or device pointer */
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    // device address
    x = (void*)arg->src;
    handle  = (cublasHandle_t)arg->flag;
    n       = (int)arg->srcSize;
    incx    = (int)arg->srcSize2;
    cublasCheck(status = cublasSasum_v2(handle, n, x, incx, &result));
    debug("handle=%lx, n=%d, x=%p, incx=%d, result=%g\n",
            (uint64_t)handle, n, x, incx, result);
    memcpy(elem->in_sg[0].iov_base, &result, sizeof(float));
    arg->cmd = status;
}

static void cublas_dasum(VirtQueueElement *elem, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    // CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    cublasHandle_t handle;
    int n;
    int incx;
    double *x;
    double result; /* host or device pointer */
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    // device address
    x = (void*)arg->src;
    handle  = (cublasHandle_t)arg->flag;
    n       = (int)arg->srcSize;
    incx    = (int)arg->srcSize2;
    cublasCheck(status = cublasDasum_v2(handle, n, x, incx, &result));
    debug("handle=%lx, n=%d, x=%p, incx=%d, result=%g\n",
            (uint64_t)handle, n, x, incx, result);
    memcpy(elem->in_sg[0].iov_base, &result, sizeof(double));
    arg->cmd = status;
}

static void cublas_scopy(VirtQueueElement *elem, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    // CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    cublasHandle_t handle;
    int n;
    int incx, incy;
    float *x, *y;
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    // device address
    x = (void*)arg->src;
    y = (void*)arg->dst;
    handle  = (cublasHandle_t)arg->flag;
    n       = (int)arg->srcSize;
    incx    = (int)arg->srcSize2;
    incy    = (int)arg->dstSize;
    cublasCheck(status = cublasScopy_v2 (handle, n, x, incx, y, incy));
    debug("handle=%lx, n=%d, x=%p, incx=%d, y=%p, incy=%d\n",
            (uint64_t)handle, n, x, incx, y, incy);
    arg->cmd = status;
}

static void cublas_dcopy(VirtQueueElement *elem, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    // CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    cublasHandle_t handle;
    int n;
    int incx, incy;
    double *x, *y;
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    // device address
    x = (void*)arg->src;
    y = (void*)arg->dst;
    handle  = (cublasHandle_t)arg->flag;
    n       = (int)arg->srcSize;
    incx    = (int)arg->srcSize2;
    incy    = (int)arg->dstSize;
    cublasCheck(status = cublasDcopy_v2 (handle, n, x, incx, y, incy));
    debug("handle=%lx, n=%d, x=%p, incx=%d, y=%p, incy=%d\n",
            (uint64_t)handle, n, x, incx, y, incy);
    arg->cmd = status;
}

static void cublas_sdot(VirtQueueElement *elem, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    // CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    cublasHandle_t handle;
    int n;
    int incx, incy;
    float *x, *y;
    float result;
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    // device address
    x = (void*)arg->src;
    y = (void*)arg->dst;
    handle  = (cublasHandle_t)arg->flag;
    n       = (int)arg->srcSize;
    incx    = (int)arg->srcSize2;
    incy    = (int)arg->dstSize;
    cublasCheck(status = cublasSdot_v2(handle, n, x, incx, y, incy, &result));
    debug("handle=%lx, n=%d, x=%p, incx=%d, y=%p, incy=%d, result=%g\n",
            (uint64_t)handle, n, x, incx, y, incy, result);
    memcpy(elem->in_sg[0].iov_base, &result, sizeof(float));
    arg->cmd = status;
}

static void cublas_ddot(VirtQueueElement *elem, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    // CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    cublasHandle_t handle;
    int n;
    int incx, incy;
    double *x, *y;
    double result;
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    // device address
    x = (void*)arg->src;
    y = (void*)arg->dst;
    handle  = (cublasHandle_t)arg->flag;
    n       = (int)arg->srcSize;
    incx    = (int)arg->srcSize2;
    incy    = (int)arg->dstSize;
    cublasCheck(status = cublasDdot_v2(handle, n, x, incx, y, incy, &result));
    debug("handle=%lx, n=%d, x=%p, incx=%d, y=%p, incy=%d, result=%g\n",
            (uint64_t)handle, n, x, incx, y, incy, result);
    memcpy(elem->in_sg[0].iov_base, &result, sizeof(double));
    arg->cmd = status;
}

static void cublas_saxpy(VirtQueueElement *elem, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    // CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    cublasHandle_t handle;
    int n;
    int incx, incy;
    float *x, *y;
    float alpha; /* host or device pointer */
    uint8_t *buf = NULL;
    int idx = 0;
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    // device address
    x = (void*)arg->src;
    y = (void*)arg->dst;
    buf = elem->out_sg[1].iov_base;
    handle  = (cublasHandle_t)arg->flag;
    n       = (int)arg->srcSize;
    incx    = (int)arg->srcSize2;
    incy    = (int)arg->dstSize;
    memcpy(&alpha, buf+idx, sizeof(float));
    idx += sizeof(float);
    assert(idx == arg->paramSize);
    cublasCheck(status = cublasSaxpy_v2 (handle, n, &alpha, x, incx, y, incy));
    debug("handle=%lx, n=%d, alpha=%g, x=%p, incx=%d, y=%p, incy=%d\n",
            (uint64_t)handle, n, alpha, x, incx, y, incy);
    arg->cmd = status;
}

static void cublas_daxpy(VirtQueueElement *elem, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    // CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    cublasHandle_t handle;
    int n;
    int incx, incy;
    double *x, *y;
    double alpha; /* host or device pointer */
    uint8_t *buf = NULL;
    int idx = 0;
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    // device address
    x = (void*)arg->src;
    y = (void*)arg->dst;
    buf = elem->out_sg[1].iov_base;
    handle  = (cublasHandle_t)arg->flag;
    n       = (int)arg->srcSize;
    incx    = (int)arg->srcSize2;
    incy    = (int)arg->dstSize;
    memcpy(&alpha, buf+idx, sizeof(double));
    idx += sizeof(double);
    assert(idx == arg->paramSize);
    cublasCheck(status = cublasDaxpy_v2 (handle, n, &alpha, x, incx, y, incy));
        debug("handle=%lx, n=%d, alpha=%g, x=%p, incx=%d, y=%p, incy=%d\n",
            (uint64_t)handle, n, alpha, x, incx, y, incy);
    arg->cmd = status;
}

static void cublas_sscal(VirtQueueElement *elem, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    // CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    cublasHandle_t handle;
    int n;
    int incx;
    float *x;
    float alpha;
    uint8_t *buf = NULL;
    int idx = 0;
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    // device address
    x = (void*)arg->src;
    buf = elem->out_sg[1].iov_base;
    handle  = (cublasHandle_t)arg->flag;
    n       = (int)arg->srcSize;
    incx    = (int)arg->srcSize2;
    memcpy(&alpha, buf+idx, sizeof(float));
    idx += sizeof(float);
    assert(idx == arg->paramSize);
    cublasCheck(status = cublasSscal_v2(handle, n, &alpha, x, incx));
    debug("handle=%lx, n=%d, alpha=%g, x=%p, incx=%d\n",
            (uint64_t)handle, n, alpha, x, incx);
    arg->cmd = status;
}

static void cublas_dscal(VirtQueueElement *elem, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    // CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    cublasHandle_t handle;
    int n;
    int incx;
    double *x;
    double alpha;
    uint8_t *buf = NULL;
    int idx = 0;
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    // device address
    x = (void*)arg->src;
    buf = elem->out_sg[1].iov_base;
    handle  = (cublasHandle_t)arg->flag;
    n       = (int)arg->srcSize;
    incx    = (int)arg->srcSize2;
    memcpy(&alpha, buf+idx, sizeof(double));
    idx += sizeof(double);
    assert(idx == arg->paramSize);
    cublasCheck(status = cublasDscal_v2(handle, n, &alpha, x, incx));
    debug("handle=%lx, n=%d, alpha=%g, x=%p, incx=%d\n",
            (uint64_t)handle, n, alpha, x, incx);
    arg->cmd = status;
}

static void cublas_sgemv(VirtQueueElement *elem, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    // CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    uint8_t *buf = NULL;
    int int_size = sizeof(int);
    int idx = 0;
    cublasHandle_t handle;
    cublasOperation_t trans;
    int m, n;
    int lda;
    int incx, incy;
    float *x, *y, *A;
    float alpha, beta; /* host or device pointer */
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    // device address
    A = (void*)arg->src;
    x = (void*)arg->src2;
    y = (void*)arg->dst;
    buf = (uint8_t *)elem->out_sg[1].iov_base;
    n       = (int)arg->srcSize2;
    m       = (int)arg->dstSize;
    memcpy(&handle, buf+idx, sizeof(cublasHandle_t));
    idx += sizeof(cublasHandle_t);
    memcpy(&trans, buf+idx, sizeof(cublasOperation_t));
    idx += sizeof(cublasOperation_t);
    memcpy(&lda, buf+idx, int_size);
    idx += int_size;
    memcpy(&incx, buf+idx, int_size);
    idx += int_size;
    memcpy(&incy, buf+idx, int_size);
    idx += int_size;
    memcpy(&alpha, buf+idx, sizeof(float));
    idx += sizeof(float);
    memcpy(&beta, buf+idx, sizeof(float));
    idx += sizeof(float);
    assert(idx == arg->paramSize);
    cublasCheck(status = cublasSgemv_v2(handle, trans, m, n, 
                                        &alpha, A, lda, x, incx,
                                        &beta, y, incy));
    debug("handle=%lx, trans=0x%lx, m=%d, n=%d, alpha=%g, A=%p, lda=%d,"
            " x=%p, incx=%d, beta=%g, y=%p, incy=%d\n",
            (uint64_t)handle, (uint64_t)trans, m, n, 
            alpha, A, lda, x, incx, beta, y, incy);
    arg->cmd = status;
}

static void cublas_dgemv(VirtQueueElement *elem, ThreadContext *tctx)
{
    cublasStatus_t status = -1;
    // CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    uint8_t *buf = NULL;
    int int_size = sizeof(int);
    int idx = 0;
    cublasHandle_t handle;
    cublasOperation_t trans;
    int m, n;
    int lda;
    int incx, incy;
    double *x, *y, *A;
    double alpha, beta; /* host or device pointer */
    VirtIOArg *arg = elem->out_sg[0].iov_base;

    func();
    // device address
    A = (void*)arg->src;
    x = (void*)arg->src2;
    y = (void*)arg->dst;
    buf = (uint8_t *)elem->out_sg[1].iov_base;
    n       = (int)arg->srcSize2;
    m       = (int)arg->dstSize;
    memcpy(&handle, buf+idx, sizeof(cublasHandle_t));
    idx += sizeof(cublasHandle_t);
    memcpy(&trans, buf+idx, sizeof(cublasOperation_t));
    idx += sizeof(cublasOperation_t);
    memcpy(&lda, buf+idx, int_size);
    idx += int_size;
    memcpy(&incx, buf+idx, int_size);
    idx += int_size;
    memcpy(&incy, buf+idx, int_size);
    idx += int_size;
    memcpy(&alpha, buf+idx, sizeof(double));
    idx += sizeof(double);
    memcpy(&beta, buf+idx, sizeof(double));
    idx += sizeof(double);
    assert(idx == arg->paramSize);
    cublasCheck(status = cublasDgemv_v2(handle, trans, m, n, 
                                        &alpha, A, lda, x, incx,
                                        &beta, y, incy));
    debug("handle=%lx, trans=0x%lx, m=%d, n=%d, alpha=%g, A=%p, lda=%d,"
            " x=%p, incx=%d, beta=%g, y=%p, incy=%d\n",
            (uint64_t)handle, (uint64_t)trans, m, n, 
            alpha, A, lda, x, incx, beta, y, incy);
    arg->cmd = status;
}

static void curand_create_generator(VirtIOArg *arg, ThreadContext *tctx)
{
    curandStatus_t status = -1;
    curandGenerator_t generator;
    curandRngType_t rng_type = (curandRngType_t)arg->dst;
    func();
    curandCheck(status = curandCreateGenerator(&generator, rng_type));
    arg->flag = (uint64_t)generator;
    arg->cmd = status;
    debug("curand generator 0x%lx\n", (uint64_t)generator);
}

static void curand_create_generator_host(VirtIOArg *arg, ThreadContext *tctx)
{
    curandStatus_t status = -1;
    curandGenerator_t generator;
    curandRngType_t rng_type = (curandRngType_t)arg->dst;
    func();
    curandCheck(status = curandCreateGeneratorHost(&generator, rng_type));
    arg->flag = (uint64_t)generator;
    arg->cmd = status;
    debug("curand generator 0x%lx\n", (uint64_t)generator);
}

static void curand_generate(VirtIOArg *arg, ThreadContext *tctx)
{
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    curandStatus_t status = -1;
    curandGenerator_t generator;
    unsigned int *ptr;
    size_t n;

    func();
    // device address
    if (arg->flag == 0) {
        if( (ptr = (unsigned int *)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==NULL) {
            error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    } else if(arg->flag == 1) {
        if((ptr = (unsigned int *)gpa_to_hva((hwaddr)arg->param2, arg->dstSize))==NULL) {
            error("No such physical address %p.\n", (void *)arg->param2);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    } else if (arg->flag == 2) {
        if( (ptr = (unsigned int *)map_host_addr_by_vaddr(arg->dst, &ctx->host_vol))==NULL) {
            error("Failed to find virtual addr %p in host vol\n", (void *)arg->dst);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    }
    generator   = (curandGenerator_t)arg->src;
    n           = (size_t)arg->param;
    curandCheck(status=curandGenerate(generator, ptr, n));
    debug("generator=%lx, ptr=%p, n=%ld\n", (uint64_t)generator, ptr, n);
    arg->cmd = status;
}

static void curand_generate_normal(VirtIOArg *arg, ThreadContext *tctx)
{
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    curandStatus_t status = -1;
    curandGenerator_t generator;
    float *ptr;
    size_t n;
    float mean, stddev;
    uint8_t *buf = NULL;
    int idx = 0;

    func();
    if((buf = gpa_to_hva((hwaddr)arg->param, arg->paramSize))==NULL) {
        error("No such param physical address 0x%lx.\n", arg->param);
        arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
        return;
    }
    // device address
    if (arg->flag == 0) {
        if( (ptr = (float *)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==NULL) {
            error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    } else if(arg->flag == 1) {
        if((ptr = (float *)gpa_to_hva((hwaddr)arg->param2, arg->dstSize))==NULL) {
            error("No such physical address %p.\n", (void *)arg->param2);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    } else if (arg->flag == 2) {
        if( (ptr = (float*)map_host_addr_by_vaddr(arg->dst, &ctx->host_vol))==NULL) {
            error("Failed to find virtual addr %p in host vol\n", (void *)arg->dst);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    }
    generator   = (curandGenerator_t)arg->src;
    n           = (size_t)arg->src2;
    memcpy(&mean, buf+idx, sizeof(float));
    idx += sizeof(float);
    memcpy(&stddev, buf+idx, sizeof(float));
    idx += sizeof(float);
    assert(idx == arg->paramSize);
    curandCheck(status=curandGenerateNormal(generator, ptr, n, mean, stddev));
    debug("generator=%lx, ptr=%p, n=%ld, mean=%g, stddev=%g\n",
            (uint64_t)generator, ptr, n, mean, stddev);
    arg->cmd = status;
}

static void curand_generate_normal_double(VirtIOArg *arg, ThreadContext *tctx)
{
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    curandStatus_t status = -1;
    curandGenerator_t generator;
    double *ptr;
    size_t n;
    double mean, stddev;
    uint8_t *buf = NULL;
    int idx = 0;

    func();
    if (arg->flag == 0) {
        if( (ptr = (double *)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==NULL) {
            error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    } else if(arg->flag == 1) {
        if((ptr = (double *)gpa_to_hva((hwaddr)arg->param2, arg->dstSize))==NULL) {
            error("No such physical address %p.\n", (void *)arg->param2);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    } else if (arg->flag == 2) {
        if((ptr = (double *)map_host_addr_by_vaddr(arg->dst, &ctx->host_vol))==NULL) {
            error("Failed to find virtual addr %p in host vol\n", (void *)arg->dst);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    }
    if((buf = gpa_to_hva((hwaddr)arg->param, arg->paramSize))==NULL) {
        error("No such physical address 0x%lx.\n", arg->param);
        arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
        return;
    }
    generator   = (curandGenerator_t)arg->src;
    n           = (size_t)arg->src2;
    memcpy(&mean, buf+idx, sizeof(double));
    idx += sizeof(double);
    memcpy(&stddev, buf+idx, sizeof(double));
    idx += sizeof(double);
    assert(idx == arg->paramSize);
    curandCheck(status=curandGenerateNormalDouble(generator, ptr, n, mean, stddev));
    debug("generator=%lx, ptr=%p, n=%ld, mean=%g, stddev=%g\n",
            (uint64_t)generator, ptr, n, mean, stddev);
    arg->cmd = status;
}

static void curand_generate_uniform(VirtIOArg *arg, ThreadContext *tctx)
{
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    curandStatus_t status = -1;
    curandGenerator_t generator;
    float *ptr;
    size_t num;
    func();
    generator   = (curandGenerator_t)arg->src;
    num         = (size_t)arg->param;
    // device address
    if (arg->flag == 0) {
        if( (ptr = (float *)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==NULL) {
            error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    } else if(arg->flag == 1) {
        if((ptr = (float *)gpa_to_hva((hwaddr)arg->param2, arg->dstSize))==NULL) {
            error("No such physical address 0x%lx.\n", arg->param2);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    } else if (arg->flag == 2) {
        if( (ptr = (float*)map_host_addr_by_vaddr(arg->dst, &ctx->host_vol))==NULL) {
            error("Failed to find virtual addr %p in host vol\n", (void *)arg->dst);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    }
    curandCheck(status = curandGenerateUniform(generator, ptr, num));
    arg->cmd = status;
    debug("curand generator 0x%lx, ptr=%p, num=0x%lx\n", 
            (uint64_t)generator, ptr, num);
}

static void curand_generate_uniform_double(VirtIOArg *arg, ThreadContext *tctx)
{
    CudaContext *ctx = &tctx->contexts[tctx->cur_dev];
    curandStatus_t status = -1;
    curandGenerator_t generator;
    double *ptr;
    size_t num;
    func();
    generator   = (curandGenerator_t)arg->src;
    num         = (size_t)arg->param;
    // device address
    if (arg->flag == 0) {
        if( (ptr = (double *)map_device_addr_by_vaddr(arg->dst, &ctx->vol))==NULL) {
            error("Failed to find virtual addr %p in vol\n", (void *)arg->dst);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    } else if(arg->flag == 1) {
        if((ptr = (double *)gpa_to_hva((hwaddr)arg->param2, arg->dstSize))==NULL) {
            error("No such physical address 0x%lx.\n", arg->param2);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    } else if (arg->flag == 2) {
        if((ptr = (double *)map_host_addr_by_vaddr(arg->dst, &ctx->host_vol))==NULL) {
            error("Failed to find virtual addr %p in host vol\n", (void *)arg->dst);
            arg->cmd = CURAND_STATUS_ALLOCATION_FAILED;
            return;
        }
    }
    curandCheck(status = curandGenerateUniformDouble(generator, ptr, num));
    arg->cmd = status;
    debug("curand generator 0x%lx, ptr=%p, num=0x%lx\n", 
            (uint64_t)generator, ptr, num);
}

static void curand_destroy_generator(VirtIOArg *arg, ThreadContext *tctx)
{
    curandStatus_t status = -1;
    curandGenerator_t generator;
    func();
    generator   = (curandGenerator_t)arg->src;
    curandCheck(status = curandDestroyGenerator(generator));
    arg->cmd = status;
    debug("curand destroy generator 0x%lx\n", (uint64_t)generator);
}

static void curand_set_generator_offset(VirtIOArg *arg, ThreadContext *tctx)
{
    curandStatus_t status = -1;
    curandGenerator_t generator;
    unsigned long long offset;
    unsigned long long seed;
    func();
    generator   = (curandGenerator_t)arg->src;
    offset      = (unsigned long long)arg->param;
    seed        = (unsigned long long)arg->param2;
    curandCheck(status = curandSetPseudoRandomGeneratorSeed(generator, seed));
    curandCheck(status = curandSetGeneratorOffset(generator, offset));
    arg->cmd = status;
    debug("curand set offset generator 0x%lx, seed 0x%llx offset 0x%llx\n", 
        (uint64_t)generator, seed, offset);
}

static void curand_set_pseudorandom_seed(VirtIOArg *arg, ThreadContext *tctx)
{
    curandStatus_t status = -1;
    curandGenerator_t generator;
    unsigned long long seed;
    func();
    generator   = (curandGenerator_t)arg->src;
    seed      = (unsigned long long)arg->param;
    curandCheck(status = curandSetPseudoRandomGeneratorSeed(generator, seed));
    arg->cmd = status;
    debug("curand set seed generator 0x%lx, seed 0x%llx\n", 
        (uint64_t)generator, seed);
}

/* SGX***********************************************************************/

static void sgx_proc_msg0(VirtQueueElement *elem, ThreadContext *tctx)
{
    VirtIOArg *arg = elem->out_sg[0].iov_base;
    ra_samp_response_header_t* p_resp_msg;
    int ret = 0;

    func();
    ret = sp_ra_proc_msg0_req((sample_ra_msg0_t *)&arg->srcSize, 
            sizeof(sample_ra_msg0_t), &p_resp_msg);
    if (0 != ret) {
        error("Error, call sp_ra_proc_msg1_req fail.\n");
        arg->cmd = 1;
        return;
    }
    debug("resp size is %x\n", p_resp_msg->size);
    arg->paramSize = p_resp_msg->size;
    if(p_resp_msg->size > arg->dstSize) {
        error("Memory copy is bufferoverflow!\n");
        free(p_resp_msg);
        arg->cmd = 1;
        return;
    }
    memcpy(elem->in_sg[0].iov_base, p_resp_msg->body, p_resp_msg->size);
    free(p_resp_msg);
    arg->cmd = 0;
}

static void sgx_proc_msg1(VirtQueueElement *elem, ThreadContext *tctx)
{
    VirtIOArg *arg = elem->out_sg[0].iov_base;
    sgx_ra_msg1_t msg1;
    ra_samp_response_header_t* p_resp_msg;
    int ret = 0;

    func();
    memcpy(&msg1, elem->out_sg[1].iov_base, sizeof(sgx_ra_msg1_t));
    ret = sp_ra_proc_msg1_req(&tctx->g_sp_db, (const sample_ra_msg1_t*)&msg1,
            sizeof(sgx_ra_msg1_t),
            &p_resp_msg);
    if(0 != ret) {
        error("Error, call sp_ra_proc_msg1_req fail.\n");
        arg->cmd = 1;
        return;
    }
    debug("resp size is %x\n", p_resp_msg->size);
    arg->paramSize = p_resp_msg->size;
    if(p_resp_msg->size > arg->dstSize) {
        error("Memory copy is bufferoverflow!\n");
        free(p_resp_msg);
        arg->cmd = 1;
        return;
    }
    memcpy(elem->in_sg[0].iov_base, p_resp_msg->body, p_resp_msg->size);
    free(p_resp_msg);
    arg->cmd = 0;
}

static void sgx_proc_msg3(VirtQueueElement *elem, ThreadContext *tctx)
{
    VirtIOArg *arg = elem->out_sg[0].iov_base;
    uint8_t *msg3;
    ra_samp_response_header_t* p_resp_msg;
    sample_ra_att_result_msg_t *p_att_result;
    int ret = 0;

    func();
    msg3 = (uint8_t *)malloc(arg->srcSize);
    memcpy(msg3, elem->out_sg[1].iov_base, arg->srcSize);
    ret = sp_ra_proc_msg3_req(&tctx->g_sp_db, (const sample_ra_msg3_t*)msg3,
            arg->srcSize,
            &p_resp_msg);
    if(0 != ret) {
        error("Error, call sp_ra_proc_msg3_req fail.\n");
        arg->cmd = 1;
        return;
    }
    p_att_result= (sample_ra_att_result_msg_t *)p_resp_msg->body;
    arg->paramSize = p_resp_msg->size + p_att_result->secret.payload_size;
    if(p_resp_msg->size > arg->dstSize) {
        error("Memory copy is bufferoverflow!"
            " resp size %x, req size %x\n", p_resp_msg->size, arg->dstSize);
        arg->cmd = 1;
        free(msg3);
        free(p_resp_msg);
        return;
    }
    debug("resp size is 0x%x\n", p_resp_msg->size);
    memcpy(elem->in_sg[0].iov_base, p_resp_msg->body, arg->paramSize);
    free(msg3);
    free(p_resp_msg);
    arg->cmd = 0;
}

/*
static inline void cpu_physical_memory_read(hwaddr addr,
                                            void *buf, int len)

static inline void cpu_physical_memory_write(hwaddr addr,
                                             const void *buf, int len)
*/
static void cuda_gpa_to_hva(VirtIOArg *arg)
{
    int a;
    hwaddr gpa = (hwaddr)(arg->src);
    //a = gpa_to_hva(arg->src);
    cpu_physical_memory_read(gpa, &a, sizeof(int));
    debug("a=%d\n", a);
    a++;
    cpu_physical_memory_write(gpa, &a, sizeof(int));
    arg->cmd = 1;//(int32_t)cudaSuccess;
}

/* 
*   Callback function that's called when the guest sends us data.  
 * Guest wrote some data to the port. This data is handed over to
 * the app via this callback.  The app can return a size less than
 * 'len'.  In this case, throttling will be enabled for this port.
 */
static ssize_t flush_buf(VirtIOSerialPort *port,
                         const uint8_t *buf, ssize_t len)
{
    VirtIOArg *msg = NULL;
    debug("port->id=%d, len=%ld\n", port->id, len);
    /*if (len != sizeof(VirtIOArg)) {
        error("buf len should be %lu, not %ld\n", sizeof(VirtIOArg), len);
        return 0;
    }*/
    msg = (VirtIOArg *)malloc(len);
    memcpy((void *)msg, (void *)buf, len);
    // chan_send(port->thread_context->worker_queue, (void *)msg);
    return 0;
}

/* 
*   Callback function that's called when the guest sends us data.  
 * Guest wrote some data to the port. This data is handed over to
 * the app via this callback.  The app can return a size less than
 * 'len'.  In this case, throttling will be enabled for this port.
 */
static ssize_t handle_output(VirtIOSerialPort *port,
                         VirtQueueElement *elem)
{
    ThreadContext *tctx = port->thread_context;
    VirtIOArg *msg = elem->out_sg[0].iov_base;
    debug("port id = %d\n", port->id);
    debug("elem->out_num=%d, in_num=%d\n",elem->out_num, elem->in_num);
    debug("sizeof(VirtIOArg) %ld, len=%ld\n", 
            sizeof(VirtIOArg), elem->out_sg[0].iov_len);
    debug("cmd = %d\n", msg->cmd);
    switch(msg->cmd) {
        case VIRTIO_CUDA_HELLO:
            cuda_gpa_to_hva(msg);
            break;
        case VIRTIO_CUDA_PRIMARYCONTEXT:
            primarycontext(elem, tctx);
            break;
        case VIRTIO_CUDA_REGISTERFATBINARY:
            cuda_register_fatbinary(msg, tctx);
            break;
        case VIRTIO_CUDA_UNREGISTERFATBINARY:
            cuda_unregister_fatbinary(msg, tctx);
            break;
        case VIRTIO_CUDA_REGISTERFUNCTION:
            cuda_register_function(msg, tctx);
            break;
        case VIRTIO_CUDA_REGISTERVAR:
            cuda_register_var(msg, tctx);
            break;
        case VIRTIO_CUDA_LAUNCH:
            cuda_launch(msg, port);
            break;
        case VIRTIO_CUDA_LAUNCH_KERNEL:
            cu_launch_kernel(elem, port);
            break;
        case VIRTIO_CUDA_MALLOC:
            cu_malloc(elem, tctx);
            break;
        case VIRTIO_CUDA_HOSTREGISTER:
            cu_host_register(elem, tctx);
            break;
        case VIRTIO_CUDA_HOSTUNREGISTER:
            cu_host_unregister(elem, tctx);
            break;
        case VIRTIO_SGX_MEMCPY:
            cuda_memcpy_safe(msg, tctx);
            break;
        case VIRTIO_CUDA_MEMCPY:
            cuda_memcpy(msg, tctx);
            break;
        case VIRTIO_CUDA_MEMCPY_HTOD:
            cu_memcpy_htod(elem, tctx);
            break;
        case VIRTIO_CUDA_MEMCPY_DTOH:
            cu_memcpy_dtoh(elem, tctx);
            break;
        case VIRTIO_CUDA_MEMCPY_DTOD:
            cu_memcpy_dtod(elem, tctx);
            break;
        case VIRTIO_CUDA_MEMCPY_HTOD_ASYNC:
            cu_memcpy_htod_async(elem, tctx);
            break;
        case VIRTIO_CUDA_MEMCPY_DTOH_ASYNC:
            cu_memcpy_dtoh_async(elem, tctx);
            break;
        case VIRTIO_CUDA_MEMCPY_DTOD_ASYNC:
            cu_memcpy_dtod_async(elem, tctx);
            break;
        case VIRTIO_CUDA_FREE:
            cu_free(elem, tctx);
            break;
        case VIRTIO_CUDA_GETDEVICE:
            cuda_get_device(msg);
            break;
        case VIRTIO_CUDA_GETDEVICEPROPERTIES:
            cuda_get_device_properties(msg);
            break;
        case VIRTIO_CUDA_GETDEVICECOUNT:
            cuda_get_device_count(msg);
            break;
        case VIRTIO_CUDA_SETDEVICE:
            cu_set_device(elem, tctx);
            break;
        case VIRTIO_CUDA_SETDEVICEFLAGS:
            cuda_set_device_flags(msg, tctx);
            break;
        case VIRTIO_CUDA_DEVICESETCACHECONFIG:
            cuda_device_set_cache_config(msg, tctx);
            break;
        case VIRTIO_CUDA_DEVICERESET:
            cuda_device_reset(elem, tctx);
            break;
        case VIRTIO_CUDA_STREAMCREATE:
            cu_stream_create(elem, tctx);
            break;
        case VIRTIO_CUDA_STREAMCREATEWITHFLAGS:
            cu_stream_create_with_flags(elem, tctx);
            break;
        case VIRTIO_CUDA_STREAMDESTROY:
            cu_stream_destroy(elem, tctx);
            break;
        case VIRTIO_CUDA_STREAMWAITEVENT:
            cu_stream_wait_event(elem, tctx);
            break;
        case VIRTIO_CUDA_STREAMSYNCHRONIZE:
            cu_stream_synchronize(elem, tctx);
            break;
        case VIRTIO_CUDA_EVENTCREATE:
            cu_event_create(elem, tctx);
            break;
        case VIRTIO_CUDA_EVENTCREATEWITHFLAGS:
            cu_event_create_with_flags(elem, tctx);
            break;
        case VIRTIO_CUDA_EVENTDESTROY:
            cu_event_destroy(elem, tctx);
            break;
        case VIRTIO_CUDA_EVENTRECORD:
            cu_event_record(elem, tctx);
            break;
        case VIRTIO_CUDA_EVENTQUERY:
            cu_event_query(elem, tctx);
            break;
        case VIRTIO_CUDA_EVENTSYNCHRONIZE:
            cu_event_synchronize(elem, tctx);
            break;
        case VIRTIO_CUDA_EVENTELAPSEDTIME:
            cu_event_elapsedtime(elem, tctx);
            break;
        case VIRTIO_CUDA_THREADSYNCHRONIZE:
            cuda_thread_synchronize(elem, tctx);
            break;
        case VIRTIO_CUDA_GETLASTERROR:
            cuda_get_last_error(msg, tctx);
            break;
        case VIRTIO_CUDA_PEEKATLASTERROR:
            cuda_peek_at_last_error(msg, tctx);
            break;
        case VIRTIO_CUDA_MEMCPY_ASYNC:
            cuda_memcpy_async(msg, tctx);
            break;
        case VIRTIO_CUDA_MEMSET:
            cu_memset(elem, tctx);
            break;
        case VIRTIO_CUDA_DEVICESYNCHRONIZE:
            cuda_device_synchronize(elem, tctx);
            break;
        case VIRTIO_CUDA_MEMGETINFO:
            cuda_mem_get_info(elem, tctx);
            break;
        case VIRTIO_CUDA_MEMCPYTOSYMBOL:
            cu_memcpy_to_symbol(elem, tctx);
            break;
        case VIRTIO_CUDA_MEMCPYFROMSYMBOL:
            cu_memcpy_from_symbol(elem, tctx);
            break;
        case VIRTIO_CUBLAS_CREATE:
            cublas_create(elem, tctx);
            break;
        case VIRTIO_CUBLAS_DESTROY:
            cublas_destroy(elem, tctx);
            break;
        case VIRTIO_CUBLAS_SETVECTOR:
            cublas_set_vector(elem, tctx);
            break;
        case VIRTIO_CUBLAS_GETVECTOR:
            cublas_get_vector(elem, tctx);
            break;
        case VIRTIO_CUBLAS_SETMATRIX:
            cublas_set_matrix(elem, tctx);
            break;
        case VIRTIO_CUBLAS_GETMATRIX:
            cublas_get_matrix(elem, tctx);
            break;
        case VIRTIO_CUBLAS_SGEMM:
            cublas_sgemm(elem, tctx);
            break;
        case VIRTIO_CUBLAS_DGEMM:
            cublas_dgemm(elem, tctx);
            break;
        case VIRTIO_CUBLAS_SETSTREAM:
            cublas_set_stream(elem, tctx);
            break;
        case VIRTIO_CUBLAS_GETSTREAM:
            cublas_get_stream(elem, tctx);
            break;
        case VIRTIO_CUBLAS_SASUM:
            cublas_sasum(elem, tctx);
            break;
        case VIRTIO_CUBLAS_DASUM:
            cublas_dasum(elem, tctx);
            break;
        case VIRTIO_CUBLAS_SAXPY:
            cublas_saxpy(elem, tctx);
            break;
        case VIRTIO_CUBLAS_DAXPY:
            cublas_daxpy(elem, tctx);
            break;
        case VIRTIO_CUBLAS_SCOPY:
            cublas_scopy(elem, tctx);
            break;
        case VIRTIO_CUBLAS_DCOPY:
            cublas_dcopy(elem, tctx);
            break;
        case VIRTIO_CUBLAS_SGEMV:
            cublas_sgemv(elem, tctx);
            break;
        case VIRTIO_CUBLAS_DGEMV:
            cublas_dgemv(elem, tctx);
            break;
        case VIRTIO_CUBLAS_SDOT:
            cublas_sdot(elem, tctx);
            break;
        case VIRTIO_CUBLAS_DDOT:
            cublas_ddot(elem, tctx);
            break;
        case VIRTIO_CUBLAS_SSCAL:
            cublas_sscal(elem, tctx);
            break;
        case VIRTIO_CUBLAS_DSCAL:
            cublas_dscal(elem, tctx);
            break;
        case VIRTIO_CURAND_CREATEGENERATOR:
            curand_create_generator(msg, tctx);
            break;
        case VIRTIO_CURAND_CREATEGENERATORHOST:
            curand_create_generator_host(msg, tctx);
            break;
        case VIRTIO_CURAND_GENERATE:
            curand_generate(msg, tctx);
            break;
        case VIRTIO_CURAND_GENERATENORMAL:
            curand_generate_normal(msg, tctx);
            break;
        case VIRTIO_CURAND_GENERATENORMALDOUBLE:
            curand_generate_normal_double(msg, tctx);
            break;
        case VIRTIO_CURAND_GENERATEUNIFORM:
            curand_generate_uniform(msg, tctx);
            break;
        case VIRTIO_CURAND_GENERATEUNIFORMDOUBLE:
            curand_generate_uniform_double(msg, tctx);
            break;
        case VIRTIO_CURAND_DESTROYGENERATOR:
            curand_destroy_generator(msg, tctx);
            break;
        case VIRTIO_CURAND_SETGENERATOROFFSET:
            curand_set_generator_offset(msg, tctx);
            break;
        case VIRTIO_CURAND_SETPSEUDORANDOMSEED:
            curand_set_pseudorandom_seed(msg, tctx);
            break;
        case VIRTIO_SGX_MSG0:
            sgx_proc_msg0(elem, tctx);
            break;
        case VIRTIO_SGX_MSG1:
            sgx_proc_msg1(elem, tctx);
            break;
        case VIRTIO_SGX_MSG3:
            sgx_proc_msg3(elem, tctx);
            break;
        default:
            error("[+] header.cmd=%u, nr= %u \n",
                  msg->cmd, _IOC_NR(msg->cmd));
        return 0;
    }
    /*int ret = virtio_serial_write(port, (const uint8_t *)msg, len);
    if (ret < len) {
        error("write error. ret = %d\n", ret);
        virtio_serial_throttle_port(port, true);
    }
    debug("[+] WRITE BACK\n");*/
    return 0;
}


static void unload_module(ThreadContext *tctx)
{
    int i=0, idx=0;
    CudaContext *ctx = NULL;
    CudaModule *mod = NULL;

    func();
    for (idx = 0; idx < tctx->deviceCount; idx++) {
        if (idx==0 || tctx->deviceBitmap & 1<<idx) {
            ctx = &tctx->contexts[idx];
            for (i=0; i < ctx->moduleCount; i++) {
                mod = &ctx->modules[i];
                // debug("Unload module 0x%lx\n", mod->handle);
                for(int j=0; j<mod->cudaKernelsCount; j++) {
                    free(mod->cudaKernels[j].func_name);
                }
                for(int j=0; j<mod->cudaVarsCount; j++) {
                    free(mod->cudaVars[j].addr_name);
                }
                if (ctx->initialized)
                    cuError(cuModuleUnload(mod->module));
// #ifdef ENABLE_ENC
//                 free(mod->fatbin);
// #endif
                memset(mod, 0, sizeof(CudaModule));
            }
            ctx->moduleCount = 0;
            if(ctx->initialized) {
                // cuErrorExit(cuDevicePrimaryCtxRelease(ctx->dev));
#ifndef PRE_INIT_CTX
                cuError(cuCtxDestroy(ctx->context));
#endif
                deinit_primary_context(ctx);
            }
        }
    }
}

/* Callback function that's called when the guest opens/closes the port */
static void set_guest_connected(VirtIOSerialPort *port, int guest_connected)
{
    func();
    DeviceState *dev = DEVICE(port);
    // VirtIOSerial *vser = VIRTIO_CUDA(dev);
    debug("guest_connected=%d\n", guest_connected);

    if (dev->id) {
        qapi_event_send_vserport_change(dev->id, guest_connected,
                                        &error_abort);
    }
    if(guest_connected) {
        gettimeofday(&port->start_time, NULL);
    } else {
        ThreadContext *tctx = port->thread_context;
        unload_module(tctx);
        debug("clean\n");
        if (tctx->deviceBitmap != 0) {
            tctx->deviceCount   = total_device;
            tctx->cur_dev       = DEFAULT_DEVICE;
            memset(&tctx->deviceBitmap, 0, sizeof(tctx->deviceBitmap));
        }
        gettimeofday(&port->end_time, NULL);
        // double time_spent = (double)(port->end_time.tv_usec - port->start_time.tv_usec)/1000000 +
        //             (double)(port->end_time.tv_sec - port->start_time.tv_sec);
        // printf("port %d spends %f seconds\n", port->id, time_spent);
    }
}

/* 
 * Enable/disable backend for virtio serial port
 * default enable is whether vm is running.
 * When vm is running, enable backend, otherwise disable backend.
 */
static void virtconsole_enable_backend(VirtIOSerialPort *port, bool enable)
{
    func();
    debug("port id=%d, enable=%d\n", port->id, enable);

    if(!enable && global_deinitialized) {
        if (!total_port)
            return;
        /*
        * kill tid thread
        */
        debug("Closing thread %d !\n", port->id);
        // chan_close(port->thread_context->worker_queue);
        return;
    }
    if(enable && !global_deinitialized)
        global_deinitialized = 1;
    return ;
}

static void guest_writable(VirtIOSerialPort *port)
{
    func();
    return;
}

/* 
* Guest is now ready to accept data (virtqueues set up). 
* When the guest has asked us for this information it means
* the guest is all setup and has its virtqueues
* initialized. If some app is interested in knowing about
* this event, let it know.
* ESPECIALLY, when front driver insmod the driver.
*/
static void virtconsole_guest_ready(VirtIOSerialPort *port)
{
    VirtIOSerial *vser = port->vser;
    func();
    debug("port %d is ready.\n", port->id);
    qemu_mutex_lock(&total_port_mutex);
    total_port = get_active_ports_nr(vser);
    qemu_mutex_unlock(&total_port_mutex);
    if( total_port > WORKER_THREADS-1) {
        error("Too much ports, over %d\n", WORKER_THREADS);
        return;
    }
}

static void init_device_once(VirtIOSerial *vser)
{
    qemu_mutex_lock(&vser->init_mutex);
    if (global_initialized==1) {
        debug("global_initialized already!\n");
        qemu_mutex_unlock(&vser->init_mutex);
        return;
    }
    global_initialized = 1;
    global_deinitialized = 0;
    qemu_mutex_unlock(&vser->init_mutex);
    total_port = 0;
    qemu_mutex_init(&total_port_mutex);
    cuError( cuInit(0));
    if(!vser->gcount) {
        cuError(cuDeviceGetCount(&total_device));
    }
    else {
        total_device = vser->gcount;
    }
    debug("vser->gcount(or total_device)=%d\n", total_device);
}

static void deinit_device_once(VirtIOSerial *vser)
{
    qemu_mutex_lock(&vser->deinit_mutex);
    if(global_deinitialized==0) {
        qemu_mutex_unlock(&vser->deinit_mutex);
        return;
    }
    global_deinitialized=0;
    qemu_mutex_unlock(&vser->deinit_mutex);
    func();

    qemu_mutex_destroy(&total_port_mutex);
}

static void init_port(VirtIOSerialPort *port)
{
    ThreadContext *tctx = NULL;
    port->thread_context = malloc(sizeof(ThreadContext));
    CudaContext *ctx = NULL;
    tctx = port->thread_context;
    tctx->deviceCount = total_device;
    tctx->cur_dev = DEFAULT_DEVICE;
    memset(&tctx->deviceBitmap, 0, sizeof(tctx->deviceBitmap));
    tctx->contexts = malloc(sizeof(CudaContext) * tctx->deviceCount);
    for (int i = 0; i < tctx->deviceCount; i++)
    {
        ctx = &tctx->contexts[i];
        ctx->moduleCount = 0;
        ctx->initialized = 0;
        ctx->tctx        = tctx;
        cuErrorExit(cuDeviceGet(&ctx->dev, i));
#ifdef PRE_INIT_CTX
        cuErrorExit(cuCtxCreate(&ctx->context, 0, ctx->dev));
        debug("devID %d, create context %p\n", ctx->dev, ctx->context);
#endif
    }
    // tctx->worker_queue  = chan_init(100);
}

static void deinit_port(VirtIOSerialPort *port)
{
    ThreadContext *tctx = port->thread_context;
    /*delete elements in struct list_head*/
    /*
    */
#ifdef PRE_INIT_CTX
    for (int i = 0; i < tctx->deviceCount; i++)
    {
        CudaContext *ctx = &tctx->contexts[i];
        cuErrorExit(cuCtxDestroy(ctx->context));
        debug("devID %d, destroy context %p\n", i, ctx->context);
    }
#endif
    free(tctx->contexts);
    tctx->contexts = NULL;
    // chan_dispose(tctx->worker_queue);
    // debug("Ending thread %d computing workloads and queue!\n", port->id);
    // qemu_thread_join(&tctx->worker_thread);
    free(tctx);
    return;
}
/*
 * The per-port (or per-app) realize function that's called when a
 * new device is found on the bus.
*/
static void virtconsole_realize(DeviceState *dev, Error **errp)
{
    func();
    VirtIOSerialPort *port = VIRTIO_SERIAL_PORT(dev);
    VirtIOSerial *vser = port->vser;
    VirtIOSerialPortClass *k = VIRTIO_SERIAL_PORT_GET_CLASS(dev);
    debug("port->id = %d\n", port->id );
    if (port->id == 0 && !k->is_console) {
        error_setg(errp, "Port number 0 on virtio-serial devices reserved "
                   "for virtconsole devices for backward compatibility.");
        return;
    }

    virtio_serial_open(port);

    /* init GPU device
    */
    init_device_once(vser);
    init_port(port);
}

/*
* Per-port unrealize function that's called when a port gets
* hot-unplugged or removed.
*/
static void virtconsole_unrealize(DeviceState *dev, Error **errp)
{
    func();
    VirtConsole *vcon = VIRTIO_CONSOLE(dev);
    VirtIOSerialPort *port = VIRTIO_SERIAL_PORT(dev);
    VirtIOSerial *vser = port->vser;

    deinit_device_once(vser);
    deinit_port(port);
    if (vcon->watch) {
        g_source_remove(vcon->watch);
    }

}

static Property virtserialport_properties[] = {
    DEFINE_PROP_CHR("chardev", VirtConsole, chr),
    DEFINE_PROP_STRING("privatekeypath", VirtConsole, privatekeypath),
    DEFINE_PROP_STRING("hmac_path", VirtConsole, hmac_path), 
    DEFINE_PROP_END_OF_LIST(),
};

static void virtserialport_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    VirtIOSerialPortClass *k = VIRTIO_SERIAL_PORT_CLASS(klass);

    k->realize = virtconsole_realize;
    k->unrealize = virtconsole_unrealize;
    k->have_data = flush_buf;
    k->handle_data = handle_output;
    k->set_guest_connected = set_guest_connected;
    k->enable_backend = virtconsole_enable_backend;
    k->guest_ready = virtconsole_guest_ready;
    k->guest_writable = guest_writable;
    dc->props = virtserialport_properties;
}

static const TypeInfo virtserialport_info = {
    .name          = TYPE_VIRTIO_CONSOLE_SERIAL_PORT,
    .parent        = TYPE_VIRTIO_SERIAL_PORT,
    .instance_size = sizeof(VirtConsole),
    .class_init    = virtserialport_class_init,
};

static void virtconsole_register_types(void)
{
    type_register_static(&virtserialport_info);
}

type_init(virtconsole_register_types)
