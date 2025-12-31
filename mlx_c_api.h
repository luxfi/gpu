// MLX C API Header for CGO bindings
// This provides a C interface to the C++ MLX library

#ifndef MLX_C_API_H
#define MLX_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

// Check for backend support
bool mlx_has_metal();
bool mlx_has_cuda();

// Device information
char* mlx_get_metal_device_name();
char* mlx_get_cuda_device_name();
size_t mlx_get_metal_memory();
size_t mlx_get_cuda_memory();
size_t mlx_get_system_memory();

// Array operations
void* mlx_zeros(int* shape, int ndim, int dtype);
void* mlx_ones(int* shape, int ndim, int dtype);
void* mlx_random(int* shape, int ndim, int dtype);
void* mlx_arange(double start, double stop, double step);

// Binary operations
void* mlx_add(void* a, void* b);
void* mlx_subtract(void* a, void* b);
void* mlx_multiply(void* a, void* b);
void* mlx_divide(void* a, void* b);
void* mlx_matmul(void* a, void* b);
void* mlx_remainder(void* a, void* b);
void* mlx_right_shift(void* a, void* b);

// Comparison operations
void* mlx_greater_equal(void* a, void* b);
void* mlx_less(void* a, void* b);

// Reduction operations
void* mlx_sum(void* array, int* axis, int naxis);
void* mlx_mean(void* array, int* axis, int naxis);

// Evaluation and synchronization
void mlx_eval(void* arrays[], int count);
void mlx_synchronize();

// Stream management
void* mlx_new_stream();
void mlx_free_stream(void* stream);

// Memory management
void mlx_free_array(void* array);

// Array creation from data
void* mlx_from_slice(float* data, int data_len, int* shape, int ndim, int dtype);
void* mlx_from_slice_int64(long long* data, int data_len, int* shape, int ndim, int dtype);

// Element-wise maximum
void* mlx_maximum(void* a, void* b);

// Unary operations
void* mlx_floor(void* a);
void* mlx_round(void* a);

// Type conversion
void* mlx_astype(void* a, int dtype);

// Shape operations
void* mlx_full(int* shape, int ndim, double value, int dtype);
void* mlx_reshape(void* a, int* shape, int ndim);
void* mlx_squeeze(void* a, int axis);
void* mlx_broadcast(void* a, int* shape, int ndim);

// Indexing operations
void* mlx_slice_nd(void* a, int* starts, int* stops, int ndim);
void* mlx_take(void* a, void* indices, int axis);
void* mlx_take_along_axis(void* a, void* indices, int axis);
void* mlx_concatenate(void** arrays, int num_arrays, int axis);

// Conditional operations
void* mlx_where(void* cond, void* a, void* b);

// Data extraction
int mlx_to_slice_int64(void* a, long long* out, int max_len);

#ifdef __cplusplus
}
#endif

#endif // MLX_C_API_H

