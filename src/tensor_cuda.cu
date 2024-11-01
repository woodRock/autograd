// tensor_cuda.cu
#include <cuda_runtime.h>
#include <cstring>  // for memcpy
#include <memory>
#include <vector>
#include <iostream>
#include <random>
#include <stdexcept>

// Forward declaration of Tensor class to match tensor.cpp
class Tensor : public std::enable_shared_from_this<Tensor> {
private:
    // Random number generator for initialization
    static std::mt19937& get_generator();
    static std::vector<float> generate_random_data(size_t rows, size_t cols);

public:
    // Dimensions and data
    size_t rows;
    size_t cols;
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<std::shared_ptr<Tensor>> parents;
    bool requires_grad;
    std::string creation_op;

    // Constructors
    Tensor(size_t rows, size_t cols, bool requires_grad = false, std::string creation_op = "");
    Tensor(const std::vector<float>& data, size_t rows, size_t cols, bool requires_grad = false, std::string creation_op = "");
    
    // Destructor
    ~Tensor();

    // Graph operations
    void clear_graph();
    void backward();
    void debug_gradient_flow();

    // Static operations
    static std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    static std::shared_ptr<Tensor> multiply(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    static std::shared_ptr<Tensor> binary_cross_entropy(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target);
    static std::shared_ptr<Tensor> cross_entropy(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target);

    // Tensor operations
    std::shared_ptr<Tensor> reshape(size_t new_rows, size_t new_cols);
    std::shared_ptr<Tensor> sum();
    std::shared_ptr<Tensor> expand(size_t dim, size_t copies);
    std::shared_ptr<Tensor> transpose();
    
    // Activation functions
    std::shared_ptr<Tensor> relu();
    std::shared_ptr<Tensor> Tanh();
    std::shared_ptr<Tensor> sigmoid();
    std::shared_ptr<Tensor> softmax();

    // Utility methods
    std::vector<size_t> shape();
    void print_grad();

    // Operator overloads
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
};

class Layer {
public:
    // Member to hold parameters (weights, biases, etc.)
    std::shared_ptr<Tensor> parameters;

    // Default constructor
    Layer();

    // Constructor with parameters
    Layer(std::shared_ptr<Tensor> parameters);

    // Method to get the parameters
    std::shared_ptr<Tensor> get_parameters();

    // Virtual method for the forward pass; must be implemented in derived classes
    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input);
};

class Conv2D : public Layer {
public:
    // Member variables
    std::shared_ptr<Tensor> weights;  // Weights of the convolution layer
    std::shared_ptr<Tensor> bias;     // Bias of the convolution layer
    size_t in_channels;                // Number of input channels
    size_t out_channels;               // Number of output channels
    size_t kernel_size;                // Size of the convolution kernel
    size_t stride;                     // Stride for the convolution
    size_t padding;                    // Padding applied to input

    // Constructor
    Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size, 
           size_t stride = 1, size_t padding = 0);

    // Forward pass function
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input);

    // Function to get the parameters (weights and bias)
    std::vector<std::shared_ptr<Tensor>> get_parameters();
};

class MaxPool2D : public Layer {
public:
    // Member variables
    size_t kernel_size;
    size_t stride;

    // Constructor
    MaxPool2D(size_t kernel_size, size_t stride);

    // Forward pass
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input);

    // Get parameters (empty for MaxPool2D)
    std::vector<std::shared_ptr<Tensor>> get_parameters();
};

// CUDA error checking macro
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA kernel for matrix multiplication
__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C,
                                     int A_rows, int A_cols, int B_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows && col < B_cols) {
        float sum = 0.0f;
        for (int k = 0; k < A_cols; k++) {
            sum += A[row * A_cols + k] * B[k * B_cols + col];
        }
        C[row * B_cols + col] = sum;
    }
}

// CUDA kernels for backward pass
__global__ void matrix_multiply_backward_weights_kernel(
    const float* input_data, const float* grad_output,
    float* grad_weights, int batch_size, int input_features,
    int output_features) {
    
    int in_feature = blockIdx.x * blockDim.x + threadIdx.x;
    int out_feature = blockIdx.y * blockDim.y + threadIdx.y;

    if (in_feature < input_features && out_feature < output_features) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            sum += input_data[b * input_features + in_feature] * 
                   grad_output[b * output_features + out_feature];
        }
        grad_weights[in_feature * output_features + out_feature] = sum;
    }
}

__global__ void matrix_multiply_backward_inputs_kernel(
    const float* weights_data, const float* grad_output,
    float* grad_inputs, int batch_size, int input_features,
    int output_features) {
    
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int in_feature = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < batch_size && in_feature < input_features) {
        float sum = 0.0f;
        for (int out_feature = 0; out_feature < output_features; out_feature++) {
            sum += grad_output[b * output_features + out_feature] * 
                   weights_data[in_feature * output_features + out_feature];
        }
        grad_inputs[b * input_features + in_feature] = sum;
    }
}

// Device memory manager
class CUDAMemory {
private:
    float* device_ptr = nullptr;
    size_t size = 0;

public:
    CUDAMemory(size_t num_elements) : size(num_elements) {
        CUDA_CHECK(cudaMalloc(&device_ptr, size * sizeof(float)));
    }

    ~CUDAMemory() {
        if (device_ptr) {
            cudaFree(device_ptr);
        }
    }

    void copyToDevice(const float* host_data, size_t num_elements) {
        CUDA_CHECK(cudaMemcpy(device_ptr, host_data, 
                             num_elements * sizeof(float), 
                             cudaMemcpyHostToDevice));
    }

    void copyToHost(float* host_data, size_t num_elements) {
        CUDA_CHECK(cudaMemcpy(host_data, device_ptr, 
                             num_elements * sizeof(float), 
                             cudaMemcpyDeviceToHost));
    }

    float* get() { return device_ptr; }
};

// Implementation of Tensor::multiply using CUDA
std::shared_ptr<Tensor> Tensor::multiply(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->cols != b->rows) {
        throw std::runtime_error("Matrix dimensions not compatible for multiplication");
    }

    // Create device memory
    CUDAMemory d_a(a->rows * a->cols);
    CUDAMemory d_b(b->rows * b->cols);
    CUDAMemory d_c(a->rows * b->cols);

    // Copy inputs to device
    d_a.copyToDevice(a->data.data(), a->rows * a->cols);
    d_b.copyToDevice(b->data.data(), b->rows * b->cols);

    // Set kernel dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (b->cols + blockDim.x - 1) / blockDim.x,
        (a->rows + blockDim.y - 1) / blockDim.y
    );

    // Launch kernel
    matrix_multiply_kernel<<<gridDim, blockDim>>>(
        d_a.get(), d_b.get(), d_c.get(),
        a->rows, a->cols, b->cols
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    std::vector<float> result_data(a->rows * b->cols);
    d_c.copyToHost(result_data.data(), result_data.size());

    // Create result tensor
    auto result = std::make_shared<Tensor>(result_data, a->rows, b->cols,
                                         a->requires_grad || b->requires_grad, "mul");
    result->parents.push_back(a);
    result->parents.push_back(b);
    return result;
}

// CUDA kernel for addition backward pass
__global__ void add_backward_kernel(const float* grad_output, float* grad_a, float* grad_b, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        grad_a[index] = grad_output[index]; // Gradient for a
        grad_b[index] = grad_output[index]; // Gradient for b
    }
}

// CUDA kernel for ReLU backward pass
__global__ void relu_backward_kernel(const float* input, const float* grad_output, float* grad_input, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        grad_input[index] = (input[index] > 0) ? grad_output[index] : 0; // Gradient is passed through if input > 0
    }
}

__global__ void softmax_backward_kernel(const float* output, const float* grad_output, float* grad_input, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        // Each thread computes its own gradient
        float sum = 0.0f;
        for (int j = 0; j < size; j++) {
            sum += output[j] * grad_output[j];
        }
        grad_input[index] = output[index] * (grad_output[index] - sum);
    }
}

// CUDA kernel for binary cross-entropy backward pass
__global__ void bce_backward_kernel(const float* targets, const float* outputs, const float* grad_output, float* grad_inputs, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        // Gradient w.r.t. outputs
        grad_inputs[index] = -(targets[index] / (outputs[index] + 1e-15f) - (1 - targets[index]) / (1 - outputs[index] + 1e-15f)) * grad_output[index];
    }
}

__global__ void cross_entropy_forward_kernel(
    const float* predictions, 
    const float* targets,
    float* output,
    size_t batch_size,
    size_t num_classes) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        float loss = 0.0f;
        
        // Calculate cross entropy for this sample
        for (size_t c = 0; c < num_classes; c++) {
            size_t idx = batch_idx * num_classes + c;
            // Add small epsilon to prevent log(0)
            float pred = fmaxf(predictions[idx], 1e-7f);
            pred = fminf(pred, 1.0f - 1e-7f);
            loss += -targets[idx] * logf(pred);
        }
        
        // Store the loss for this sample
        output[batch_idx] = loss;
    }
}

// Forward method implementation
std::shared_ptr<Tensor> Tensor::cross_entropy(std::shared_ptr<Tensor> predictions, std::shared_ptr<Tensor> targets) {
    
    size_t batch_size = predictions->rows;
    size_t num_classes = predictions->cols;
    
    // Verify input dimensions match
    if (predictions->rows != targets->rows || predictions->cols != targets->cols) {
        throw std::runtime_error("Predictions and targets dimensions must match");
    }
    
    // Allocate device memory
    CUDAMemory d_predictions(predictions->data.size());
    CUDAMemory d_targets(targets->data.size());
    CUDAMemory d_output(batch_size);
    
    // Copy data to device
    d_predictions.copyToDevice(predictions->data.data(), predictions->data.size());
    d_targets.copyToDevice(targets->data.data(), targets->data.size());
    
    // Set kernel dimensions
    int block_size = 256;
    int num_blocks = (batch_size + block_size - 1) / block_size;
    
    // Launch kernel
    cross_entropy_forward_kernel<<<num_blocks, block_size>>>(
        d_predictions.get(),
        d_targets.get(),
        d_output.get(),
        batch_size,
        num_classes
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    std::vector<float> output_data(batch_size);
    d_output.copyToHost(output_data.data(), output_data.size());
    
    // Calculate mean loss across batch
    float mean_loss = 0.0f;
    for (size_t i = 0; i < batch_size; i++) {
        mean_loss += output_data[i];
    }
    mean_loss /= static_cast<float>(batch_size);
    
    // Create result tensor (scalar output)
    std::vector<float> final_output = {mean_loss};
    auto output = std::make_shared<Tensor>(final_output, 1, 1, 
                                         predictions->requires_grad, "ce");
    
    if (predictions->requires_grad) {
        output->parents = {predictions, targets};
    }
    
    return output;
}

// CUDA kernel for categorical cross-entropy backward pass
__global__ void cross_entropy_backward_kernel(const float* targets, const float* outputs, 
                                              const float* grad_output, float* grad_inputs, 
                                              int size, int num_classes) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        // Assuming targets are one-hot encoded
        int class_index = index % num_classes;  // This assumes the output is shaped as (batch_size, num_classes)
        grad_inputs[index] = (outputs[index] - targets[index]) * grad_output[class_index];
    }
}


// CUDA kernel for convolution forward pass
#define TILE_SIZE 16
#define MAX_KERNEL_SIZE 7  // Adjust based on your max kernel size

// Optimized CNN CUDA implementation

// 1. Use shared memory for input and kernel tiles
#define TILE_SIZE 16
#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024
#define ELEMENTS_PER_THREAD 4

// Optimized convolution forward kernel with improved memory access patterns
__global__ void conv2d_forward_kernel_optimized(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const size_t batch_size,
    const size_t in_channels,
    const size_t out_channels,
    const size_t height,
    const size_t width,
    const size_t kernel_size,
    const size_t stride,
    const size_t padding,
    const size_t output_height,
    const size_t output_width) {
    
    // Shared memory for input and weight tiles
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weights = shared_mem + TILE_SIZE * TILE_SIZE;
    
    // Calculate thread and block indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    
    // Output position
    const int batch_idx = bx;
    const int out_channel = by;
    const int out_y = (bz / output_width) * TILE_SIZE + ty;
    const int out_x = (bz % output_width) * TILE_SIZE + tx;
    
    // Register to accumulate partial results
    float partial_sum = 0.0f;
    
    // Loop over input channels
    for (int in_c = 0; in_c < in_channels; ++in_c) {
        // Load input tile into shared memory
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i += WARP_SIZE) {
            int y = out_y * stride - padding + i;
            int x = out_x * stride - padding;
            
            if (y >= 0 && y < height && x >= 0 && x < width) {
                shared_input[ty * TILE_SIZE + tx] = input[
                    batch_idx * (in_channels * height * width) +
                    in_c * (height * width) +
                    y * width + x
                ];
            } else {
                shared_input[ty * TILE_SIZE + tx] = 0.0f;
            }
        }
        
        // Load weights into shared memory
        if (tx < kernel_size && ty < kernel_size) {
            shared_weights[ty * kernel_size + tx] = weights[
                out_channel * (in_channels * kernel_size * kernel_size) +
                in_c * (kernel_size * kernel_size) +
                ty * kernel_size + tx
            ];
        }
        
        __syncthreads();
        
        // Compute convolution for this tile
        #pragma unroll
        for (int ky = 0; ky < kernel_size; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < kernel_size; ++kx) {
                int y = out_y * stride - padding + ky;
                int x = out_x * stride - padding + kx;
                
                if (y >= 0 && y < height && x >= 0 && x < width) {
                    partial_sum += shared_input[(ty + ky) * TILE_SIZE + (tx + kx)] *
                                 shared_weights[ky * kernel_size + kx];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write output with vectorized memory access
    if (out_y < output_height && out_x < output_width) {
        int output_idx = 
            batch_idx * (out_channels * output_height * output_width) +
            out_channel * (output_height * output_width) +
            out_y * output_width + out_x;
        
        output[output_idx] = partial_sum + bias[out_channel];
    }
}

// Optimized implementation of Conv2D forward pass
std::shared_ptr<Tensor> Conv2D::forward(std::shared_ptr<Tensor> input) {
    size_t batch_size = input->rows;
    size_t height = static_cast<size_t>(std::sqrt(input->cols / in_channels));
    size_t width = height;
    
    size_t output_height = (height + 2 * padding - kernel_size) / stride + 1;
    size_t output_width = (width + 2 * padding - kernel_size) / stride + 1;
    
    // Create CUDA stream for asynchronous operations
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    
    // Use pinned memory for faster transfers
    float *h_input, *h_weights, *h_bias, *h_output;
    cudaHostAlloc(&h_input, input->data.size() * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_weights, weights->data.size() * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_bias, bias->data.size() * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_output, batch_size * out_channels * output_height * output_width * sizeof(float), 
                  cudaHostAllocDefault);
    
    // Copy data to pinned memory
    std::memcpy(h_input, input->data.data(), input->data.size() * sizeof(float));
    std::memcpy(h_weights, weights->data.data(), weights->data.size() * sizeof(float));
    std::memcpy(h_bias, bias->data.data(), bias->data.size() * sizeof(float));
    
    // Allocate device memory
    float *d_input, *d_weights, *d_bias, *d_output;
    cudaMalloc(&d_input, input->data.size() * sizeof(float));
    cudaMalloc(&d_weights, weights->data.size() * sizeof(float));
    cudaMalloc(&d_bias, bias->data.size() * sizeof(float));
    cudaMalloc(&d_output, batch_size * out_channels * output_height * output_width * sizeof(float));
    
    // Asynchronous memory transfers
    cudaMemcpyAsync(d_input, h_input, input->data.size() * sizeof(float), 
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_weights, h_weights, weights->data.size() * sizeof(float), 
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_bias, h_bias, bias->data.size() * sizeof(float), 
                    cudaMemcpyHostToDevice, stream);
    
    // Calculate optimal grid and block dimensions
    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim(
        batch_size,
        out_channels,
        (output_height * output_width + TILE_SIZE * TILE_SIZE - 1) / (TILE_SIZE * TILE_SIZE)
    );
    
    // Calculate shared memory size
    size_t shared_mem_size = (TILE_SIZE * TILE_SIZE + kernel_size * kernel_size) * sizeof(float);
    
    // Launch kernel with stream
    conv2d_forward_kernel_optimized<<<grid_dim, block_dim, shared_mem_size, stream>>>(
        d_input, d_weights, d_bias, d_output,
        batch_size, in_channels, out_channels,
        height, width, kernel_size,
        stride, padding, output_height, output_width
    );
    
    // Asynchronous copy back to host
    cudaMemcpyAsync(h_output, d_output, 
                    batch_size * out_channels * output_height * output_width * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    
    // Synchronize stream
    cudaStreamSynchronize(stream);
    
    // Create output tensor
    std::vector<float> output_data(h_output, 
                                 h_output + batch_size * out_channels * output_height * output_width);
    
    // Cleanup
    cudaFreeHost(h_input);
    cudaFreeHost(h_weights);
    cudaFreeHost(h_bias);
    cudaFreeHost(h_output);
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
    
    auto output = std::make_shared<Tensor>(output_data, batch_size, 
                                         out_channels * output_height * output_width,
                                         true, "Conv2d");
    output->parents = {input, weights, bias};
    return output;
}

// CUDA kernel for convolution backward pass - input gradients
__global__ void conv2d_backward_input_kernel(
    const float* weights, const float* grad_output,
    float* grad_input, size_t batch_size, size_t in_channels, size_t out_channels,
    size_t height, size_t width, size_t kernel_size, size_t stride, size_t padding,
    size_t output_height, size_t output_width) {
    
    int b = blockIdx.x;  // batch index
    int ic = blockIdx.y; // input channel index
    int h = threadIdx.x; // height index
    int w = threadIdx.y; // width index
    
    if (b >= batch_size || ic >= in_channels || h >= height || w >= width)
        return;
        
    float sum = 0.0f;
    
    // For each output channel
    #pragma omp parallel for collapse(3)
    for (int oc = 0; oc < out_channels; oc++) {
        // For each kernel position that could have contributed to this input
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int oh = (h + padding - kh) / stride;
                int ow = (w + padding - kw) / stride;
                
                if (oh >= 0 && oh < output_height && ow >= 0 && ow < output_width &&
                    (oh * stride + kh - padding) == h && (ow * stride + kw - padding) == w) {
                    
                    int grad_output_idx = b * (out_channels * output_height * output_width) +
                                        oc * (output_height * output_width) +
                                        oh * output_width + ow;
                    int weight_idx = oc * (in_channels * kernel_size * kernel_size) +
                                   ic * (kernel_size * kernel_size) +
                                   kh * kernel_size + kw;
                    
                    sum += grad_output[grad_output_idx] * weights[weight_idx];
                }
            }
        }
    }
    
    int input_idx = b * (in_channels * height * width) +
                    ic * (height * width) +
                    h * width + w;
    grad_input[input_idx] = sum;
}

// CUDA kernel for convolution backward pass - weight gradients
__global__ void conv2d_backward_weights_kernel(
    const float* input, const float* grad_output,
    float* grad_weights, size_t batch_size, size_t in_channels, size_t out_channels,
    size_t height, size_t width, size_t kernel_size, size_t stride, size_t padding,
    size_t output_height, size_t output_width) {
    
    int oc = blockIdx.x;  // output channel index
    int ic = blockIdx.y;  // input channel index
    int kh = threadIdx.x; // kernel height index
    int kw = threadIdx.y; // kernel width index
    
    if (oc >= out_channels || ic >= in_channels || kh >= kernel_size || kw >= kernel_size)
        return;
        
    float sum = 0.0f;
    
    // For each batch
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch_size; b++) {
        // For each output position
        for (int oh = 0; oh < output_height; oh++) {
            for (int ow = 0; ow < output_width; ow++) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;
                
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int input_idx = b * (in_channels * height * width) +
                                  ic * (height * width) +
                                  ih * width + iw;
                    int grad_output_idx = b * (out_channels * output_height * output_width) +
                                        oc * (output_height * output_width) +
                                        oh * output_width + ow;
                    
                    sum += input[input_idx] * grad_output[grad_output_idx];
                }
            }
        }
    }
    
    int weight_idx = oc * (in_channels * kernel_size * kernel_size) +
                     ic * (kernel_size * kernel_size) +
                     kh * kernel_size + kw;
    grad_weights[weight_idx] = sum;
}

// CUDA kernel for maxpool2d forward pass
__global__ void maxpool2d_forward_kernel(
    const float* input,
    float* output,
    size_t batch_size,
    size_t channels,
    size_t height,
    size_t width,
    size_t kernel_size,
    size_t stride,
    size_t output_height,
    size_t output_width) {
    
    int b = blockIdx.x;  // batch index
    int c = blockIdx.y;  // channel index
    int oh = blockIdx.z / output_width;  // output height index
    int ow = blockIdx.z % output_width;  // output width index
    
    if (b >= batch_size || c >= channels || oh >= output_height || ow >= output_width)
        return;
    
    // Calculate the window boundaries in the input
    int h_start = oh * stride;
    int w_start = ow * stride;
    int h_end = min(h_start + kernel_size, height);
    int w_end = min(w_start + kernel_size, width);
    
    // Find maximum value in the window
    float max_val = -INFINITY;
    
    #pragma omp parallel for collapse(2)
    for (int h = h_start; h < h_end; h++) {
        for (int w = w_start; w < w_end; w++) {
            int input_idx = b * (channels * height * width) +
                           c * (height * width) +
                           h * width + w;
            max_val = max(max_val, input[input_idx]);
        }
    }
    
    // Write output
    int output_idx = b * (channels * output_height * output_width) +
                     c * (output_height * output_width) +
                     oh * output_width + ow;
    output[output_idx] = max_val;
}

// Method to perform forward pass for MaxPool2D
std::shared_ptr<Tensor> MaxPool2D::forward(std::shared_ptr<Tensor> input) {
    size_t batch_size = input->rows;
    size_t channels = 32;  // Assuming input from Conv2D with 32 channels
    size_t height = static_cast<size_t>(std::sqrt(input->cols / channels));
    size_t width = height;
    
    size_t output_height = (height - kernel_size) / stride + 1;
    size_t output_width = (width - kernel_size) / stride + 1;
    
    // Create device memory
    CUDAMemory d_input(input->data.size());
    CUDAMemory d_output(batch_size * channels * output_height * output_width);
    
    // Copy input to device
    d_input.copyToDevice(input->data.data(), input->data.size());
    
    // Set kernel dimensions
    dim3 gridDim(batch_size, channels, output_height * output_width);
    dim3 blockDim(1, 1, 1);
    
    // Launch kernel
    maxpool2d_forward_kernel<<<gridDim, blockDim>>>(
        d_input.get(),
        d_output.get(),
        batch_size,
        channels,
        height,
        width,
        kernel_size,
        stride,
        output_height,
        output_width
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    std::vector<float> output_data(batch_size * channels * output_height * output_width);
    d_output.copyToHost(output_data.data(), output_data.size());
    
    // Create result tensor
    auto output = std::make_shared<Tensor>(output_data, batch_size, 
                                         channels * output_height * output_width,
                                         input->requires_grad, "maxpool2d");
    if (input->requires_grad) {
        output->parents = {input};
    }
    return output;
}

__global__ void maxpool2d_backward_kernel(
    const float* input_data,
    float* grad_input,
    const float* grad_output,
    size_t batch_size,
    size_t channels,
    size_t height,
    size_t width,
    size_t kernel_size,
    size_t stride,
    size_t new_height,
    size_t new_width) {
    
    int b = blockIdx.x;
    int c = blockIdx.y;
    int h = threadIdx.x;
    int w = threadIdx.y;

    if (b >= batch_size || c >= channels || h >= height || w >= width)
        return;

    size_t oh = h / stride;
    size_t ow = w / stride;
    
    if (oh < new_height && ow < new_width) {
        // Get current input value
        float current_val = input_data[b * (channels * height * width) +
                                     c * (height * width) +
                                     h * width + w];
        
        // Check if this was the max value
        bool is_max = true;
        
        // Compare with all values in the kernel window
        #pragma omp parallel for collapse(2)
        for (size_t kh = 0; kh < kernel_size && is_max; kh++) {
            for (size_t kw = 0; kw < kernel_size && is_max; kw++) {
                size_t h_index = oh * stride + kh;
                size_t w_index = ow * stride + kw;
                
                if (h_index < height && w_index < width) {
                    float val = input_data[b * (channels * height * width) +
                                         c * (height * width) +
                                         h_index * width + w_index];
                    if (val > current_val) {
                        is_max = false;
                    }
                }
            }
        }
        
        // If this was the max value, propagate the gradient
        if (is_max) {
            size_t grad_idx = b * (channels * new_height * new_width) +
                             c * (new_height * new_width) +
                             oh * new_width + ow;
            atomicAdd(&grad_input[b * (channels * height * width) +
                                c * (height * width) +
                                h * width + w],
                     grad_output[grad_idx]);
        }
    }
}

// Kernel to compute mean for each feature
__global__ void batchnorm_mean_kernel(
    const float* input,
    float* mean,
    size_t batch_size,
    size_t num_features) {
    
    int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= num_features) return;

    float sum = 0.0f;
    for (int batch = 0; batch < batch_size; batch++) {
        sum += input[batch * num_features + feature];
    }
    mean[feature] = sum / batch_size;
}

// Kernel to compute variance for each feature
__global__ void batchnorm_var_kernel(
    const float* input,
    const float* mean,
    float* var,
    size_t batch_size,
    size_t num_features) {
    
    int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= num_features) return;

    float sum = 0.0f;
    float mean_val = mean[feature];
    
    for (int batch = 0; batch < batch_size; batch++) {
        float diff = input[batch * num_features + feature] - mean_val;
        sum += diff * diff;
    }
    var[feature] = sum / batch_size;
}

// Kernel for the main batchnorm forward computation
__global__ void batchnorm_forward_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    const float* mean,
    const float* var,
    float* output,
    size_t batch_size,
    size_t num_features,
    float epsilon) {
    
    int batch = blockIdx.x;
    int feature = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch >= batch_size || feature >= num_features) return;
    
    int idx = batch * num_features + feature;
    float normalized = (input[idx] - mean[feature]) / sqrt(var[feature] + epsilon);
    output[idx] = gamma[feature] * normalized + beta[feature];
}

// Kernel for backward pass
__global__ void batchnorm_backward_kernel(
    const float* input,
    const float* grad_output,
    const float* gamma,
    const float* mean,
    const float* var,
    float* grad_input,
    float* grad_gamma,
    float* grad_beta,
    size_t batch_size,
    size_t num_features,
    float epsilon) {
    
    int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= num_features) return;

    float sum_grad = 0.0f;
    float sum_grad_h = 0.0f;
    float sum_grad_h_input = 0.0f;
    
    float mean_val = mean[feature];
    float var_val = var[feature];
    float inv_std = 1.0f / sqrt(var_val + epsilon);
    
    // First pass to compute sums
    for (int batch = 0; batch < batch_size; batch++) {
        int idx = batch * num_features + feature;
        float x_centered = input[idx] - mean_val;
        float h = x_centered * inv_std;
        
        sum_grad += grad_output[idx];
        sum_grad_h += grad_output[idx] * h;
        sum_grad_h_input += grad_output[idx] * x_centered;
    }
    
    // Update gamma and beta gradients
    grad_gamma[feature] = sum_grad_h;
    grad_beta[feature] = sum_grad;
    
    // Second pass to compute input gradients
    float gamma_val = gamma[feature];
    float factor = gamma_val * inv_std / batch_size;
    
    for (int batch = 0; batch < batch_size; batch++) {
        int idx = batch * num_features + feature;
        float x_centered = input[idx] - mean_val;
        
        grad_input[idx] = factor * (
            batch_size * grad_output[idx] 
            - sum_grad 
            - (x_centered * inv_std * inv_std * sum_grad_h_input)
        );
    }
}

// Extension of Tensor::backward() for CUDA multiplication
void Tensor::backward() {
    if (!requires_grad) return;

    if (grad.empty()) {
        grad = std::vector<float>(rows * cols, 1.0f);
    }

    if (creation_op == "add") {
        auto a = parents[0];
        auto b = parents[1];

        // Allocate device memory
        CUDAMemory d_grad_output(grad.size());
        CUDAMemory d_grad_a(a->grad.size());
        CUDAMemory d_grad_b(b->grad.size());

        // Copy data to device
        d_grad_output.copyToDevice(grad.data(), grad.size());

        int blockSize = 256;
        int numBlocks = (a->data.size() + blockSize - 1) / blockSize;

        // Launch backward kernel for addition
        add_backward_kernel<<<numBlocks, blockSize>>>(d_grad_output.get(), d_grad_a.get(), d_grad_b.get(), a->data.size());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy gradients back to host
        d_grad_a.copyToHost(a->grad.data(), a->grad.size());
        d_grad_b.copyToHost(b->grad.data(), b->grad.size());

        // Continue backprop
        a->backward();
        b->backward();
    }
    if (creation_op == "mul") {
        auto inputs = parents[0];
        auto weights = parents[1];

        // Allocate device memory
        CUDAMemory d_inputs_data(inputs->data.size());
        CUDAMemory d_weights_data(weights->data.size());
        CUDAMemory d_grad_output(grad.size());
        CUDAMemory d_grad_inputs(inputs->grad.size());
        CUDAMemory d_grad_weights(weights->grad.size());

        // Copy data to device
        d_inputs_data.copyToDevice(inputs->data.data(), inputs->data.size());
        d_weights_data.copyToDevice(weights->data.data(), weights->data.size());
        d_grad_output.copyToDevice(grad.data(), grad.size());

        // Set dimensions for backward kernels
        dim3 block_dim(16, 16);
        
        // For weights gradient
        dim3 grid_dim_weights(
            (inputs->cols + block_dim.x - 1) / block_dim.x,
            (cols + block_dim.y - 1) / block_dim.y
        );

        // For inputs gradient
        dim3 grid_dim_inputs(
            (inputs->rows + block_dim.x - 1) / block_dim.x,
            (inputs->cols + block_dim.y - 1) / block_dim.y
        );

        // Launch backward kernels
        matrix_multiply_backward_weights_kernel<<<grid_dim_weights, block_dim>>>(
            d_inputs_data.get(), d_grad_output.get(),
            d_grad_weights.get(), inputs->rows, inputs->cols, cols
        );

        matrix_multiply_backward_inputs_kernel<<<grid_dim_inputs, block_dim>>>(
            d_weights_data.get(), d_grad_output.get(),
            d_grad_inputs.get(), inputs->rows, inputs->cols, cols
        );

        // Copy gradients back to host
        d_grad_inputs.copyToHost(inputs->grad.data(), inputs->grad.size());
        d_grad_weights.copyToHost(weights->grad.data(), weights->grad.size());

        // Continue backprop
        weights->backward();
        inputs->backward();
    }
    if (creation_op == "relu") {
        auto input = parents[0];

        // Allocate device memory
        CUDAMemory d_input_data(input->data.size());
        CUDAMemory d_grad_output(grad.size());
        CUDAMemory d_grad_input(input->grad.size());

        // Copy data to device
        d_input_data.copyToDevice(input->data.data(), input->data.size());
        d_grad_output.copyToDevice(grad.data(), grad.size());

        // Set dimensions for backward kernel
        int blockSize = 256;
        int numBlocks = (input->data.size() + blockSize - 1) / blockSize;

        // Launch backward kernel for ReLU
        relu_backward_kernel<<<numBlocks, blockSize>>>(d_input_data.get(), d_grad_output.get(), d_grad_input.get(), input->data.size());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy gradients back to host
        d_grad_input.copyToHost(input->grad.data(), input->grad.size());
        
        // Continue backprop
        input->backward();
    }
    if (creation_op == "softmax") {
        // When used with cross entropy, the gradient of softmax+cross-entropy 
        // is just (pred - target), so we can pass through the gradient unchanged
        for (size_t i = 0; i < data.size(); i++) {
            parents[0]->grad[i] += grad[i];
        }
        parents[0]->backward();
    }
    if (creation_op == "bce") {
        auto targets = parents[0];
        auto outputs = parents[1];

        // Allocate device memory
        CUDAMemory d_targets(targets->data.size());
        CUDAMemory d_outputs(outputs->data.size());
        CUDAMemory d_grad_output(grad.size());
        CUDAMemory d_grad_inputs(outputs->grad.size());

        // Copy data to device
        d_targets.copyToDevice(targets->data.data(), targets->data.size());
        d_outputs.copyToDevice(outputs->data.data(), outputs->data.size());
        d_grad_output.copyToDevice(grad.data(), grad.size());

        int blockSize = 256;
        int numBlocks = (targets->data.size() + blockSize - 1) / blockSize;

        // Launch backward kernel for BCE
        bce_backward_kernel<<<numBlocks, blockSize>>>(d_targets.get(), d_outputs.get(), d_grad_output.get(), d_grad_inputs.get(), targets->data.size());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy gradients back to host
        d_grad_inputs.copyToHost(outputs->grad.data(), outputs->grad.size());

        // Continue backprop
        // targets->backward();
        outputs->backward();
    }
    if (creation_op == "ce") {
        auto pred = parents[0];
        auto target = parents[1];

        // The gradient should be softmax - target for each sample
        for (size_t i = 0; i < pred->rows; i++) {
            for (size_t j = 0; j < pred->cols; j++) {
                size_t idx = i * pred->cols + j;
                pred->grad[idx] += grad[0] * (pred->data[idx] - target->data[idx]) / pred->rows;
            }
        }
        pred->backward();
    }
    if (creation_op == "softmax") {
            // When used with cross entropy, the gradient of softmax+cross-entropy 
            // is just (pred - target), so we can pass through the gradient unchanged
            for (size_t i = 0; i < data.size(); i++) {
                parents[0]->grad[i] += grad[i];
            }
            parents[0]->backward();
    }
    if (creation_op == "Conv2d") {
        auto input = parents[0];
        auto weights = parents[1];
        auto bias = parents[2];

        size_t batch_size = input->rows;
        size_t in_channels = input->cols / (input->rows * input->cols);
        size_t out_channels = weights->rows;
        
        size_t height = 28;  // MNIST image height
        size_t width = 28;   // MNIST image width
        size_t kernel_size = 3; // From model architecture
        size_t stride = 1;
        size_t padding = 1;
        
        size_t output_height = (height + 2 * padding - kernel_size) / stride + 1;
        size_t output_width = (width + 2 * padding - kernel_size) / stride + 1;
        
        // Allocate device memory
        CUDAMemory d_input(input->data.size());
        CUDAMemory d_weights(weights->data.size());
        CUDAMemory d_grad_output(grad.size());
        CUDAMemory d_grad_input(input->grad.size());
        CUDAMemory d_grad_weights(weights->grad.size());
        CUDAMemory d_grad_bias(bias->grad.size());
        
        // Copy data to device
        d_input.copyToDevice(input->data.data(), input->data.size());
        d_weights.copyToDevice(weights->data.data(), weights->data.size());
        d_grad_output.copyToDevice(grad.data(), grad.size());
        
        // Launch kernel for input gradients
        dim3 gridDim_input(batch_size, in_channels, 1);
        dim3 blockDim_input(height, width, 1);
        
        conv2d_backward_input_kernel<<<gridDim_input, blockDim_input>>>(
            d_weights.get(), d_grad_output.get(),
            d_grad_input.get(), batch_size, in_channels, out_channels,
            height, width, kernel_size, stride, padding,
            output_height, output_width
        );
        
        // Launch kernel for weight gradients
        dim3 gridDim_weights(out_channels, in_channels, 1);
        dim3 blockDim_weights(kernel_size, kernel_size, 1);
        
        conv2d_backward_weights_kernel<<<gridDim_weights, blockDim_weights>>>(
            d_input.get(), d_grad_output.get(),
            d_grad_weights.get(), batch_size, in_channels, out_channels,
            height, width, kernel_size, stride, padding,
            output_height, output_width
        );
        
        // Copy gradients back to host
        d_grad_input.copyToHost(input->grad.data(), input->grad.size());
        d_grad_weights.copyToHost(weights->grad.data(), weights->grad.size());
        
        // For bias, just sum gradients across batch and spatial dimensions
        for (size_t oc = 0; oc < out_channels; oc++) {
            float bias_grad = 0.0f;
            for (size_t b = 0; b < batch_size; b++) {
                for (size_t h = 0; h < output_height; h++) {
                    for (size_t w = 0; w < output_width; w++) {
                        size_t idx = b * (out_channels * output_height * output_width) +
                                    oc * (output_height * output_width) +
                                    h * output_width + w;
                        bias_grad += grad[idx];
                    }
                }
            }
            bias->grad[oc] += bias_grad;
        }
        
        input->backward();
        weights->backward();
        bias->backward();
    }

    if (creation_op == "maxpool2d") {
        auto input = parents[0];
        size_t batch_size = input->rows;
        size_t in_channels = 32;  // From previous Conv2D
        size_t height = static_cast<size_t>(std::sqrt(input->cols / in_channels));
        size_t width = height;
        size_t kernel_size = 2;
        size_t stride = 2;
        size_t new_height = (height - kernel_size) / stride + 1;
        size_t new_width = (width - kernel_size) / stride + 1;

        // Initialize gradients if needed
        if (input->grad.empty()) {
            input->grad.resize(input->data.size(), 0.0f);
        }

        // Allocate device memory
        CUDAMemory d_input(input->data.size());
        CUDAMemory d_grad_output(grad.size());
        CUDAMemory d_grad_input(input->grad.size());

        // Copy data to device
        d_input.copyToDevice(input->data.data(), input->data.size());
        d_grad_output.copyToDevice(grad.data(), grad.size());
        d_grad_input.copyToDevice(input->grad.data(), input->grad.size());

        // Set kernel dimensions
        dim3 gridDim(batch_size, in_channels, 1);
        dim3 blockDim(height, width, 1);

        // Launch kernel
        maxpool2d_backward_kernel<<<gridDim, blockDim>>>(
            d_input.get(),
            d_grad_input.get(),
            d_grad_output.get(),
            batch_size,
            in_channels,
            height,
            width,
            kernel_size,
            stride,
            new_height,
            new_width
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy results back to host
        d_grad_input.copyToHost(input->grad.data(), input->grad.size());

        input->backward();
    }
    if (creation_op == "flatten") {
        for (size_t i = 0; i < grad.size(); i++) {
            parents[0]->grad[i] += grad[i];
        }
        parents[0]->backward();
    }
    if (creation_op == "batchnorm") {
        auto input = parents[0];
        auto gamma = parents[1];
        auto beta = parents[2];
        float eps = 1e-5f;

        size_t batch_size = input->rows;
        size_t num_features = input->cols;

        // Initialize gradients if empty
        if (input->grad.empty()) input->grad.resize(input->data.size(), 0.0f);
        if (gamma->grad.empty()) gamma->grad.resize(gamma->data.size(), 0.0f);
        if (beta->grad.empty()) beta->grad.resize(beta->data.size(), 0.0f);

        // Allocate device memory
        CUDAMemory d_input(input->data.size());
        CUDAMemory d_gamma(gamma->data.size());
        CUDAMemory d_mean(num_features);
        CUDAMemory d_var(num_features);
        CUDAMemory d_grad_output(grad.size());
        CUDAMemory d_grad_input(input->grad.size());
        CUDAMemory d_grad_gamma(gamma->grad.size());
        CUDAMemory d_grad_beta(beta->grad.size());

        // Copy data to device
        d_input.copyToDevice(input->data.data(), input->data.size());
        d_gamma.copyToDevice(gamma->data.data(), gamma->data.size());
        d_grad_output.copyToDevice(grad.data(), grad.size());

        // Compute mean and variance
        int block_size = 256;
        int num_blocks = (num_features + block_size - 1) / block_size;

        batchnorm_mean_kernel<<<num_blocks, block_size>>>(
            d_input.get(),
            d_mean.get(),
            batch_size,
            num_features
        );
        CUDA_CHECK(cudaGetLastError());

        batchnorm_var_kernel<<<num_blocks, block_size>>>(
            d_input.get(),
            d_mean.get(),
            d_var.get(),
            batch_size,
            num_features
        );
        CUDA_CHECK(cudaGetLastError());

        // Compute gradients
        batchnorm_backward_kernel<<<num_blocks, block_size>>>(
            d_input.get(),
            d_grad_output.get(),
            d_gamma.get(),
            d_mean.get(),
            d_var.get(),
            d_grad_input.get(),
            d_grad_gamma.get(),
            d_grad_beta.get(),
            batch_size,
            num_features,
            eps
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy results back to host
        d_grad_input.copyToHost(input->grad.data(), input->grad.size());
        d_grad_gamma.copyToHost(gamma->grad.data(), gamma->grad.size());
        d_grad_beta.copyToHost(beta->grad.data(), beta->grad.size());

        // Continue backpropagation
        input->backward();
        gamma->backward();
        beta->backward();
    }

}