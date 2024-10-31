// tensor_cuda.cu
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <iostream>
#include <random>
#include <stdexcept>

// Forward declaration of Tensor class to match tensor.cpp
class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    size_t rows;
    size_t cols;
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<std::shared_ptr<Tensor>> parents;
    bool requires_grad;
    std::string creation_op;

    Tensor(size_t rows, size_t cols, bool requires_grad = false, std::string creation_op = "");
    Tensor(const std::vector<float>& data, size_t rows, size_t cols, bool requires_grad = false, std::string creation_op = "");
    static std::shared_ptr<Tensor> multiply(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    void backward();
    void debug_gradient_flow();
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

__global__ void conv2d_forward_kernel(
    const float* input, const float* weights, const float* bias,
    float* output,
    size_t batch_size, size_t in_channels, size_t out_channels,
    size_t height, size_t width, size_t kernel_size,
    size_t stride, size_t padding,
    size_t output_height, size_t output_width) {
    
    // Calculate the index of the output element
    size_t b = blockIdx.x; // Batch index
    size_t oc = blockIdx.y; // Output channel index
    size_t oh = threadIdx.x / output_width; // Output height index
    size_t ow = threadIdx.x % output_width; // Output width index

    // Ensure we're within output dimensions
    if (oh < output_height && ow < output_width) {
        float sum = 0.0f;

        // Perform convolution
        for (size_t ic = 0; ic < in_channels; ic++) {
            for (size_t kh = 0; kh < kernel_size; kh++) {
                for (size_t kw = 0; kw < kernel_size; kw++) {
                    // Calculate input coordinates
                    int ih = static_cast<int>(oh * stride + kh) - static_cast<int>(padding);
                    int iw = static_cast<int>(ow * stride + kw) - static_cast<int>(padding);

                    // Check bounds
                    if (ih >= 0 && ih < static_cast<int>(height) && 
                        iw >= 0 && iw < static_cast<int>(width)) {
                        
                        // Calculate indices
                        size_t input_idx = b * (in_channels * height * width) + 
                                           ic * (height * width) + 
                                           ih * width + iw;
                        size_t weight_idx = (ic * kernel_size * kernel_size + 
                                             kh * kernel_size + kw) * out_channels + oc;

                        sum += input[input_idx] * weights[weight_idx];
                    }
                }
            }
        }

        // Add bias
        sum += bias[oc];

        // Write to output
        size_t output_idx = b * (out_channels * output_height * output_width) +
                            oc * (output_height * output_width) +
                            oh * output_width + ow;
        output[output_idx] = sum;
    }
}

// std::shared_ptr<Tensor> Conv2D::forward(std::shared_ptr<Tensor> input) {
//     size_t batch_size = input->rows;
//     size_t in_channels = input->cols / (input->rows * input->cols);
//     size_t out_channels = weights->rows;
    
//     size_t height = 28;  // MNIST image height
//     size_t width = 28;   // MNIST image width
//     size_t kernel_size = 3; // From model architecture
//     size_t stride = 1;
//     size_t padding = 1;
    
//     size_t output_height = (height + 2 * padding - kernel_size) / stride + 1;
//     size_t output_width = (width + 2 * padding - kernel_size) / stride + 1;

//     std::vector<float> output_data(batch_size * out_channels * output_height * output_width, 0.0f);
//     auto output = std::make_shared<Tensor>(output_data, batch_size, 
//                                            out_channels * output_height * output_width,
//                                            true, "Conv2d");

//     // Iterate over the batch, output channels, and output spatial dimensions
//     for (size_t b = 0; b < batch_size; b++) {
//         for (size_t oc = 0; oc < out_channels; oc++) {
//             for (size_t oh = 0; oh < output_height; oh++) {
//                 for (size_t ow = 0; ow < output_width; ow++) {
//                     float sum = 0.0f;

//                     // Perform convolution
//                     for (size_t ic = 0; ic < in_channels; ic++) {
//                         for (size_t kh = 0; kh < kernel_size; kh++) {
//                             for (size_t kw = 0; kw < kernel_size; kw++) {
//                                 int ih = static_cast<int>(oh * stride + kh) - static_cast<int>(padding);
//                                 int iw = static_cast<int>(ow * stride + kw) - static_cast<int>(padding);

//                                 // Check if the input coordinates are within the input tensor bounds
//                                 if (ih >= 0 && ih < static_cast<int>(height) && 
//                                     iw >= 0 && iw < static_cast<int>(width)) {
//                                     size_t input_idx = b * (in_channels * height * width) + 
//                                                      ic * (height * width) + 
//                                                      ih * width + iw;
//                                     size_t weight_idx = (ic * kernel_size * kernel_size + 
//                                                          kh * kernel_size + kw) * out_channels + oc;
//                                     sum += input->data[input_idx] * weights->data[weight_idx];
//                                 }
//                             }
//                         }
//                     }

//                     // Add the bias
//                     sum += bias->data[oc];

//                     // Store the result in the output tensor
//                     size_t output_idx = b * (out_channels * output_height * output_width) +
//                                         oc * (output_height * output_width) +
//                                         oh * output_width + ow;
//                     output->data[output_idx] = sum;
//                 }
//             }
//         }
//     }

//     return output;
// }

__global__ void conv2d_backward_kernel(
    const float* d_output,          // Gradient of the output (next layer)
    const float* input,             // Input to the Conv2D layer
    float* d_weights,               // Gradient of the weights
    float* d_bias,                  // Gradient of the bias
    float* d_input,                 // Gradient of the input
    size_t batch_size,              // Batch size
    size_t in_channels,             // Number of input channels
    size_t out_channels,            // Number of output channels
    size_t height,                  // Input height
    size_t width,                   // Input width
    size_t kernel_size,             // Kernel size (assuming square kernels)
    size_t stride,                  // Stride
    size_t padding)                 // Padding
{
    // Calculate output dimensions
    size_t output_height = (height + 2 * padding - kernel_size) / stride + 1;
    size_t output_width = (width + 2 * padding - kernel_size) / stride + 1;

    // Calculate global thread indices
    size_t b = blockIdx.x; // Batch index
    size_t oc = blockIdx.y; // Output channel index
    size_t oh = blockIdx.z; // Output height index
    size_t ow = threadIdx.x; // Output width index

    if (b < batch_size && oc < out_channels && oh < output_height && ow < output_width) {
        float grad_out = d_output[b * (out_channels * output_height * output_width) +
                                   oc * (output_height * output_width) +
                                   oh * output_width + ow];

        // Compute gradient for bias
        atomicAdd(&d_bias[oc], grad_out);

        // Compute gradients for weights and input
        for (size_t ic = 0; ic < in_channels; ic++) {
            for (size_t kh = 0; kh < kernel_size; kh++) {
                for (size_t kw = 0; kw < kernel_size; kw++) {
                    // Calculate the input location based on padding and stride
                    int ih = oh * stride + kh - padding;
                    int iw = ow * stride + kw - padding;

                    if (ih >= 0 && ih < static_cast<int>(height) &&
                        iw >= 0 && iw < static_cast<int>(width)) {
                        // Compute weight gradients
                        atomicAdd(&d_weights[(ic * kernel_size * kernel_size + 
                                               kh * kernel_size + kw) * out_channels + oc], 
                                   input[b * (in_channels * height * width) + 
                                         ic * (height * width) + 
                                         ih * width + iw] * grad_out);

                        // Compute input gradients
                        atomicAdd(&d_input[b * (in_channels * height * width) + 
                                            ic * (height * width) + 
                                            ih * width + iw], 
                                   d_weights[(ic * kernel_size * kernel_size + 
                                             kh * kernel_size + kw) * out_channels + oc] * grad_out);
                    }
                }
            }
        }
    }
}

// Extension of Tensor::backward() for CUDA multiplication
void Tensor::backward() {
    if (!requires_grad) return;

    if (grad.empty()) {
        grad = std::vector<float>(rows * cols, 1.0f);
    }

    if (creation_op == "add") {
        std::cout << "Add backward" << std::endl;

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
        std::cout << "Multiply backward" << std::endl;

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
        std::cout << "Relu backward" << std::endl;

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
        std::cout << "BCE backward" << std::endl;

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
        // Convolutional backward implementation
        auto input = parents[0];
        auto weights = parents[1];
        auto bias = parents[2];

        // Retrieve necessary dimensions
        size_t batch_size = input->rows;
        size_t in_channels = input->cols / (input->rows * input->cols);
        size_t out_channels = weights->rows;
        size_t height = static_cast<size_t>(std::sqrt(input->cols / in_channels));
        size_t width = height;
        size_t stride = 1;
        size_t padding = 1;
        size_t kernel_size = weights->cols / in_channels; // Assuming square kernel
        size_t output_height = (height + 2 * padding - kernel_size) / stride + 1;
        size_t output_width = (width + 2 * padding - kernel_size) / stride + 1;

        // Allocate device memory for gradients
        float* d_weights, *d_bias, *d_input, *d_grad_output;
        cudaMalloc(&d_weights, weights->data.size() * sizeof(float));
        cudaMalloc(&d_bias, bias->data.size() * sizeof(float));
        cudaMalloc(&d_input, input->data.size() * sizeof(float));
        cudaMalloc(&d_grad_output, grad.size() * sizeof(float));

        // Copy data to device
        cudaMemcpy(d_weights, weights->data.data(), weights->data.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias, bias->data.data(), bias->data.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_input, input->data.data(), input->data.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_grad_output, grad.data(), grad.size() * sizeof(float), cudaMemcpyHostToDevice);

        // Launch the backward kernel
        dim3 block_size(output_width); // Each thread computes one output pixel
        dim3 grid_size(batch_size, out_channels, output_height); // Each block computes one output channel for a batch
        
        conv2d_backward_kernel<<<grid_size, block_size>>>(
            d_grad_output,         // Gradient of the output (next layer)
            d_input,               // Input to the Conv2D layer
            d_weights,             // Gradient of the weights
            d_bias,                // Gradient of the bias
            d_input,               // Gradient of the input
            batch_size,
            in_channels,
            out_channels,
            height,
            width,
            kernel_size,
            stride,
            padding
        );

        // Copy the gradients back to the Tensor objects
        std::vector<float> weights_grad(weights->data.size(), 0.0f);
        std::vector<float> bias_grad(bias->data.size(), 0.0f);
        std::vector<float> input_grad(input->data.size(), 0.0f);

        cudaMemcpy(weights_grad.data(), d_weights, weights->data.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(bias_grad.data(), d_bias, bias->data.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(input_grad.data(), d_input, input->data.size() * sizeof(float), cudaMemcpyDeviceToHost);

        // Update the gradients in the Tensor objects
        weights->grad = weights_grad;
        bias->grad = bias_grad;
        input->grad = input_grad;

        // Free device memory
        cudaFree(d_weights);
        cudaFree(d_bias);
        cudaFree(d_input);
        cudaFree(d_grad_output);

        // Return the gradient of the input tensor for further backpropagation
        input->backward();
        bias->backward();
    }

    if (creation_op == "maxpool2d") {
        auto input = parents[0];
        size_t batch_size = input->rows;
        size_t in_channels = input->cols / (input->rows * input->cols);
        size_t height = input->rows;
        size_t width = input->cols;
        size_t new_height = (height - 2) / 2 + 1;
        size_t new_width = (width - 2) / 2 + 1;
        
        #pragma omp parallel for collapse(4)
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t ic = 0; ic < in_channels; ic++) {
                for (size_t h = 0; h < height; h++) {
                    for (size_t w = 0; w < width; w++) {
                        float max_val = -1e9;
                        size_t max_h = 0;
                        size_t max_w = 0;
                        for (size_t kh = 0; kh < 2; kh++) {
                            for (size_t kw = 0; kw < 2; kw++) {
                                size_t nh = h + kh;
                                size_t nw = w + kw;
                                if (nh < height && nw < width) {
                                    float val = input->data[b * height * width + ic * height * width + nh * width + nw];
                                    if (val > max_val) {
                                        max_val = val;
                                        max_h = nh;
                                        max_w = nw;
                                    }
                                }
                            }
                        }
                        input->grad[b * height * width + ic * height * width + max_h * width + max_w] += grad[b * height * width + ic * height * width + h * width + w];
                    }
                }
            }
        }
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
        
        if (input->grad.empty()) input->grad.resize(input->data.size(), 0.0f);
        if (gamma->grad.empty()) gamma->grad.resize(gamma->data.size(), 0.0f);
        if (beta->grad.empty()) beta->grad.resize(beta->data.size(), 0.0f);

        size_t batch_size = input->rows;
        size_t num_features = gamma->cols;
        
        // Compute batch mean
        std::vector<float> batch_mean(num_features, 0.0f);
        #pragma omp parallel for collapse(1)
        for (size_t j = 0; j < num_features; j++) {
            for (size_t i = 0; i < batch_size; i++) {
                batch_mean[j] += input->data[i * num_features + j];
            }
            batch_mean[j] /= batch_size;
        }

        // Compute batch variance
        std::vector<float> batch_var(num_features, 0.0f);
        #pragma omp parallel for collapse(1)
        for (size_t j = 0; j < num_features; j++) {
            for (size_t i = 0; i < batch_size; i++) {
                float diff = input->data[i * num_features + j] - batch_mean[j];
                batch_var[j] += diff * diff;
            }
            batch_var[j] = batch_var[j] / batch_size + eps;
        }

        // Compute gradients for gamma and 
        for (size_t j = 0; j < num_features; j++) {
            float dgamma = 0.0f;
            float dbeta = 0.0f;
            for (size_t i = 0; i < batch_size; i++) {
                size_t idx = i * num_features + j;
                float x_normalized = (input->data[idx] - batch_mean[j]) / std::sqrt(batch_var[j]);
                dgamma += grad[idx] * x_normalized;
                dbeta += grad[idx];
            }
            gamma->grad[j] += dgamma;
            beta->grad[j] += dbeta;
        }

        // Compute gradients for input
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < batch_size; i++) {
            for (size_t j = 0; j < num_features; j++) {
                size_t idx = i * num_features + j;
                float x_centered = input->data[idx] - batch_mean[j];
                float std_inv = 1.0f / std::sqrt(batch_var[j]);
                float dx_normalized = grad[idx] * gamma->data[j];
                input->grad[idx] += dx_normalized * std_inv;
            }
        }

        input->backward();
        gamma->backward();
        beta->backward();
    }

}