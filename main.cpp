#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <random>

class Tensor : public std::enable_shared_from_this<Tensor> {
private:
    static std::mt19937& get_generator() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        return gen;
    }

    static std::vector<float> generate_random_data(size_t rows, size_t cols) {
        // Xavier/Glorot initialization with reduced scale
        float scale = std::sqrt(2.0f / (static_cast<float>(rows + cols)));
        std::normal_distribution<float> dist(0.0f, scale);
        std::vector<float> data(rows * cols);
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = dist(get_generator());
        }
        return data;
    }

public:
    size_t rows;
    size_t cols;
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<std::shared_ptr<Tensor>> parents;
    bool requires_grad;
    std::string creation_op;

    Tensor(size_t rows, size_t cols, bool requires_grad = false, std::string creation_op = "")
        : rows(rows), cols(cols), requires_grad(requires_grad), creation_op(creation_op) {
        data = generate_random_data(rows, cols);
        grad = std::vector<float>(rows * cols, 0);
    }

    Tensor(const std::vector<float>& data, size_t rows, size_t cols, bool requires_grad = false, std::string creation_op = "")
        : data(data), rows(rows), cols(cols), requires_grad(requires_grad), creation_op(creation_op) {
        grad = std::vector<float>(rows * cols, 0);
    }

    ~Tensor() {
        parents.clear();

    }
    
    void clear_graph() {
        parents.clear();
        grad.clear();
    }

    void backward() {
        if (!requires_grad) return;

        if (grad.empty()) {
            grad = std::vector<float>(rows * cols, 1.0f);
        }

        
        if (creation_op == "add") {
            for (const auto& parent : parents) {
                // Initialize parent gradients if needed
                if (parent->grad.empty()) {
                    parent->grad = std::vector<float>(parent->rows * parent->cols, 0.0f);
                }

                if (parent->rows == rows && parent->cols == cols) {
                    // Same shape case
                    for (size_t i = 0; i < grad.size(); i++) {
                        parent->grad[i] += grad[i];
                    }
                } else if (parent->rows == 1) {
                    // Broadcasting case: parent is (1, cols)
                    if (parent->cols == cols) {
                        // Standard broadcasting
                        for (size_t i = 0; i < rows; i++) {
                            for (size_t j = 0; j < cols; j++) {
                                parent->grad[j] += grad[i * cols + j];
                            }
                        }
                    } else if (parent->cols * rows == cols) {
                        // Handle repeated broadcasting
                        for (size_t i = 0; i < rows; i++) {
                            for (size_t j = 0; j < parent->cols; j++) {
                                parent->grad[j] += grad[i * cols + j];
                            }
                        }
                    }
                }
                parent->backward();
            }
        }
        if (creation_op == "mul") {
            auto inputs = parents[0];
            auto weights = parents[1];
            
            // For weights: grad = input.T @ upstream_grad
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < inputs->cols; i++) {  // input features
                for (size_t j = 0; j < cols; j++) {  // output features
                    float weight_grad = 0.0f;
                    for (size_t b = 0; b < inputs->rows; b++) {  // batch dimension
                        weight_grad += inputs->data[b * inputs->cols + i] * grad[b * cols + j];
                    }
                    weights->grad[i * weights->cols + j] += weight_grad;
                }
            }
            
            // For inputs: grad = upstream_grad @ weight
            #pragma omp parallel for collapse(2)
            for (size_t b = 0; b < inputs->rows; b++) {  // batch dimension
                for (size_t i = 0; i < inputs->cols; i++) {  // input features
                    float input_grad = 0.0f;
                    for (size_t j = 0; j < cols; j++) {  // output features
                        input_grad += grad[b * cols + j] * weights->data[i * weights->cols + j];
                    }
                    inputs->grad[b * inputs->cols + i] += input_grad;
                }
            }
            
            weights->backward();
            inputs->backward();
        }
        if (creation_op == "sigmoid") {
            for (size_t i = 0; i < data.size(); i++) {
                if (i < grad.size()) {
                    float sigmoid_val = data[i];
                    parents[0]->grad[i] += grad[i] * sigmoid_val * (1.0f - sigmoid_val);
                }
            }
            parents[0]->backward();
        }
        if (creation_op == "relu") {
            for (size_t i = 0; i < data.size(); i++) {
                parents[0]->grad[i] += grad[i] * (data[i] > 0 ? 1.0f : 0.0f);
            }
            parents[0]->backward();
        }
        if (creation_op == "tanh") {
            for (size_t i = 0; i < data.size(); i++) {
                float tanh_val = data[i];
                parents[0]->grad[i] += grad[i] * (1.0f - tanh_val * tanh_val);
            }
            parents[0]->backward();
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
            auto pred = parents[0];
            auto target = parents[1];
            
            for (size_t i = 0; i < pred->data.size(); i++) {
                float p = std::max(std::min(pred->data[i], 1.0f - 1e-15f), 1e-15f);
                float t = target->data[i];
                pred->grad[i] += grad[i] * (p - t);
            }
            pred->backward();
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
        if (creation_op == "transpose") {
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    parents[0]->grad[j * rows + i] += grad[i * cols + j];
                }
            }
            parents[0]->backward();
        }
        if (creation_op == "sum") {
            for (size_t i = 0; i < parents[0]->data.size(); i++) {
                parents[0]->grad[i] += grad[0];  // Broadcast the gradient
            }
            parents[0]->backward();
        }
        if (creation_op == "expand") {
            // Sum up gradients for each copy
            for (size_t i = 0; i < parents[0]->data.size(); i++) {
                for (size_t j = 0; j < cols/parents[0]->cols; j++) {
                    parents[0]->grad[i] += grad[i * (cols/parents[0]->cols) + j];
                }
            }
            parents[0]->backward();
        }
        if (creation_op == "reshape") {
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    parents[0]->grad[i * cols + j] += grad[i * cols + j];
                }
            }
            parents[0]->backward();
        }
        if (creation_op == "Conv2d") {
            auto input = parents[0];
            auto weights = parents[1];
            auto bias = parents[2];

            if (input->grad.empty()) {
                input->grad.resize(input->data.size(), 0.0f);
            }
            if (weights->grad.empty()) {
                weights->grad.resize(weights->data.size(), 0.0f);
            }
            if (bias->grad.empty()) {
                bias->grad.resize(bias->data.size(), 0.0f);
            }

            size_t batch_size = input->rows;
            size_t in_channels = input->cols / (input->rows * input->cols);
            size_t out_channels = weights->rows;
            
            size_t height = 28;  // MNIST image height
            size_t width = 28;   // MNIST image width
            size_t kernel_size = 3; // From model architecture
            size_t stride = 1;
            size_t padding = 1;
            
            size_t new_height = (height + 2 * padding - kernel_size) / stride + 1;
            size_t new_width = (width + 2 * padding - kernel_size) / stride + 1;
            
            // Compute gradients for weights
            #pragma omp parallel for collapse(3)
            for (size_t oc = 0; oc < out_channels; oc++) {
                for (size_t ic = 0; ic < in_channels; ic++) {
                    for (size_t kh = 0; kh < weights->cols; kh++) {
                        for (size_t kw = 0; kw < weights->cols; kw++) {
                            float weight_grad = 0.0f;
                            for (size_t b = 0; b < batch_size; b++) {
                                for (size_t nh = 0; nh < new_height; nh++) {
                                    for (size_t nw = 0; nw < new_width; nw++) {
                                        size_t h = nh * stride + kh - padding;
                                        size_t w = nw * stride + kw - padding;
                                        if (h >= 0 && h < height && w >= 0 && w < width) {
                                            weight_grad += input->data[b * height * width + ic * height * width + h * width + w] *
                                                          grad[b * out_channels * new_height * new_width + oc * new_height * new_width + nh * new_width + nw];
                                        }
                                    }
                                }
                            }
                            weights->grad[oc * in_channels * weights->cols * weights->cols + ic * weights->cols * weights->cols + kh * weights->cols + kw] += weight_grad;
                        }
                    }
                }
            }
            
            // Compute gradients for bias
            for (size_t oc = 0; oc < out_channels; oc++) {
                float bias_grad = 0.0f;
                for (size_t b = 0; b < batch_size; b++) {
                    for (size_t h = 0; h < new_height; h++) {
                        for (size_t w = 0; w < new_width; w++) {
                            size_t idx = b * (out_channels * new_height * new_width) +
                                    oc * new_height * new_width +
                                    h * new_width + w;
                            if (idx < grad.size()) {
                                bias_grad += grad[idx];
                            }
                        }
                    }
                }
                bias->grad[oc] += bias_grad;
            }

            // Compute gradients for input
            #pragma omp parallel for collapse(4)
            for (size_t b = 0; b < batch_size; b++) {
                for (size_t ic = 0; ic < in_channels; ic++) {
                    for (size_t h = 0; h < height; h++) {
                        for (size_t w = 0; w < width; w++) {
                            float input_grad = 0.0f;
                            for (size_t oc = 0; oc < out_channels; oc++) {
                                for (size_t kh = 0; kh < weights->cols; kh++) {
                                    for (size_t kw = 0; kw < weights->cols; kw++) {
                                        size_t nh = (h + padding - kh) / stride;
                                        size_t nw = (w + padding - kw) / stride;
                                        if (nh * stride + kh - padding == h && nw * stride + kw - padding == w) {
                                            input_grad += weights->data[oc * in_channels * weights->cols * weights->cols + ic * weights->cols * weights->cols + kh * weights->cols + kw] *
                                                          grad[b * out_channels * new_height * new_width + oc * new_height * new_width + nh * new_width + nw];
                                        }
                                    }
                                }
                            }
                            input->grad[b * height * width + ic * height * width + h * width + w] += input_grad;
                        }
                    }
                }
            }
            
            weights->backward();
            bias->backward();
            input->backward();
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
    };

    // Add debug printing function to help track gradient flow
    void debug_gradient_flow() {
        std::cout << "Operation: " << creation_op << std::endl;
        std::cout << "Has gradient: " << (grad.size() > 0) << std::endl;
        if (!grad.empty()) {
            std::cout << "Gradient values: ";
            for (float g : grad) {
                std::cout << g << " ";
            }
            std::cout << std::endl;
        }
    }

    static std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
        // Case 1: Same shapes
        if (a->rows == b->rows && a->cols == b->cols) {
            std::vector<float> new_data(a->data.size());
            for (size_t i = 0; i < a->data.size(); i++) {
                new_data[i] = a->data[i] + b->data[i];
            }
            auto result = std::make_shared<Tensor>(new_data, a->rows, a->cols, 
                                                a->requires_grad || b->requires_grad, "add");
            result->parents.push_back(a);
            result->parents.push_back(b);
            return result;
        }
        
        // Case 2: Broadcasting b to match a's batch size
        if (b->rows == 1 && (b->cols == a->cols || b->cols * a->rows == a->cols)) {
            std::vector<float> new_data(a->rows * a->cols);
            if (b->cols == a->cols) {
                // Broadcasting (1, N) to (M, N)
                for (size_t i = 0; i < a->rows; i++) {
                    for (size_t j = 0; j < a->cols; j++) {
                        new_data[i * a->cols + j] = a->data[i * a->cols + j] + b->data[j];
                    }
                }
            } else if (b->cols * a->rows == a->cols) {
                // Handle the case where b needs to be repeated for each row
                for (size_t i = 0; i < a->rows; i++) {
                    for (size_t j = 0; j < b->cols; j++) {
                        new_data[i * a->cols + j] = a->data[i * a->cols + j] + b->data[j];
                    }
                }
            }
            auto result = std::make_shared<Tensor>(new_data, a->rows, a->cols, 
                                                a->requires_grad || b->requires_grad, "add");
            result->parents.push_back(a);
            result->parents.push_back(b);
            return result;
        }
        
        // Case 3: Broadcasting a to match b's batch size
        if (a->rows == 1 && (a->cols == b->cols || a->cols * b->rows == b->cols)) {
            return add(b, a);  // Reuse case 2 by swapping arguments
        }

        // If we get here, shapes are incompatible
        throw std::runtime_error("Tensor shapes " + 
                            std::to_string(a->rows) + "x" + std::to_string(a->cols) + 
                            " and " + 
                            std::to_string(b->rows) + "x" + std::to_string(b->cols) + 
                            " are not compatible for addition");
    }

    static std::shared_ptr<Tensor> multiply(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
        if (a->cols != b->rows) {
            throw std::runtime_error("Matrix dimensions " + 
                                std::to_string(a->rows) + "x" + std::to_string(a->cols) + 
                                " and " + 
                                std::to_string(b->rows) + "x" + std::to_string(b->cols) + 
                                " are not compatible for multiplication");
        }

        std::vector<float> new_data(a->rows * b->cols);
        for (size_t i = 0; i < a->rows; i++) {
            for (size_t j = 0; j < b->cols; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < a->cols; k++) {
                    sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
                }
                new_data[i * b->cols + j] = sum;
            }
        }

        auto result = std::make_shared<Tensor>(new_data, a->rows, b->cols, 
                                            a->requires_grad || b->requires_grad, "mul");
        result->parents.push_back(a);
        result->parents.push_back(b);
        return result;
    }

    std::shared_ptr<Tensor> reshape(size_t new_rows, size_t new_cols) {
        if (new_rows * new_cols != rows * cols) {
            throw std::runtime_error("Cannot reshape tensor of size " + 
                                std::to_string(rows) + "x" + std::to_string(cols) + 
                                " to " + 
                                std::to_string(new_rows) + "x" + std::to_string(new_cols));
        }
        auto result = std::make_shared<Tensor>(data, new_rows, new_cols, requires_grad, "reshape");
        result->parents.push_back(shared_from_this());
        return result;
    }

    std::shared_ptr<Tensor> sum() {
        float sum = 0.0f;
        for (size_t i = 0; i < data.size(); i++) {
            sum += data[i];
        }
        auto result = std::make_shared<Tensor>(std::vector<float>{sum}, 1, 1, requires_grad, "sum");
        result->parents.push_back(shared_from_this());
        return result;
    }

    std::shared_ptr<Tensor> expand(size_t dim, size_t copies) {
        if (dim > 1) {
            throw std::runtime_error("Not implemented");
        }
        std::vector<float> new_data(data.size() * copies);
        for (size_t i = 0; i < data.size(); i++) {
            for (size_t j = 0; j < copies; j++) {
                new_data[i * copies + j] = data[i];
            }
        }
        auto result = std::make_shared<Tensor>(new_data, rows, cols * copies, requires_grad, "expand");
        result->parents.push_back(shared_from_this());
        return result;
    }

    std::shared_ptr<Tensor> transpose() {
        std::vector<float> new_data(data.size());
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                new_data[j * rows + i] = data[i * cols + j];
            }
        }
        auto result = std::make_shared<Tensor>(new_data, cols, rows, requires_grad, "transpose");
        result->parents.push_back(shared_from_this());
        return result;
    }

    std::shared_ptr<Tensor> relu() {
        std::vector<float> new_data(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            new_data[i] = std::max(0.0f, data[i]);
        }
        auto result = std::make_shared<Tensor>(new_data, rows, cols, requires_grad, "relu");
        result->parents.push_back(shared_from_this());
        return result;
    }

    std::shared_ptr<Tensor> Tanh() {
        std::vector<float> new_data(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            new_data[i] = std::tanh(data[i]);
        }
        auto result = std::make_shared<Tensor>(new_data, rows, cols, requires_grad, "tanh");
        result->parents.push_back(shared_from_this());
        return result;
    }

    std::shared_ptr<Tensor> sigmoid() {
        std::vector<float> new_data(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            new_data[i] = 1.0f / (1.0f + std::exp(-data[i]));
        }
        auto result = std::make_shared<Tensor>(new_data, rows, cols, requires_grad, "sigmoid");
        result->parents.push_back(shared_from_this());
        return result;
    }

    std::shared_ptr<Tensor> softmax() {
        std::vector<float> new_data(data.size());
        for (size_t i = 0; i < rows; i++) {
            // Find max value in this row for numerical stability
            float max_val = data[i * cols];
            for (size_t j = 1; j < cols; j++) {
                max_val = std::max(max_val, data[i * cols + j]);
            }
            
            // Compute exp(x - max) and sum
            float sum = 0.0f;
            for (size_t j = 0; j < cols; j++) {
                float exp_val = std::exp(data[i * cols + j] - max_val);
                new_data[i * cols + j] = exp_val;
                sum += exp_val;
            }
            
            // Normalize
            for (size_t j = 0; j < cols; j++) {
                new_data[i * cols + j] /= sum;
            }
        }
        auto result = std::make_shared<Tensor>(new_data, rows, cols, requires_grad, "softmax");
        result->parents.push_back(shared_from_this());
        return result;
    }

    static std::shared_ptr<Tensor> binary_cross_entropy(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target) {
        std::vector<float> new_data(pred->data.size());
        float scale_factor = 100.0f;  // Scale factor to prevent tiny gradients
        for (size_t i = 0; i < pred->data.size(); i++) {
            float p = std::max(std::min(pred->data[i], 1.0f - 1e-7f), 1e-7f);  // Changed epsilon
            new_data[i] = (-target->data[i] * std::log(p) - (1 - target->data[i]) * std::log(1 - p)) * scale_factor;
        }
        auto result = std::make_shared<Tensor>(new_data, pred->rows, pred->cols, pred->requires_grad, "bce");
        result->parents.push_back(pred);
        result->parents.push_back(target);
        return result;
    }

    static std::shared_ptr<Tensor> cross_entropy(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target) {
        if (pred->rows != target->rows || pred->cols != target->cols) {
            throw std::runtime_error("Tensor shapes must match for cross entropy");
        }
        float total_loss = 0.0f;
        for (size_t i = 0; i < pred->rows; i++) {
            float sample_loss = 0.0f;
            for (size_t j = 0; j < pred->cols; j++) {
                size_t idx = i * pred->cols + j;
                if (target->data[idx] > 0) {  // Only compute loss for the target class
                    // Clip prediction to prevent log(0)
                    float p = std::max(std::min(pred->data[idx], 1.0f - 1e-7f), 1e-7f);
                    sample_loss -= std::log(p);
                }
            }
            total_loss += sample_loss;
        }
        // Average over batch
        total_loss /= pred->rows;
        auto result = std::make_shared<Tensor>(std::vector<float>{total_loss}, 1, 1, pred->requires_grad, "ce");
        result->parents.push_back(pred);
        result->parents.push_back(target);
        return result;
    }

    std::vector<size_t> shape() {
        return {rows, cols};
    }

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        os << "tensor(";
        os << "[";
        for (size_t i = 0; i < tensor.rows; ++i) {
            os << "[";
            for (size_t j = 0; j < tensor.cols; ++j) {
                if (j > 0) os << ", ";
                os << tensor.data[i * tensor.cols + j];
            }
            os << "]";
            if (i < tensor.rows - 1) os << ",\n";
        }
        os << "]";
        if (tensor.requires_grad) {
            os << ", requires_grad=True";
        }
        os << ")";
        return os;
    }


    void print_grad() {
        // Print the gradients nicely in the tensor format.
        std::cout << "tensor(";
        std::cout << "[";
        for (size_t i = 0; i < rows; ++i) {
            std::cout << "[";
            for (size_t j = 0; j < cols; ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << grad[i * cols + j];
            }
            std::cout << "]";
            if (i < rows - 1) std::cout << ",\n";
        }
        std::cout << "]";
        std::cout << ")" << std::endl;
    }
};


class Layer {
public:
    std::shared_ptr<Tensor> parameters = {};

    Layer() {}

    Layer(std::shared_ptr<Tensor> parameters): parameters(parameters) {}

    std::shared_ptr<Tensor> get_parameters() {
        return parameters;
    }

    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) {
        throw std::runtime_error("Not implemented");
    }
};

class ReLU : public Layer {
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) {
        return input->relu();
    }
};

class Tanh : public Layer {
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) {
        return input->Tanh();
    }
};

class Sigmoid : public Layer {
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) {
        return input->sigmoid();
    }
};

class Softmax : public Layer {
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) {
        return input->softmax();
    }
}; 

class SGD {
public:
    std::vector<std::shared_ptr<Tensor>> parameters;
    float learning_rate;

    SGD(std::vector<std::shared_ptr<Tensor>> parameters, float learning_rate)
        : parameters(parameters), learning_rate(learning_rate) {}

    void zero_grad() {
        for (auto& param : parameters) {
            if (param->requires_grad) {
                std::fill(param->grad.begin(), param->grad.end(), 0.0f);
            }
        }
    }

    void step() {
        for (auto& param : parameters) {
            if (param->requires_grad) {
                for (size_t i = 0; i < param->data.size(); ++i) {
                    param->data[i] -= learning_rate * param->grad[i];
                }
            }
        }
    }
};

class Linear : public Layer {
public:
    std::shared_ptr<Tensor> weights;
    std::shared_ptr<Tensor> bias;
    bool use_bias;

    Linear(size_t input_size, size_t output_size, bool use_bias = true)
        : use_bias(use_bias) {
        // weights shape should be (input_size, output_size)
        weights = std::make_shared<Tensor>(input_size, output_size, true, "linear"); 
        if (use_bias) {
            // bias shape should be (1, output_size)
            bias = std::make_shared<Tensor>(1, output_size, true, "bias");
        }
    }

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override {
        // input shape: (batch_size, input_size)
        // weights shape: (input_size, output_size)
        // output shape will be: (batch_size, output_size)
        auto output = Tensor::multiply(input, weights);
        
        if (use_bias) {
            // bias shape: (1, output_size)
            // output shape: (batch_size, output_size)
            output = Tensor::add(output, bias);
        }
        return output;
    }

    std::vector<std::shared_ptr<Tensor>> get_parameters() {
        std::vector<std::shared_ptr<Tensor>> params;
        params.push_back(weights);
        if (use_bias) {
            params.push_back(bias);
        }
        return params;
    }
};

class Conv2D : public Layer {
public:
    std::shared_ptr<Tensor> weights;
    std::shared_ptr<Tensor> bias;
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;

    Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride = 1, size_t padding = 0)
        : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), 
          stride(stride), padding(padding) {
        // Initialize weights with correct shape
        weights = std::make_shared<Tensor>(in_channels * kernel_size * kernel_size, 
                                         out_channels, true, "conv_weights");
        bias = std::make_shared<Tensor>(1, out_channels, true, "conv_bias");
    }

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override {
        // Reshape input to extract spatial dimensions
        size_t batch_size = input->rows;
        size_t height = static_cast<size_t>(std::sqrt(input->cols / in_channels));
        size_t width = height;
        
        // Calculate output dimensions
        size_t output_height = (height + 2 * padding - kernel_size) / stride + 1;
        size_t output_width = (width + 2 * padding - kernel_size) / stride + 1;
        
        // Initialize output tensor
        std::vector<float> output_data(batch_size * out_channels * output_height * output_width, 0.0f);
        
        // Perform convolution
        #pragma omp parallel for collapse(4)
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t oc = 0; oc < out_channels; oc++) {
                for (size_t oh = 0; oh < output_height; oh++) {
                    for (size_t ow = 0; ow < output_width; ow++) {
                        float sum = 0.0f;
                        
                        for (size_t ic = 0; ic < in_channels; ic++) {
                            for (size_t kh = 0; kh < kernel_size; kh++) {
                                for (size_t kw = 0; kw < kernel_size; kw++) {
                                    int ih = static_cast<int>(oh * stride + kh) - static_cast<int>(padding);
                                    int iw = static_cast<int>(ow * stride + kw) - static_cast<int>(padding);
                                    
                                    if (ih >= 0 && ih < static_cast<int>(height) && 
                                        iw >= 0 && iw < static_cast<int>(width)) {
                                        size_t input_idx = b * (in_channels * height * width) + 
                                                         ic * (height * width) + 
                                                         ih * width + iw;
                                        size_t weight_idx = (ic * kernel_size * kernel_size + 
                                                           kh * kernel_size + kw) * out_channels + oc;
                                        
                                        sum += input->data[input_idx] * weights->data[weight_idx];
                                    }
                                }
                            }
                        }
                        
                        // Add bias
                        sum += bias->data[oc];
                        
                        size_t output_idx = b * (out_channels * output_height * output_width) +
                                          oc * (output_height * output_width) +
                                          oh * output_width + ow;
                        output_data[output_idx] = sum;
                    }
                }
            }
        }
        
        auto output = std::make_shared<Tensor>(output_data, batch_size, 
                                             out_channels * output_height * output_width,
                                             true, "Conv2d");
        output->parents = {input, weights, bias};
        return output;
    }

    std::vector<std::shared_ptr<Tensor>> get_parameters() {
        return {weights, bias};
    }
};

class MaxPool2D : public Layer {
public:
    size_t kernel_size;
    size_t stride;

    MaxPool2D(size_t kernel_size, size_t stride)
        : kernel_size(kernel_size), stride(stride) {}

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override {
        size_t batch_size = input->rows;
        size_t channels = 32;  // From previous Conv2D
        size_t height = static_cast<size_t>(std::sqrt(input->cols / channels));
        size_t width = height;
        
        size_t output_height = (height - kernel_size) / stride + 1;
        size_t output_width = (width - kernel_size) / stride + 1;
        size_t output_size = channels * output_height * output_width;
        
        std::vector<float> output_data(batch_size * output_size);
        
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t c = 0; c < channels; c++) {
                for (size_t oh = 0; oh < output_height; oh++) {
                    for (size_t ow = 0; ow < output_width; ow++) {
                        float max_val = -std::numeric_limits<float>::infinity();
                        
                        for (size_t kh = 0; kh < kernel_size; kh++) {
                            for (size_t kw = 0; kw < kernel_size; kw++) {
                                size_t ih = oh * stride + kh;
                                size_t iw = ow * stride + kw;
                                
                                size_t input_idx = b * (channels * height * width) +
                                                 c * (height * width) +
                                                 ih * width + iw;
                                
                                if (input_idx < input->data.size()) {
                                    max_val = std::max(max_val, input->data[input_idx]);
                                }
                            }
                        }
                        
                        size_t output_idx = b * output_size +
                                          c * (output_height * output_width) +
                                          oh * output_width + ow;
                        output_data[output_idx] = max_val;
                    }
                }
            }
        }
        
        auto output = std::make_shared<Tensor>(output_data, batch_size, output_size,
                                             input->requires_grad, "maxpool2d");
        output->parents = {input};
        return output;
    }

    std::vector<std::shared_ptr<Tensor>> get_parameters() {
        return {};
    }

};

class Flatten : public Layer {
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override {
        // Calculate actual feature dimensions
        size_t batch_size = input->rows;
        size_t features = input->cols;
        
        // Create new tensor that preserves all values but reshapes them
        std::vector<float> flattened_data(input->data);
        auto output = std::make_shared<Tensor>(flattened_data, batch_size, features, 
                                             input->requires_grad, "flatten");
        output->parents = {input};
        return output;
    }

    std::vector<std::shared_ptr<Tensor>> get_parameters() {
        return {};
    }
};

class BatchNorm2D : public Layer {
public:
    std::shared_ptr<Tensor> gamma;
    std::shared_ptr<Tensor> beta;
    std::shared_ptr<Tensor> running_mean;
    std::shared_ptr<Tensor> running_var;
    float eps = 1e-5;
    float momentum = 0.1;
    bool training = true;
    size_t num_features;

    BatchNorm2D(size_t num_features) : num_features(num_features) {
        gamma = std::make_shared<Tensor>(1, num_features, true, "bn_gamma");
        beta = std::make_shared<Tensor>(1, num_features, true, "bn_beta");
        running_mean = std::make_shared<Tensor>(1, num_features, false, "bn_mean");
        running_var = std::make_shared<Tensor>(1, num_features, false, "bn_var");
        
        std::fill(gamma->data.begin(), gamma->data.end(), 1.0f);
        std::fill(beta->data.begin(), beta->data.end(), 0.0f);
        std::fill(running_mean->data.begin(), running_mean->data.end(), 0.0f);
        std::fill(running_var->data.begin(), running_var->data.end(), 1.0f);
    }

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override {
        size_t batch_size = input->rows;
        size_t height = static_cast<size_t>(std::sqrt(input->cols / num_features));
        size_t width = height;
        size_t spatial_size = height * width;

        if (training) {
            std::vector<float> batch_mean(num_features, 0.0f);
            std::vector<float> batch_var(num_features, 0.0f);

            // Calculate mean across batch and spatial dimensions
            for (size_t c = 0; c < num_features; c++) {
                for (size_t b = 0; b < batch_size; b++) {
                    for (size_t h = 0; h < height; h++) {
                        for (size_t w = 0; w < width; w++) {
                            size_t idx = b * (num_features * spatial_size) + 
                                       c * spatial_size + 
                                       h * width + w;
                            batch_mean[c] += input->data[idx];
                        }
                    }
                }
                batch_mean[c] /= (batch_size * spatial_size);
            }

            // Calculate variance
            for (size_t c = 0; c < num_features; c++) {
                for (size_t b = 0; b < batch_size; b++) {
                    for (size_t h = 0; h < height; h++) {
                        for (size_t w = 0; w < width; w++) {
                            size_t idx = b * (num_features * spatial_size) + 
                                       c * spatial_size + 
                                       h * width + w;
                            float diff = input->data[idx] - batch_mean[c];
                            batch_var[c] += diff * diff;
                        }
                    }
                }
                batch_var[c] = batch_var[c] / (batch_size * spatial_size) + eps;
            }

            // Update running statistics
            for (size_t c = 0; c < num_features; c++) {
                running_mean->data[c] = momentum * batch_mean[c] + (1 - momentum) * running_mean->data[c];
                running_var->data[c] = momentum * batch_var[c] + (1 - momentum) * running_var->data[c];
            }

            // Normalize and scale
            std::vector<float> output_data(input->data.size());
            for (size_t b = 0; b < batch_size; b++) {
                for (size_t c = 0; c < num_features; c++) {
                    for (size_t h = 0; h < height; h++) {
                        for (size_t w = 0; w < width; w++) {
                            size_t idx = b * (num_features * spatial_size) + 
                                       c * spatial_size + 
                                       h * width + w;
                            float normalized = (input->data[idx] - batch_mean[c]) / std::sqrt(batch_var[c]);
                            output_data[idx] = gamma->data[c] * normalized + beta->data[c];
                        }
                    }
                }
            }

            auto output = std::make_shared<Tensor>(output_data, batch_size, input->cols, 
                                                 input->requires_grad, "batchnorm");
            output->parents = {input, gamma, beta};
            return output;
        } else {
            std::vector<float> output_data(input->data.size());
            for (size_t b = 0; b < batch_size; b++) {
                for (size_t c = 0; c < num_features; c++) {
                    for (size_t h = 0; h < height; h++) {
                        for (size_t w = 0; w < width; w++) {
                            size_t idx = b * (num_features * spatial_size) + 
                                       c * spatial_size + 
                                       h * width + w;
                            float normalized = (input->data[idx] - running_mean->data[c]) / 
                                             std::sqrt(running_var->data[c] + eps);
                            output_data[idx] = gamma->data[c] * normalized + beta->data[c];
                        }
                    }
                }
            }

            auto output = std::make_shared<Tensor>(output_data, batch_size, input->cols, 
                                                 input->requires_grad, "batchnorm");
            output->parents = {input, gamma, beta};
            return output;
        }
    }

    std::vector<std::shared_ptr<Tensor>> get_parameters() {
        return {gamma, beta};
    }
};

class Dropout : public Layer {
public:
    float dropout_rate;
    bool training;
    std::vector<bool> mask;

    Dropout(float dropout_rate) : dropout_rate(dropout_rate), training(true) {}

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override {
        if (!training) {
            return input;
        }

        std::vector<float> output_data(input->data.size());
        mask.resize(input->data.size());

        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution d(1.0f - dropout_rate);

        for (size_t i = 0; i < input->data.size(); i++) {
            mask[i] = d(gen);
            output_data[i] = mask[i] ? input->data[i] / (1.0f - dropout_rate) : 0.0f;
        }

        auto output = std::make_shared<Tensor>(output_data, input->rows, input->cols, 
                                               input->requires_grad, "dropout");
        output->parents = {input};
        return output;
    }

    std::vector<std::shared_ptr<Tensor>> get_parameters() {
        return {};
    }
};

class Sequential : public Layer {
public:
    std::vector<std::shared_ptr<Layer>> layers;

    void add(std::shared_ptr<Layer> layer) {
        layers.push_back(layer);
    }

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override {
        auto output = input;
        for (auto& layer : layers) {
            output = layer->forward(output);
        }
        return output;
    }

    void train() {
        for (auto& layer : layers) {
            if (auto bn = std::dynamic_pointer_cast<BatchNorm2D>(layer)) {
                bn->training = true;
            }
            if (auto dropout = std::dynamic_pointer_cast<Dropout>(layer)) {
                dropout->training = true;
            }
        }   
    }

    void eval() {
        for (auto& layer : layers) {
            if (auto bn = std::dynamic_pointer_cast<BatchNorm2D>(layer)) {
                bn->training = false;
            }
            if (auto dropout = std::dynamic_pointer_cast<Dropout>(layer)) {
                dropout->training = false;
            }
        }
    }

    std::vector<std::shared_ptr<Tensor>> get_parameters() {
        std::vector<std::shared_ptr<Tensor>> parameters;
        for (auto& layer : layers) {
            if (auto linear = std::dynamic_pointer_cast<Linear>(layer)) {
                auto layer_params = linear->get_parameters();
                parameters.insert(parameters.end(), layer_params.begin(), layer_params.end());
            } else if (auto conv = std::dynamic_pointer_cast<Conv2D>(layer)) {
                auto layer_params = conv->get_parameters();
                parameters.insert(parameters.end(), layer_params.begin(), layer_params.end());
            } else if (auto bn = std::dynamic_pointer_cast<BatchNorm2D>(layer)) {
                auto layer_params = bn->get_parameters();
                parameters.insert(parameters.end(), layer_params.begin(), layer_params.end());
            }
        }
        return parameters;
    }
};

std::pair<std::vector<std::shared_ptr<Tensor>>, std::vector<std::shared_ptr<Tensor>>> load_mnist(const std::string& filename, int max_samples = -1) {
    std::vector<std::shared_ptr<Tensor>> inputs;
    std::vector<std::shared_ptr<Tensor>> targets;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    std::string line;
    
    // Skip header
    std::getline(file, line);
    
    int count = 0;
    while (std::getline(file, line) && (max_samples == -1 || count < max_samples)) {
        std::stringstream ss(line);
        std::string value;
        
        // First value is the label
        std::getline(ss, value, ',');
        int label = std::stoi(value);
        
        // Create target with single value instead of one-hot encoding
        std::vector<float> target_data(1, static_cast<float>(label));
        targets.push_back(std::make_shared<Tensor>(target_data, 1, 1, true));
        
        // Read pixel values
        std::vector<float> input_data;
        while (std::getline(ss, value, ',')) {
            // Normalize pixel values to [-1,1] instead of [0,1]
            input_data.push_back((std::stof(value) / 127.5f) - 1.0f);
        }
        
        inputs.push_back(std::make_shared<Tensor>(input_data, 1, 784, true));
        count++;
    }
    
    return {inputs, targets};
}

int main() {
    std::cout << "Loading MNIST dataset..." << std::endl;
    auto [inputs, targets] = load_mnist("mnist_train.csv", 1000);
    std::cout << "Loaded " << inputs.size() << " samples" << std::endl;

    // Model architecture
    auto model = std::make_shared<Sequential>();
    model->add(std::make_shared<Conv2D>(1, 16, 3, 1, 1));
    model->add(std::make_shared<BatchNorm2D>(16));
    model->add(std::make_shared<ReLU>());
    model->add(std::make_shared<Dropout>(0.1));
    model->add(std::make_shared<Conv2D>(16, 32, 3, 2, 1));
    model->add(std::make_shared<BatchNorm2D>(32));
    model->add(std::make_shared<ReLU>());
    model->add(std::make_shared<Dropout>(0.1));
    model->add(std::make_shared<MaxPool2D>(2, 2));            
    model->add(std::make_shared<Flatten>());                   
    model->add(std::make_shared<Linear>(7*7*32, 128));        
    model->add(std::make_shared<ReLU>());
    model->add(std::make_shared<Linear>(128, 10));         
    model->add(std::make_shared<Softmax>());

    auto parameters = model->get_parameters();
    SGD optimizer(parameters, 0.001f);

    const size_t epochs = 50;
    const size_t batch_size = 64;
    const size_t print_every = 1;

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0f;
        size_t correct = 0;
        size_t total = 0;

        std::vector<size_t> indices(inputs.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(indices.begin(), indices.end(), gen);

        size_t num_batches = (inputs.size() + batch_size - 1) / batch_size;

        for (size_t batch = 0; batch < num_batches; batch++) {
            size_t current_batch_size = std::min(batch_size, inputs.size() - batch * batch_size);

            std::vector<float> batch_input_data;
            std::vector<float> batch_target_data;
            batch_input_data.reserve(current_batch_size * 784);
            batch_target_data.reserve(current_batch_size * 10);

            for (size_t i = 0; i < current_batch_size; i++) {
                size_t idx = indices[batch * batch_size + i];
                batch_input_data.insert(batch_input_data.end(), inputs[idx]->data.begin(), inputs[idx]->data.end());

                std::vector<float> one_hot(10, 0.0f);
                one_hot[static_cast<size_t>(targets[idx]->data[0])] = 1.0f;
                batch_target_data.insert(batch_target_data.end(), one_hot.begin(), one_hot.end());
            }

            auto batch_input = std::make_shared<Tensor>(batch_input_data, current_batch_size, 784, true);
            auto batch_target = std::make_shared<Tensor>(batch_target_data, current_batch_size, 10, true);

            model->train();
            optimizer.zero_grad();
            auto batch_output = model->forward(batch_input);
            auto loss = Tensor::cross_entropy(batch_output, batch_target);

            loss->grad = std::vector<float>(loss->data.size(), 1.0f);
            loss->backward();
            optimizer.step();

            float batch_loss = loss->data[0];
            epoch_loss += batch_loss * current_batch_size;

            // Track correct predictions per batch
            size_t batch_correct = 0;
            for (size_t i = 0; i < current_batch_size; i++) {
                size_t predicted = 0;
                float max_val = batch_output->data[i * 10];
                for (size_t j = 1; j < 10; j++) {
                    if (batch_output->data[i * 10 + j] > max_val) {
                        max_val = batch_output->data[i * 10 + j];
                        predicted = j;
                    }
                }
                // Get actual label
                size_t actual = static_cast<size_t>(targets[indices[batch * batch_size + i]]->data[0]);
                if (predicted == actual) {
                    batch_correct++;
                }
            }
            correct += batch_correct;
            total += current_batch_size;

            if (batch % print_every == 0) {
                float accuracy = static_cast<float>(correct) / total * 100.0f;
                std::cout << "Epoch " << epoch << " - Batch " << batch << "/" << num_batches << " - Loss: " << batch_loss << " - Accuracy: " << accuracy << "%" << std::endl;
            }
        }
    }
}

// int main() {
//     try {
//         // Load dataset
//         std::cout << "Loading MNIST dataset..." << std::endl;
//         auto [inputs, targets] = load_mnist("mnist_train.csv", 1000);
//         std::cout << "Loaded " << inputs.size() << " samples" << std::endl;
        
//         // Create model with larger architecture
//         auto model = std::make_shared<Sequential>();
//         model->add(std::make_shared<Linear>(784, 512, true));  // Larger first layer
//         model->add(std::make_shared<ReLU>());
//         model->add(std::make_shared<Linear>(512, 256, true));  // Larger second layer
//         model->add(std::make_shared<ReLU>());
//         model->add(std::make_shared<Linear>(256, 10, true));
//         model->add(std::make_shared<Softmax>());
        
//         auto parameters = model->get_parameters();
//         SGD optimizer(parameters, 0.01f);  // Increased learning rate
        
//         const size_t epochs = 10;
//         const size_t batch_size = 32;
//         const size_t print_every = 1;
        
//         for (size_t epoch = 0; epoch < epochs; epoch++) {
//             float epoch_loss = 0.0f;
//             size_t correct = 0;
//             size_t total = 0;
            
//             // Shuffle dataset
//             std::vector<size_t> indices(inputs.size());
//             std::iota(indices.begin(), indices.end(), 0);
//             std::random_device rd;
//             std::mt19937 gen(rd());
//             std::shuffle(indices.begin(), indices.end(), gen);
            
//             // Handle partial batches correctly
//             size_t num_batches = (inputs.size() + batch_size - 1) / batch_size;
            
//             for (size_t batch = 0; batch < num_batches; batch++) {
//                 // Calculate actual batch size (might be smaller for last batch)
//                 size_t current_batch_size = std::min(batch_size, 
//                                                    inputs.size() - batch * batch_size);
                
//                 // Create batch tensors
//                 std::vector<float> batch_input_data;
//                 std::vector<float> batch_target_data;
//                 batch_input_data.reserve(current_batch_size * 784);
//                 batch_target_data.reserve(current_batch_size * 10);
                
//                 for (size_t i = 0; i < current_batch_size; i++) {
//                     size_t idx = indices[batch * batch_size + i];
//                     batch_input_data.insert(batch_input_data.end(), 
//                                           inputs[idx]->data.begin(), 
//                                           inputs[idx]->data.end());
                    
//                     // Create one-hot target
//                     std::vector<float> one_hot(10, 0.0f);
//                     one_hot[static_cast<size_t>(targets[idx]->data[0])] = 1.0f;
//                     batch_target_data.insert(batch_target_data.end(),
//                                            one_hot.begin(),
//                                            one_hot.end());
//                 }
                
//                 auto batch_input = std::make_shared<Tensor>(batch_input_data, 
//                                                           current_batch_size, 784, true);
//                 auto batch_target = std::make_shared<Tensor>(batch_target_data, 
//                                                            current_batch_size, 10, true);
                
//                 // Forward pass
//                 optimizer.zero_grad();
//                 auto batch_output = model->forward(batch_input);
//                 auto loss = Tensor::cross_entropy(batch_output, batch_target);
                
//                 // Backward pass - no need for extra scaling
//                 loss->grad = std::vector<float>(loss->data.size(), 1.0f);
//                 loss->backward();
                
//                 // Update weights
//                 optimizer.step();
                
//                 // Compute batch statistics
//                 float batch_loss = loss->data[0];  // Already averaged in cross_entropy
//                 epoch_loss += batch_loss * current_batch_size;
                
//                 // Calculate accuracy
//                 for (size_t i = 0; i < current_batch_size; i++) {
//                     size_t predicted = 0;
//                     float max_val = batch_output->data[i * 10];
//                     for (size_t j = 1; j < 10; j++) {
//                         if (batch_output->data[i * 10 + j] > max_val) {
//                             max_val = batch_output->data[i * 10 + j];
//                             predicted = j;
//                         }
//                     }
                    
//                     size_t actual = static_cast<size_t>(targets[indices[batch * batch_size + i]]->data[0]);
//                     if (predicted == actual) {
//                         correct++;
//                     }
//                     total++;
//                 }
                
//                 if (batch % print_every == 0) {
//                     float accuracy = static_cast<float>(correct) / total * 100.0f;
//                     std::cout << "Epoch " << epoch << " - Batch " << batch 
//                              << "/" << num_batches 
//                              << " - Loss: " << batch_loss
//                              << " - Accuracy: " << accuracy << "%" 
//                              << std::endl;
//                 }
//             }
            
//             float accuracy = static_cast<float>(correct) / total * 100.0f;
//             float avg_loss = epoch_loss / total;
//             std::cout << "\nEpoch " << epoch << " completed"
//                      << " - Average Loss: " << avg_loss
//                      << " - Accuracy: " << accuracy << "%\n" 
//                      << std::endl;
//         }
        
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return 1;
//     }
    
//     return 0;
// }


// int main() {
//     std::vector<std::shared_ptr<Tensor>> inputs = {
//         std::make_shared<Tensor>(std::vector<float>{0.0f, 0.0f}, 1, 2, true),
//         std::make_shared<Tensor>(std::vector<float>{0.0f, 1.0f}, 1, 2, true),
//         std::make_shared<Tensor>(std::vector<float>{1.0f, 0.0f}, 1, 2, true),
//         std::make_shared<Tensor>(std::vector<float>{1.0f, 1.0f}, 1, 2, true)
//     };
//     std::vector<std::shared_ptr<Tensor>> targets = {
//         std::make_shared<Tensor>(std::vector<float>{0.0f}, 1, 1, true),
//         std::make_shared<Tensor>(std::vector<float>{1.0f}, 1, 1, true),
//         std::make_shared<Tensor>(std::vector<float>{1.0f}, 1, 1, true),
//         std::make_shared<Tensor>(std::vector<float>{0.0f}, 1, 1, true)
//     };

//     auto model = std::make_shared<Sequential>();
//     model->add(std::make_shared<Linear>(2, 10, true));
//     model->add(std::make_shared<Tanh>());
//     model->add(std::make_shared<Linear>(10, 1, true));

//     auto parameters = model->get_parameters();
//     SGD optimizer(parameters, 0.1f); 

//     const size_t epochs = 1000;
//     const size_t print_every = 100;

//     for (size_t epoch = 0; epoch < epochs; epoch++) {
//         std::shared_ptr<Tensor> loss = std::make_shared<Tensor>(std::vector<float>{0.0f}, 1, 1, true);
//         for (size_t i = 0; i < inputs.size(); i++) {
//             auto input = inputs[i];
//             auto target = targets[i];
//             auto output = model->forward(input);
//             loss = Tensor::add(loss, Tensor::binary_cross_entropy(output, target));
//         }
        
//         optimizer.zero_grad();
        
//         loss->grad = std::vector<float>(loss->data.size(), 1.0f / inputs.size());  
//         loss->backward();
//         optimizer.step();

//         // Print progress
//         if (epoch % print_every == 0) {
//             std::cout << "\nEpoch " << epoch << " - Loss: " << loss->data[0] << std::endl;
            
//             // Print gradients for diagnosis
//             std::cout << "Gradients:" << std::endl;
//             for (auto& param : parameters) {
//                 param->debug_gradient_flow();
//             }

//             // Check predictions
//             bool all_correct = true;
//             for (size_t i = 0; i < inputs.size(); i++) {
//                 auto output = model->forward(inputs[i]);
//                 int predicted = (output->data[0] > 0.5f ? 1 : 0);
//                 int expected = (targets[i]->data[0] > 0.5f ? 1 : 0);
//                 all_correct &= (predicted == expected);
                
//                 std::cout << "Input: [" << inputs[i]->data[0] << ", " << inputs[i]->data[1] 
//                          << "] Expected: " << expected 
//                          << " Raw: " << output->data[0]
//                          << " Predicted: " << predicted 
//                          << std::endl;
//             }
//         }
//     }
// }
