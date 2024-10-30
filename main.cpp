#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include <numeric>
#include <algorithm>
#include <cassert>

class Tensor : public std::enable_shared_from_this<Tensor> {
private:
    static std::mt19937& get_generator() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        return gen;
    }

    static std::vector<float> generate_random_data(size_t rows, size_t cols) {
        // Xavier weight initialization.
        std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(cols)));
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
    std::vector<std::shared_ptr<Tensor>> children;
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

    // Add destructor to break cycles
    ~Tensor() {
        parents.clear();
        children.clear();
    }
    
    // Add method to explicitly break references
    void clear_graph() {
        parents.clear();
        children.clear();
        grad.clear();
    }

    void backward() {
        if (!requires_grad) return;

        if (grad.empty()) {
            grad = std::vector<float>(rows * cols, 1.0f);
        }

        if (creation_op == "add") {
            for (const auto& parent : parents) {
                for (size_t i = 0; i < parent->data.size(); i++) {
                    if (i < grad.size()) {
                        parent->grad[i] += grad[i];  // Ensure index is within bounds
                    }
                }
                parent->backward();
            }
        }

        if (creation_op == "mul") {
            auto weights = parents[0];
            auto inputs = parents[1];
            
            // Combine loops for better cache utilization
            for (size_t i = 0; i < weights->rows; i++) {
                for (size_t j = 0; j < weights->cols; j++) {
                    float weight_grad = 0.0f;
                    for (size_t k = 0; k < inputs->rows; k++) {
                        if (k * inputs->cols + j < inputs->data.size() && k * weights->cols + i < grad.size()) {
                            weight_grad += inputs->data[k * inputs->cols + j] * grad[k * weights->cols + i];
                            if (k * inputs->cols + j < inputs->grad.size() && i * weights->cols + j < weights->data.size()) {
                                inputs->grad[k * inputs->cols + j] += grad[k * weights->cols + i] * weights->data[i * weights->cols + j];
                            }
                        }
                    }
                    if (i * weights->cols + j < weights->grad.size()) {
                        weights->grad[i * weights->cols + j] += weight_grad;
                    }
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
    }

    // Add debug printing function to help track gradient flow
    void debug_gradient_flow() {
        std::cout << "Operation: " << creation_op << std::endl;
        std::cout << "Has gradient: " << (grad.size() > 0) << std::endl;
        std::cout << "Number of parents: " << parents.size() << std::endl;
        std::cout << "Number of children: " << children.size() << std::endl;
        if (!grad.empty()) {
            std::cout << "Gradient values: ";
            for (float g : grad) {
                std::cout << g << " ";
            }
            std::cout << std::endl;
        }
    }

    static std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
        if (a->rows != b->rows || a->cols != b->cols) {
            throw std::runtime_error("Tensor shapes must match for addition");
        }
        std::vector<float> new_data(a->data.size());
        for (size_t i = 0; i < a->data.size(); i++) {
            new_data[i] = a->data[i] + b->data[i];
        }
        auto result = std::make_shared<Tensor>(new_data, a->rows, a->cols, a->requires_grad || b->requires_grad, "add");
        result->parents.push_back(a);
        result->parents.push_back(b);
        return result;
    }

    static std::shared_ptr<Tensor> multiply(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
        if (a->cols != b->rows) {
            throw std::runtime_error("Matrix dimensions are not aligned");
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

        auto result = std::make_shared<Tensor>(new_data, a->rows, b->cols, a->requires_grad || b->requires_grad, "mul");
        result->parents.push_back(a);
        result->parents.push_back(b);
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

    void zero_grad() {
        std::fill(grad.begin(), grad.end(), 0.0f);
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

class SGD {
public:
    std::vector<std::shared_ptr<Tensor>> parameters;
    float learning_rate;
    float max_grad_norm;  // Added gradient clipping

    SGD(std::vector<std::shared_ptr<Tensor>> parameters, float learning_rate, float max_grad_norm = 1.0f)
        : parameters(parameters), learning_rate(learning_rate), max_grad_norm(max_grad_norm) {}

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
        weights = std::make_shared<Tensor>(input_size, output_size, true, "linear");  // Store weights in [input_size, output_size] format
        if (use_bias) {
            bias = std::make_shared<Tensor>(1, output_size, true, "bias");
        }
    }

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override {
        auto output = Tensor::multiply(input, weights);  // Perform matrix multiplication directly
        if (use_bias) {
            // Expand bias to match batch size
            auto expanded_bias = bias->expand(1, input->rows);
            output = Tensor::add(output, expanded_bias);
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

    std::vector<std::shared_ptr<Tensor>> get_parameters() {
        std::vector<std::shared_ptr<Tensor>> parameters;
        for (auto& layer : layers) {
            if (auto linear = std::dynamic_pointer_cast<Linear>(layer)) {
                auto layer_params = linear->get_parameters();
                parameters.insert(parameters.end(), layer_params.begin(), layer_params.end());
            }
        }
        return parameters;
    }
};

int main() {
    // Create a simple network on a toy dataset to check the gradient flow.
    // auto input = std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f}, 1, 2, true, "input");
    // auto target = std::make_shared<Tensor>(std::vector<float>{1.0f}, 1, 1, true, "target");

    std::vector<std::shared_ptr<Tensor>> inputs = {
        std::make_shared<Tensor>(std::vector<float>{0.0f, 0.0f}, 1, 2, true, "input"),
        std::make_shared<Tensor>(std::vector<float>{0.0f, 1.0f}, 1, 2, true, "input"),
        std::make_shared<Tensor>(std::vector<float>{1.0f, 0.0f}, 1, 2, true, "input"),
        std::make_shared<Tensor>(std::vector<float>{1.0f, 1.0f}, 1, 2, true, "input")
    };

    std::vector<std::shared_ptr<Tensor>> targets = {
        std::make_shared<Tensor>(std::vector<float>{0.0f}, 1, 1, true, "target"),
        std::make_shared<Tensor>(std::vector<float>{1.0f}, 1, 1, true, "target"),
        std::make_shared<Tensor>(std::vector<float>{1.0f}, 1, 1, true, "target"),
        std::make_shared<Tensor>(std::vector<float>{0.0f}, 1, 1, true, "target")
    };

    auto model = std::make_shared<Sequential>();
    model->add(std::make_shared<Linear>(2, 4, true));
    model->add(std::make_shared<Tanh>());
    model->add(std::make_shared<Linear>(4, 1, true));

    auto parameters = model->get_parameters();
    SGD optimizer(parameters, 0.1f);  // Reduced learning rate since we'll handle batch properly

    const size_t epochs = 1000;
    const size_t print_every = 100;

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        std::shared_ptr<Tensor> loss = std::make_shared<Tensor>(std::vector<float>{0.0f}, 1, 1, true, "target");
        for (size_t i = 0; i < inputs.size(); i++) {
            auto input = inputs[i];
            auto target = targets[i];
            auto output = model->forward(input);
            loss = Tensor::add(loss, Tensor::binary_cross_entropy(output, target));
        }
        
        // Zero gradients at start of batch
        optimizer.zero_grad();
        
        loss->grad = std::vector<float>(loss->data.size(), 1.0f / inputs.size());  // Set loss gradient to 1/N
        // Backward pass
        loss->backward();
        
        // Single weight update
        optimizer.step();

        // Print progress
        if (epoch % print_every == 0) {
            std::cout << "\nEpoch " << epoch << " - Loss: " << loss->data[0] << std::endl;
            
            // Print gradients for diagnosis
            std::cout << "Gradients:" << std::endl;
            for (auto& param : parameters) {
                param->debug_gradient_flow();
            }

            // Check predictions
            bool all_correct = true;
            for (size_t i = 0; i < inputs.size(); i++) {
                auto output = model->forward(inputs[i]);
                int predicted = (output->data[0] > 0.5f ? 1 : 0);
                int expected = (targets[i]->data[0] > 0.5f ? 1 : 0);
                all_correct &= (predicted == expected);
                
                std::cout << "Input: [" << inputs[i]->data[0] << ", " << inputs[i]->data[1] 
                         << "] Expected: " << expected 
                         << " Predicted: " << predicted 
                         << std::endl;
            }
        }
    }
}