#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include <numeric>
#include <algorithm>

class Tensor : public std::enable_shared_from_this<Tensor> {
private:
    static std::vector<float> generate_random_data(size_t size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> data(size);
        for (size_t i = 0; i < size; ++i) {
            data[i] = dist(gen);
        }
        return data;
    }

public:
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<std::shared_ptr<Tensor>> children;
    std::vector<std::shared_ptr<Tensor>> parents;
    bool requires_grad;
    std::string creation_op;

    Tensor() : requires_grad(false), creation_op("") {}

    Tensor(const std::vector<float>& data, bool requires_grad = false, std::string creation_op = "") 
        : data(data), requires_grad(requires_grad), creation_op(creation_op) {
        grad = std::vector<float>(data.size(), 0);
    }

    Tensor(size_t size, bool requires_grad = false, std::string creation_op = "") 
        : data(generate_random_data(size)), requires_grad(requires_grad), creation_op(creation_op) {
        if (requires_grad) {
            grad = std::vector<float>(size, 0);
        }
    }

    void backward() {
        if (!requires_grad) return;

        if (grad.empty()) {
            grad = std::vector<float>(data.size(), 1);
        }

        if (creation_op == "add") {
            for (const auto& parent : parents) {
                for (size_t i = 0; i < parent->data.size(); i++) {
                    parent->grad[i] += grad[i];
                }
                parent->backward();
            }
        }  
        if (creation_op == "mul") {
            if (parents.size() == 2) {    
                auto weights = parents[0];
                auto inputs = parents[1];
                size_t m = weights->data.size() / inputs->data.size();
                size_t n = inputs->data.size();
                
                // Gradient with respect to weights
                for (size_t i = 0; i < m; i++) {
                    for (size_t j = 0; j < n; j++) {
                        weights->grad[i * n + j] += grad[i] * inputs->data[j];
                    }
                }
                
                // Gradient with respect to inputs
                for (size_t j = 0; j < n; j++) {
                    for (size_t i = 0; i < m; i++) {
                        inputs->grad[j] += grad[i] * weights->data[i * n + j];
                    }
                }
                
                weights->backward();
                inputs->backward();
            }
        }
        if (creation_op == "relu") {
            auto parent = parents[0];
            for (size_t j = 0; j < parent->data.size(); j++) {
                parent->grad[j] += grad[j] * (data[j] > 0 ? 1 : 0);
            }
            parent->backward();
        }
        if (creation_op == "sigmoid") {
            if (!parents.empty()) {
                auto parent = parents[0];
                for (size_t j = 0; j < parent->data.size(); j++) {
                    float sig = 1.0f / (1.0f + std::exp(-data[j])); // Use data[j] instead of parent->data[j]
                    parent->grad[j] += grad[j] * sig * (1.0f - sig);
                }
                parent->backward();
            }
        }
        if (creation_op == "bce") {
            if (parents.size() >= 2) {
                auto pred = parents[0];
                auto target = parents[1];
                for (size_t j = 0; j < pred->data.size(); j++) {
                    float p = std::max(std::min(pred->data[j], 1.0f - 1e-7f), 1e-7f);
                    pred->grad[j] += grad[j] * (p - target->data[j]) / (p * (1 - p));
                }
                pred->backward();
            }
        }
    }

    static std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
        std::vector<float> new_data(a->data.size());
        for (size_t i = 0; i < a->data.size(); i++) {
            new_data[i] = a->data[i] + b->data[i];
        }
        auto result = std::make_shared<Tensor>(new_data, a->requires_grad || b->requires_grad, "add");
        result->parents.push_back(a);
        result->parents.push_back(b);
        return result;
    }

    static std::shared_ptr<Tensor> multiply(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
        // Assume a is the weight matrix (m x n) and b is the input vector (n x 1)
        size_t m = a->data.size() / b->data.size(); // Number of rows in output
        size_t n = b->data.size(); // Number of columns in weight matrix
        
        std::vector<float> new_data(m);
        for (size_t i = 0; i < m; i++) {
            new_data[i] = 0;
            for (size_t j = 0; j < n; j++) {
                new_data[i] += a->data[i * n + j] * b->data[j];
            }
        }
        
        auto result = std::make_shared<Tensor>(new_data, a->requires_grad || b->requires_grad, "mul");
        result->parents.push_back(a);
        result->parents.push_back(b);
        return result;
    }

    std::shared_ptr<Tensor> relu() {
        std::vector<float> new_data(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            new_data[i] = std::max(0.0f, data[i]);
        }
        auto result = std::make_shared<Tensor>(new_data, requires_grad, "relu");
        result->parents.push_back(this->shared_from_this());
        return result;
    }

    std::shared_ptr<Tensor> sigmoid() {
        std::vector<float> new_data(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            new_data[i] = 1.0f / (1.0f + std::exp(-data[i]));
        }
        auto result = std::make_shared<Tensor>(new_data, requires_grad, "sigmoid");
        result->parents.push_back(this->shared_from_this());

        return result;
    }

    static std::shared_ptr<Tensor> binary_cross_entropy(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target) {
        std::vector<float> new_data(1);  // BCE returns scalar loss
        float total_loss = 0.0f;
        for (size_t i = 0; i < pred->data.size(); i++) {
            float p = std::max(std::min(pred->data[i], 1.0f - 1e-7f), 1e-7f);
            total_loss += -(target->data[i] * std::log(p) + (1 - target->data[i]) * std::log(1 - p));
        }
        new_data[0] = total_loss / pred->data.size();  // Average loss

        auto result = std::make_shared<Tensor>(new_data, pred->requires_grad, "bce");

        result->parents.push_back(pred);
        result->parents.push_back(target);
        return result;
    }

    void zero_grad() {
        std::fill(grad.begin(), grad.end(), 0.0f);
    }

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        os << "[";
        for (size_t i = 0; i < tensor.data.size(); i++) {
            if (i > 0) os << ", ";
            os << tensor.data[i];
        }
        os << "]";
        return os;
    }

    void print_grad() {
        std::cout << "[";
        for (size_t i = 0; i < grad.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << grad[i];
        }
        std::cout << "]";
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
            for (size_t i = 0; i < param->data.size(); ++i) {
                param->data[i] -= learning_rate * param->grad[i];
            }
        }
    }
};

class Linear : public Layer {
public:
    std::shared_ptr<Tensor> weights;
    std::shared_ptr<Tensor> bias;
    bool use_bias;

    Linear(size_t input_size, size_t output_size, bool use_bias = true) {
        this->use_bias = use_bias;
        // Kaiming initialization
        float limit = std::sqrt(2.0f / input_size);
        std::vector<float> weight_data(input_size * output_size);
        std::vector<float> bias_data(output_size);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dis(0.0f, limit);

        for (auto& w : weight_data) {
            w = dis(gen);
        }
        for (auto& b : bias_data) {
            b = dis(gen);
        }

        weights = std::make_shared<Tensor>(weight_data, true);
        bias = std::make_shared<Tensor>(bias_data, true);
    }

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override {
        auto output = Tensor::multiply(weights, input);
        if (use_bias) {
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
    try {
        // Create the XOR dataset
        std::vector<std::shared_ptr<Tensor>> inputs = {
            std::make_shared<Tensor>(std::vector<float>{0.0f, 0.0f}, true),
            std::make_shared<Tensor>(std::vector<float>{0.0f, 1.0f}, true),
            std::make_shared<Tensor>(std::vector<float>{1.0f, 0.0f}, true),
            std::make_shared<Tensor>(std::vector<float>{1.0f, 1.0f}, true)
        };
        
        std::vector<std::shared_ptr<Tensor>> targets = {
            std::make_shared<Tensor>(std::vector<float>{0.0f}, false),
            std::make_shared<Tensor>(std::vector<float>{1.0f}, false),
            std::make_shared<Tensor>(std::vector<float>{1.0f}, false),
            std::make_shared<Tensor>(std::vector<float>{0.0f}, false)
        };

        // Create a network for XOR
        auto model = std::make_shared<Sequential>();
        model->add(std::make_shared<Linear>(2, 4, true));  // Input -> 4 hidden neurons
        model->add(std::make_shared<ReLU>());
        model->add(std::make_shared<Linear>(4, 1, true));  // 4 hidden -> 1 output
        model->add(std::make_shared<Sigmoid>());

        auto parameters = model->get_parameters();
        SGD optimizer(parameters, 0.01f);

        // Training loop with multiple epochs
        const size_t epochs = 10000;
        const size_t print_every = 1000;

        for (size_t epoch = 0; epoch < epochs; epoch++) {
            float total_loss = 0.0f;

            optimizer.zero_grad();

            // Shuffle the dataset.
            std::vector<size_t> indices(inputs.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::random_shuffle(indices.begin(), indices.end());
            std::vector<std::shared_ptr<Tensor>> shuffled_inputs(inputs.size());
            std::vector<std::shared_ptr<Tensor>> shuffled_targets(targets.size());
            for (size_t i = 0; i < inputs.size(); i++) {
                shuffled_inputs[i] = inputs[indices[i]];
                shuffled_targets[i] = targets[indices[i]];
            }

            // Train on all XOR patterns
            for (size_t i = 0; i < inputs.size(); i++) {
                auto output = model->forward(shuffled_inputs[i]);
                auto loss = Tensor::binary_cross_entropy(output, shuffled_targets[i]);
                total_loss += loss->data[0];

                // Set the gradient for the loss.
                loss->grad = std::vector<float>(loss->data.size(), 1.0f);

                // Backpropagate the error.
                loss->backward();
            }

            // Update weights
            optimizer.step();

            // Print progress and check predictions
            if (epoch % print_every == 0 || epoch == epochs - 1) {
                std::cout << "\nEpoch " << epoch << " - Total Loss: " << total_loss << std::endl;
                std::cout << "Predictions:" << std::endl;
                for (size_t i = 0; i < inputs.size(); i++) {
                    auto output = model->forward(inputs[i]);
                    std::cout << "Input: " << *inputs[i] 
                            << " Expected: " << *targets[i] 
                            << " Predicted: " << *output 
                            << " (Rounded: " << (output->data[0] > 0.5f ? 1 : 0) << ")" 
                            << std::endl;
                }
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}