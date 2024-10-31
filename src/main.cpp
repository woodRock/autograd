// main.cpp
#include <iostream>
#include <random>
#include <chrono>
#include <memory>
#include <vector>

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    size_t rows;
    size_t cols;
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<std::shared_ptr<Tensor>> parents;
    bool requires_grad;
    std::string creation_op;

    // Just declare these - don't implement here
    Tensor(size_t rows, size_t cols, bool requires_grad = false, std::string creation_op = "");
    Tensor(const std::vector<float>& data, size_t rows, size_t cols, 
           bool requires_grad = false, std::string creation_op = "");
    static std::shared_ptr<Tensor> multiply(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    void backward();
};

int main() {
    // Create test matrices
    size_t size = 1024;  // Test with 1024x1024 matrices
    
    // // Create input matrix A
    auto a = std::make_shared<Tensor>(std::vector<float>(size*size, 0.0f), 1024, 1024, true, "a");  // 1024x1024 matrix
    
    // // Create input matrix B
    auto b = std::make_shared<Tensor>(std::vector<float>(size*size, 0.0f), 1024, 1024, true, "b");  // 1024x1024 matrix

    // Print sample of input matrices
    std::cout << "Matrix A" << std::endl;
    std::cout << a << std::endl;

    std::cout << "Matrix B" << std::endl;
    std::cout << b << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

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
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Matrix multiplication took " << duration.count() << " milliseconds\n";

    // Measure time for CUDA multiplication
    start = std::chrono::high_resolution_clock::now();
    
    // Perform matrix multiplication
    auto c = Tensor::multiply(a, b);
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Print sample of result matrix
    std::cout << "Result Matrix C (top-left corner):\n";
    std::cout << c << std::endl;

    std::cout << "Matrix multiplication took " << duration.count() << " milliseconds\n";

    // Test backward pass
    c->grad = std::vector<float>(size * size, 1.0f);  // Set gradient to all ones
    start = std::chrono::high_resolution_clock::now();
    c->backward();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Backward pass took " << duration.count() << " milliseconds\n";

    return 0;
}