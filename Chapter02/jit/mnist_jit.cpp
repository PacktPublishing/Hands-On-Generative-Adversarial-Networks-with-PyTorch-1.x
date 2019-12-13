#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

using namespace torch;

int main(int argc, const char* argv[]) {

    torch::Device device = torch::kCUDA;

    // Deserialize the ScriptModule from a file using torch::jit::load().
    std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("../model_jit.pth");
    module->to(device);

    assert(module != nullptr);
    std::cout << "model loading ok\n";

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1000, 1, 28, 28}).to(device));

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int64_t itr = 1; itr <= 10; ++itr) {
        at::Tensor output = module->forward(inputs).toTensor();
    }
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    std::cout << "Time elapsed = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

    return 0;
}
