#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>

using namespace std;

class LayerNorm{
  private:
    vector<double> gamma;
    vector<double> beta;
    double epsilon;

  public:
    LayerNorm(size_t embedding_dim, double eps = 1e-5):epsilon(eps){
      gamma.resize(embedding_dim, 1.0);
      beta.resize(embedding_dim, 0.0);  
    }

    vector<vector<vector<double>>> forward(
      const vector<vector<vector<double>>> &input){
          size_t batch_size = input.size();
          size_t seq_len = input[0].size();
          size_t embedding_dim = input[0][0].size();

          vector<vector<vector<double>>> output(
            batch_size, 
            vector<vector<double>>(seq_len,
              vector<double>(embedding_dim,0.0))
            );

          // Iterating over each batch
          for(size_t b = 0; b < batch_size ; ++b){
              // Iterating over each sequence position
              for(size_t s = 0; s < seq_len ; ++s){
                  // Computing mean across each embedding dimension
                  double mean = 0.0;
                  for(size_t e = 0; e < embedding_dim; ++e){
                    mean += input[b][s][e];
                  }
                  mean = mean/ embedding_dim;

                  // Computing variance across each embedding dimension
                  double variance = 0.0;
                  for(size_t e = 0; e < embedding_dim; ++e){
                      variance += (input[b][s][e] - mean) * (input[b][s][e]-mean);
                  }
                  variance /= embedding_dim;

                  double stddev = sqrt(variance + epsilon);

                  // Normalize and apply gamma and beta
                  for(size_t e =0; e < embedding_dim; ++e){
                      output[b][s][e] = gamma[e] * ((input[b][s][e] - mean)/ stddev) + beta[e];
                  }
              }
          }
      return output;
      }
};

int main() {
    //(batch_size = 2, seq_length = 3, embedding_dim = 4)
    std::vector<std::vector<std::vector<double>>> input = {
        { {0.2, 0.1, 0.3, 0.4}, {0.5, 0.1, 0.1, 0.3}, {0.5, 0.4, 0.2, 0.6} },
        { {0.6, 0.7, 0.5, 0.4}, {0.1, 0.2, 0.3, 0.4}, {0.3, 0.7, 0.9, 0.1} }
    };

    // Initialize LayerNorm with embedding_dim = 4
    LayerNorm layer_norm(4);

    std::vector<std::vector<std::vector<double>>> output = layer_norm.forward(input);

    // Print output
    std::cout << "Layer Normalized Output:\n";
    for (const auto& batch : output) {
        for (const auto& seq : batch) {
            for (double val : seq) {
                std::cout << val << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    return 0;
}

