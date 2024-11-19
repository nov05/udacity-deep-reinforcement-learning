#include "scalar_multiply_improved.hpp"
using namespace std;

vector< vector<int> > scalar_multiply_improved(vector< vector<int> > matrix, int scalar) {
    
  	// OPTIMIZATION: Instead of creating a new variable
    // called resultmatrix, update the input matrix directly
    vector< vector<int> > resultmatrix;
    vector<int> new_row;
    
    vector<int>::size_type num_rows = matrix.size();
    vector<int>::size_type num_cols = matrix[0].size();
    
    for (int i = 0; i < num_rows; i++) {
        new_row.clear();
        for (int j = 0; j < num_cols; j++) {
            new_row.push_back(matrix[i][j] * scalar);
        }
        resultmatrix.push_back(new_row);
    }
    
    return resultmatrix;
}
