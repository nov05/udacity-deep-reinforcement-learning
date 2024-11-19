#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

class Matrix 
{
    private:

        std::vector< std::vector<float> > grid;
        std::vector<float>::size_type rows;
        std::vector<float>::size_type cols;
        
    public:
        
        // constructor function declarations
        Matrix ();
        Matrix (std::vector< std::vector<float> >);
    
        // set and get function declarations
        void setGrid(std::vector< std::vector<float> >);
    
        std::vector< std::vector<float> > getGrid();
        std::vector<float>::size_type getRows();
        std::vector<float>::size_type getCols();

        // matrix function declarations
        Matrix matrix_transpose();
        Matrix matrix_addition(Matrix);
        void matrix_print();  
};

#endif