/**
 * Evaluate the tangent vector autocorrelation function for the given 
 * ensemble of polymer configurations.  
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     5/25/2026
 */

#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "../include/polymerConfiguration.hpp"
#include "../include/polymerEnsemble.hpp"

int main(int argc, char** argv)
{
    // Parse input filename 
    std::string filename = argv[1]; 

    // Parse the polymer configurations in the file  
    auto result = parseEnsemble<double>(filename, Units::NANO, 300);
    PolymerEnsemble<double> ensemble = result.first;

    // Check that there are at least two configurations 
    if (ensemble.size() < 2)
        throw std::runtime_error(
            "Invalid ensemble size for persistence length calculation"
        ); 

    // Get the tangent vectors along each configuration in the ensemble
    const int n = ensemble.size();  
    std::vector<Matrix<double, Dynamic, 3> > tangent_vectors; 
    for (int i = 0; i < n; ++i)
        tangent_vectors.push_back(ensemble[i].tangentVectors());

    // Get the polymer length (assuming that it is constant over the 
    // ensemble)
    const int length = ensemble[0].getLength(); 

    // Get the mean bond length over the ensemble 
    double mean_bond_length = 0; 
    for (int i = 0; i < n; ++i)
    {
        double mean_i = ensemble[i].meanBondLength(); 
        mean_bond_length += ((mean_i - mean_bond_length) / (i + 1));
    }

    // Evaluate the autocorrelation function
    Matrix<double, Dynamic, 1> autocorrs(length - 2); 
    for (int k = 1; k < length - 1; ++k)
        autocorrs(k - 1) = getTangentVectorAutocorrelation<double>(tangent_vectors, k).mean(); 

    // Write each value to an output file 
    std::string outfilename = argv[2];
    std::ofstream outfile(outfilename);
    outfile << std::setprecision(10)
            << "# MEAN_BOND_LENGTH\t" << mean_bond_length << std::endl; 
    for (int k = 1; k < length - 1; ++k)
        outfile << k << '\t' << autocorrs(k - 1) << std::endl;
    outfile.close();  

    return 0; 
}
