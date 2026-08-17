/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     8/16/2026
 */

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include "../include/polymerMelt.hpp"
#include "../include/primitivePath.hpp"

using namespace Eigen;

using std::pow; 
using std::ceil; 

int main(int argc, char** argv)
{
    // Parse input arguments 
    std::string config_filename = argv[1];
    std::string path_filename = argv[2];
    std::string outfilename = argv[3]; 

    // Parse the input polymer configurations and primitive paths  
    auto result = parseMeltConfigFile<double>(config_filename);
    auto melt_configs = result.first; 
    std::cout << "... parsed " << melt_configs.size() << " configurations from: "
              << config_filename << std::endl;  
    auto params = result.second; 
    const double xmin = params["domain_xmin"]; 
    const double xmax = params["domain_xmax"]; 
    const double ymin = params["domain_ymin"]; 
    const double ymax = params["domain_ymax"]; 
    const double zmin = params["domain_zmin"]; 
    const double zmax = params["domain_zmax"]; 
    const double xlen = xmax - xmin; 
    const double ylen = ymax - ymin; 
    const double zlen = zmax - zmin;
    const double volume = xlen * ylen * zlen;
    const int n_configs = melt_configs.size();
    const int n_chains = melt_configs[0].numChains(); 
    const int length = melt_configs[0].getLength(0);
    const int n_total = n_chains * length; 

    // Parse primitive paths 
    auto paths = parseZ1SPFile<double>(path_filename);
    std::cout << "... parsed " << paths.size() << " configurations from: "
              << path_filename << std::endl; 

    // Use a bin width that is a small multiple of sigma 
    const double dr = 0.1 * 1.8 / 0.965; 

    // Define the RDF bins from 0 to (xmax - xmin) / 2, assuming that the 
    // periodic box is a cube 
    const int n_bins = static_cast<int>(ceil((xlen / 2) / dr));
    Matrix<double, Dynamic, 1> rdf_bins(n_bins + 1); 
    for (int i = 0; i < n_bins + 1; ++i)
        rdf_bins(i) = i * dr;

    // Get the volume of each shell 
    Array<double, Dynamic, 1> shells(n_bins);
    for (int i = 0; i < n_bins; ++i)
    {
        double dr3 = pow(rdf_bins(i + 1), 3) - pow(rdf_bins(i), 3);  
        shells(i) = boost::math::constants::four_thirds_pi<double>() * dr3;
    }

    // Calculate the intra-chain and inter-chain RDF for each configuration 
    Array<double, Dynamic, Dynamic> inter_rdf = Array<double, Dynamic, Dynamic>::Zero(n_configs, n_bins); 

    // For each chain in each configuration ... 
    for (int i = 0; i < n_configs; ++i)
    {
        int n_nodes = 0; 

        for (int j = 0; j < n_chains; ++j)
        {
            // Get the contour indices of the nodes along the corresponding 
            // primitive path  
            Matrix<int, Dynamic, 1> nearest = paths[i].getNearestContourIndices(j);
            n_nodes += (nearest.size() - 2);  

            // For each node ... 
            for (int k = 1; k < nearest.size() - 1; ++k)
            {
                Matrix<double, 3, 1> p = melt_configs[i].getSegment(j, nearest(k), 1).transpose(); 

                // Get the distance to every bead along every other chain 
                for (int m = 0; m < n_chains; ++m)
                {
                    Matrix<double, Dynamic, 3> coords = melt_configs[i].getSegment(m, 0, length); 
                    if (m != j)
                    {
                        for (int s = 0; s < length; ++s)
                        {
                            Matrix<double, 3, 1> q = coords.row(s); 
                            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                                p, q, xlen, ylen, zlen
                            );
                            double dist = dvec.norm();
                            if (dist < rdf_bins(n_bins))
                            {
                                for (int t = 0; t < n_bins; ++t)
                                {
                                    if (dist >= rdf_bins(t) && dist < rdf_bins(t + 1))
                                    {
                                        inter_rdf(i, t) += 1; 
                                        break; 
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Normalize the RDF
        inter_rdf.row(i) *= (
            volume * shells.pow(-1).transpose() / (n_nodes * n_total)
        ); 
    }

    // Write to output file 
    std::ofstream outfile(outfilename);
    outfile << std::setprecision(10); 
    outfile << "# n_chains = " << n_chains << std::endl
            << "# length = " << length << std::endl
            << "# volume = " << volume << std::endl; 
    outfile << "# BIN_EDGES\t";      // Write bin edges in the first line 
    for (int i = 0; i < n_bins; ++i)
        outfile << rdf_bins(i) << '\t';
    outfile << rdf_bins(n_bins) << std::endl;
    for (int i = 0; i < n_configs; ++i)    // Write a separate RDF for each configuration
    {
        for (int j = 0; j < n_bins - 1; ++j)
        {
            outfile << inter_rdf(i, j) << '\t'; 
        }
        outfile << inter_rdf(i, n_bins - 1) << std::endl; 
    }
    outfile.close(); 
    
    return 0; 
}
