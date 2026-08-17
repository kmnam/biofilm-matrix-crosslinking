/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     8/14/2026
 */

#include <iostream>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include "../include/polymerMelt.hpp"

using namespace Eigen;

using std::pow; 
using std::ceil; 

int main(int argc, char** argv)
{
    // Parse input filename
    std::string filename = argv[1];

    // Parse the polymer melt configurations in the file
    auto parse_result = parseMeltConfigFile<double>(filename);
    auto melt_configs = parse_result.first;
    auto params = parse_result.second; 
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

    // Use a bin width that is a small multiple of sigma 
    const double dr = 0.1 * 1.8 / 0.965; 

    // Define the RDF bins from 0 to (xmax - xmin) / 2, assuming that the 
    // periodic box is a cube 
    const int n_bins = static_cast<int>(ceil((xlen / 2) / dr));
    Matrix<double, Dynamic, 1> rdf_bins(n_bins + 1); 
    for (int i = 0; i < n_bins + 1; ++i)
        rdf_bins(i) = i * dr;

    // Calculate the intra-chain and inter-chain RDF for each configuration 
    Array<double, Dynamic, Dynamic> intra_rdf = Array<double, Dynamic, Dynamic>::Zero(n_configs, n_bins); 
    Array<double, Dynamic, Dynamic> inter_rdf = Array<double, Dynamic, Dynamic>::Zero(n_configs, n_bins);  
    
    // For each configuration ...
    for (int i = 0; i < n_configs; ++i)
    {   
        // For each chain ... 
        for (int j = 0; j < n_chains; ++j)
        {
            // For each bead along the chain ... 
            for (int k = 0; k < length; ++k)
            {
                // Get the distance to each other bead along the chain 
                for (int m = k + 1; m < length; ++m)
                {
                    Matrix<double, 3, 1> p = melt_configs[i].getSegment(j, k, 1).transpose(); 
                    Matrix<double, 3, 1> q = melt_configs[i].getSegment(j, m, 1).transpose();  
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        p, q, xlen, ylen, zlen
                    );
                    double dist = dvec.norm();
                    if (dist < rdf_bins(n_bins))
                    {
                        for (int s = 0; s < n_bins; ++s)
                        {
                            if (dist >= rdf_bins(s) && dist < rdf_bins(s + 1))
                            {
                                intra_rdf(i, s) += 1;
                                break; 
                            }
                        }
                    }
                }
            }
        }
        std::cout << "... processed intra-chain distances for config " << i << std::endl;  

        // For each pair of chains ... 
        for (int j = 0; j < n_chains; ++j)
        {
            for (int k = j + 1; k < n_chains; ++k)
            {
                // For each bead along chain 1 and each bead along chain 2 ... 
                for (int m1 = 0; m1 < length; ++m1)
                {
                    for (int m2 = 0; m2 < length; ++m2)
                    {
                        // Get the distance between the two beads 
                        Matrix<double, 3, 1> p = melt_configs[i].getSegment(j, m1, 1).transpose(); 
                        Matrix<double, 3, 1> q = melt_configs[i].getSegment(k, m2, 1).transpose();  
                        Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                            p, q, xlen, ylen, zlen
                        );
                        double dist = dvec.norm();
                        if (dist < rdf_bins(n_bins))
                        {
                            for (int s = 0; s < n_bins; ++s)
                            {
                                if (dist >= rdf_bins(s) && dist < rdf_bins(s + 1))
                                {
                                    inter_rdf(i, s) += 1; 
                                    break; 
                                }
                            }
                        }
                    }
                }
            }
        }
        std::cout << "... processed inter-chain distances for config " << i << std::endl; 
    }

    // Get the volume of each shell 
    Array<double, Dynamic, 1> shells(n_bins);
    for (int i = 0; i < n_bins; ++i)
    {
        double dr3 = pow(rdf_bins(i + 1), 3) - pow(rdf_bins(i), 3);  
        shells(i) = boost::math::constants::four_thirds_pi<double>() * dr3;
    }

    // Normalize the two RDFs 
    const int n_total = n_chains * length; 
    intra_rdf.rowwise() *= (
        2 * volume * shells.pow(-1).transpose() / (n_total * n_total)
    ); 
    inter_rdf.rowwise() *= (
        2 * volume * shells.pow(-1).transpose() / (n_total * n_total)
    ); 

    // Write to output file ... 
    std::stringstream ss_intra, ss_inter; 
    ss_intra << argv[2] << "_intra.txt"; 
    ss_inter << argv[2] << "_inter.txt";

    // Write the intra-chain RDF file 
    std::string outfilename1 = ss_intra.str(); 
    std::ofstream outfile1(outfilename1);
    outfile1 << std::setprecision(10); 
    outfile1 << "# n_chains = " << n_chains << std::endl
             << "# length = " << length << std::endl
             << "# volume = " << volume << std::endl; 
    outfile1 << "# BIN_EDGES\t";      // Write bin edges in the first line 
    for (int i = 0; i < n_bins; ++i)
        outfile1 << rdf_bins(i) << '\t';
    outfile1 << rdf_bins(n_bins) << std::endl;
    for (int i = 0; i < n_configs; ++i)    // Write a separate RDF for each configuration
    {
        for (int j = 0; j < n_bins - 1; ++j)
        {
            outfile1 << intra_rdf(i, j) << '\t'; 
        }
        outfile1 << intra_rdf(i, n_bins - 1) << std::endl; 
    }
    outfile1.close(); 

    // Write the inter-chain RDF file 
    std::string outfilename2 = ss_inter.str(); 
    std::ofstream outfile2(outfilename2);
    outfile2 << std::setprecision(10); 
    outfile2 << "# n_chains = " << n_chains << std::endl
             << "# length = " << length << std::endl
             << "# volume = " << volume << std::endl; 
    outfile2 << "# BIN_EDGES\t";      // Write bin edges in the first line 
    for (int i = 0; i < n_bins; ++i)
        outfile2 << rdf_bins(i) << '\t';
    outfile2 << rdf_bins(n_bins) << std::endl;
    for (int i = 0; i < n_configs; ++i)    // Write a separate RDF for each configuration
    {
        for (int j = 0; j < n_bins - 1; ++j)
        {
            outfile2 << inter_rdf(i, j) << '\t'; 
        }
        outfile2 << inter_rdf(i, n_bins - 1) << std::endl; 
    }
    outfile2.close();

    return 0; 
}
