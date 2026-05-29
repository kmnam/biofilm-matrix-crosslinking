/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     4/24/2026
 */

#include <fstream>
#include "../include/polymerMelt.hpp"
#include "../include/primitivePath.hpp"

int main(int argc, char** argv)
{
    // Parse input arguments 
    std::string config_filename = argv[1];
    std::string path_filename = argv[2];
    std::string outprefix = argv[3]; 

    // Parse the input polymer configurations and primitive paths  
    auto result = parseMeltConfigFile<double>(config_filename);
    std::vector<PolymerMeltConfiguration<double> > melt_configs = std::get<0>(result); 
    std::cout << "... parsed " << melt_configs.size() << " configurations from: "
              << config_filename << std::endl;  
    std::vector<PrimitivePathCollection<double> > paths = parseZ1SPFile<double>(path_filename);
    std::cout << "... parsed " << paths.size() << " configurations from: "
              << path_filename << std::endl; 
    const int n_configs = melt_configs.size();
    const int n_chains = melt_configs[0].numChains();
    const double angle_threshold = 0.5 * (
        boost::math::constants::half_pi<double>() +
        160.0 * boost::math::constants::pi<double>() / 180.0
    ); 

    // Open output files 
    std::stringstream ss; 
    ss << outprefix << "_kink_indices.txt"; 
    std::string kink_filename = ss.str(); 
    ss.str(std::string()); 
    ss.clear(); 
    ss << outprefix << "_node_indices.txt"; 
    std::string node_filename = ss.str(); 
    ss.str(std::string()); 
    ss.clear(); 
    ss << outprefix << "_node_kink_dists.txt"; 
    std::string dist_filename = ss.str();  
    std::ofstream outfile1(kink_filename), outfile2(node_filename), outfile3(dist_filename);

    // For each chain in each configuration ...  
    for (int i = 0; i < n_configs; ++i)
    {
        for (int j = 0; j < n_chains; ++j)
        {
            // Get the contour indices of the nodes along the corresponding 
            // primitive path  
            Matrix<int, Dynamic, 1> nearest = paths[i].getNearestContourIndices(j);

            // Get the bond angles along the chain contour 
            Matrix<double, Dynamic, 1> angles = melt_configs[i].bondAngles(j);

            // Identify which bond angles are small
            int n_below_threshold = 0;
            Matrix<int, Dynamic, 1> below_threshold(0); 
            for (int k = 0; k < angles.size(); ++k)
            {
                if (angles(k) < angle_threshold)
                {
                    n_below_threshold++; 
                    below_threshold.conservativeResize(n_below_threshold); 
                    below_threshold(n_below_threshold - 1) = k;
                } 
            }

            // Write the kink indices to file
            if (n_below_threshold > 0)
            { 
                outfile1 << i << '\t' << j << '\t';
                for (int k = 0; k < n_below_threshold - 1; ++k)
                    outfile1 << below_threshold(k) << '\t';
                outfile1 << below_threshold(n_below_threshold - 1) << std::endl;
            }
            else 
            {
                outfile1 << i << '\t' << j << std::endl; 
            }

            // Write the node indices to file
            if (nearest.size() > 0)
            { 
                outfile2 << i << '\t' << j << '\t';
                for (int k = 0; k < nearest.size() - 1; ++k)
                    outfile2 << nearest(k) << '\t'; 
                outfile2 << nearest(nearest.size() - 1) << std::endl; 
            }
            else 
            {
                outfile2 << i << '\t' << j << std::endl; 
            }

            // For each node along the primitive path ...
            if (nearest.size() > 0 && n_below_threshold > 0)
            {
                Matrix<int, Dynamic, 1> dists_to_nearest_kink(nearest.size()); 
                for (int k = 0; k < nearest.size(); ++k)
                {
                    // Get the index of the kink nearest to the node's nearest bead
                    // along the chain contour
                    int nearest_below_threshold_idx = nearestValue<int>(
                        below_threshold, nearest(k)
                    );
                    dists_to_nearest_kink(k) = abs(
                        nearest(k) - below_threshold(nearest_below_threshold_idx)
                    );
                }

                // Write the node-kink distances to file
                outfile3 << i << '\t' << j << '\t';
                for (int k = 0; k < nearest.size() - 1; ++k)
                    outfile3 << dists_to_nearest_kink(k) << '\t';
                outfile3 << dists_to_nearest_kink(nearest.size() - 1) << std::endl;
            }
            else 
            {
                outfile3 << i << '\t' << j << std::endl; 
            }
        }
    }

    return 0; 
}
