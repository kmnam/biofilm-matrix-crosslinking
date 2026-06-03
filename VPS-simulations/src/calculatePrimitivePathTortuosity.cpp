/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     5/31/2026
 */

#include <fstream>
#include "../include/polymerMelt.hpp"
#include "../include/primitivePath.hpp"

int main(int argc, char** argv)
{
    std::string config_filename = argv[1]; 
    std::string path_filename = argv[2]; 

    auto result = parseMeltConfigFile<double>(config_filename, Units::NANO, 300.0);
    std::vector<PolymerMeltConfiguration<double> > melt_configs = result.first;
    std::unordered_map<std::string, double> params = result.second; 
    const int n_configs = melt_configs.size(); 
    const int n_chains = melt_configs[0].numChains(); 
    const double xmin = params["domain_xmin"]; 
    const double xmax = params["domain_xmax"];
    const double ymin = params["domain_ymin"];
    const double ymax = params["domain_ymax"];
    const double zmin = params["domain_zmin"];
    const double zmax = params["domain_zmax"]; 
    const double xlen = xmax - xmin;
    const double ylen = ymax - ymin; 
    const double zlen = zmax - zmin;  
    Matrix<double, Dynamic, Dynamic> Re(n_configs, n_chains); 
    for (int i = 0; i < n_configs; ++i)
    {
        for (int j = 0; j < n_chains; ++j)
        {
            const int nj = melt_configs[i].getLength(j); 
            Matrix<double, Dynamic, 3> coords = melt_configs[i].getSegment(j, 0, nj); 
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                coords.row(0), coords.row(nj - 1), xlen, ylen, zlen
            );
            Re(i, j) = dvec.norm(); 
        } 
    }

    Matrix<double, Dynamic, Dynamic> Lpp = Matrix<double, Dynamic, Dynamic>::Zero(n_configs, n_chains); 
    std::vector<PrimitivePathCollection<double> > paths = parseZ1SPFile<double>(path_filename);
    for (int i = 0; i < n_configs; ++i)
    {
        for (int j = 0; j < n_chains; ++j)
        {
            Matrix<double, Dynamic, 3> node_coords = paths[i].getNodeCoords(j);
            for (int k = 0; k < node_coords.rows() - 1; ++k)
            { 
                Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                    node_coords.row(k), node_coords.row(k + 1), xlen, ylen, zlen
                );  
                Lpp(i, j) += dvec.norm();
            }
        }
    }

    Matrix<double, Dynamic, Dynamic> Lpp_norm = (Lpp.array() / Re.array()).matrix();
    std::cout << Lpp_norm.mean() << std::endl;  

    return 0; 
}
