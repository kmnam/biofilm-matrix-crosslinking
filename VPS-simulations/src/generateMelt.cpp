/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     4/6/2026
 */

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include "../include/utils.hpp"
#include "../include/polymerConfiguration.hpp"
#include "../include/polymerEnsemble.hpp"
#include "../include/polymerMelt.hpp"

int main(int argc, char** argv)
{
    // Parse input json file 
    std::string json_filename = argv[1];
    boost::json::object json_data = parseConfigFile(json_filename).as_object();

    // Define potential and sampling parameters 
    std::unordered_map<std::string, double> lj_params, 
                                            fene_params, 
                                            angle_params, 
                                            dihedral_params,
                                            internal_move_params;
    const double kT = 1.380649e-2 * 300;    // Use "nano" units 
    const int length = json_data["length"].as_int64(); 
    lj_params["eps"] = json_data["lj_eps"].as_double() * kT; 
    lj_params["sigma"] = json_data["lj_sigma"].as_double(); 
    fene_params["K"] = json_data["fene_K"].as_double() * kT;
    fene_params["R0"] = json_data["fene_R0"].as_double();
    AngleMode angle_mode = (
        json_data["angle_mode"] == 0 ? AngleMode::COSINE : AngleMode::GAUSSIAN
    ); 
    if (angle_mode == AngleMode::COSINE)
    { 
        angle_params["K"] = json_data["cosine_K"].as_double() * kT; 
        angle_params["theta0"] = (
            json_data["cosine_theta0"].as_double() * boost::math::constants::pi<double>() / 180
        );
    }
    else 
    {
        angle_params["A1"] = json_data["gaussian_A1"].as_double(); 
        angle_params["A2"] = json_data["gaussian_A2"].as_double(); 
        angle_params["w1"] = json_data["gaussian_w1"].as_double(); 
        angle_params["w2"] = json_data["gaussian_w2"].as_double(); 
        angle_params["theta1"] = (
            json_data["gaussian_theta1"].as_double() * boost::math::constants::pi<double>() / 180
        ); 
        angle_params["theta2"] = (
            json_data["gaussian_theta2"].as_double() * boost::math::constants::pi<double>() / 180
        );
    } 
    dihedral_params["K"] = json_data["dihedral_K"].as_double() * kT;
    dihedral_params["d"] = 1; 
    dihedral_params["n"] = 1;
    const double intra_collision_threshold = json_data["init_intra_collision_threshold"].as_double();
    const double inter_collision_threshold = json_data["init_inter_collision_threshold"].as_double();  
    const int max_tries_per_atom = json_data["init_max_tries_per_atom"].as_int64();
    const int max_tries_per_kmer = json_data["init_max_tries_per_kmer"].as_int64();
    const int max_tries_per_seed = json_data["init_max_tries_per_seed"].as_int64();  
    const int max_n_backtracks = json_data["init_max_n_backtracks"].as_int64();
    const int max_n_restarts = json_data["init_max_n_restarts"].as_int64(); 
    const int n_bins = json_data["n_bins_fene_cdf"].as_int64(); 
    const double xmax = json_data["domain_xmax"].as_double(); 
    const double ymax = json_data["domain_ymax"].as_double(); 
    const double zmax = json_data["domain_zmax"].as_double();
    const double mass = json_data["monomer_mass"].as_double();  

    // Initialize random number generator 
    const int seed = std::stoi(argv[3]);
    boost::random::mt19937 rng(seed); 
    boost::random::uniform_01<> uniform_dist; 

    // Pre-compute FENE bond length CDF 
    Matrix<double, Dynamic, 2> bond_length_cdf = getFeneCDF<double>(
        lj_params["eps"], lj_params["sigma"], fene_params["K"], fene_params["R0"],
        kT, n_bins
    ); 

    // Generate an initial configuration
    const int n_chains = json_data["n_chains"].as_int64();
    std::cout << "Generating " << n_chains << " chains of length "
              << length << " ...\n";  
    PolymerMeltConfiguration<double> melt_config = generateKMerMelt<double>(
        length, n_chains, lj_params, fene_params, angle_mode, angle_params,
        dihedral_params, intra_collision_threshold, inter_collision_threshold,
        max_tries_per_atom, max_tries_per_kmer, max_tries_per_seed,
        max_n_backtracks, max_n_restarts, rng, uniform_dist, xmax, ymax, zmax,
        bond_length_cdf, Units::NANO, 300, true
    );

    // Write the melt configuration to file
    std::stringstream ss; 
    ss << "Melt of " << n_chains << " polymers of length " << length; 
    std::string header = ss.str(); 
    melt_config.writeLammps(
        argv[2], lj_params, fene_params, angle_mode, angle_params,
        dihedral_params, header, mass
    ); 

    return 0; 
}
