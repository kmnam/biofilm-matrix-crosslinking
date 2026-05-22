/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     5/21/2026
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
    // Parse input arguments
    std::string json_filename = argv[1];
    std::string outprefix = argv[2]; 
    const int seed = std::stoi(argv[3]);

    // Parse input json file 
    boost::json::object json_data = parseConfigFile(json_filename).as_object();

    // Define potential and sampling parameters ...  
    std::unordered_map<std::string, double> lj_params, 
                                            fene_params, 
                                            angle_params, 
                                            dihedral_params;
    const double kT = 1.380649e-2 * 300;    // Use "nano" units 
    const int length = json_data["length"].as_int64();
    const int n_chains = json_data["n_chains"].as_int64(); 

    // Parse Lennard-Jones and FENE potential parameters 
    lj_params["eps"] = json_data["lj_eps"].as_double() * kT; 
    lj_params["sigma"] = json_data["lj_sigma"].as_double(); 
    fene_params["K"] = json_data["fene_K"].as_double() * kT;
    fene_params["R0"] = json_data["fene_R0"].as_double();

    // Parse angle potential parameters 
    AngleMode angle_mode = (
        json_data["angle_mode"].as_int64() == 0 ? AngleMode::COSINE : AngleMode::GAUSSIAN
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

    // Parse dihedral potential parameters  
    dihedral_params["K"] = json_data["dihedral_K"].as_double() * kT;
    dihedral_params["d"] = 1; 
    dihedral_params["n"] = 1;
    try
    {
        dihedral_params["delta"] = (
            json_data["dihedral_delta"].as_double() * boost::math::constants::pi<double>() / 180
        ); 
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }

    // Parse domain limits (xmin = -xmax, ymin = -ymax, zmin = -zmax)
    const double xmax = json_data["domain_xmax"].as_double(); 
    const double ymax = json_data["domain_ymax"].as_double(); 
    const double zmax = json_data["domain_zmax"].as_double();

    // Parse additional initialization parameters
    const double intra_collision_threshold = json_data["init_intra_collision_threshold"].as_double();
    const double inter_collision_threshold = json_data["init_inter_collision_threshold"].as_double(); 
    const int max_tries_per_atom = json_data["init_max_tries_per_atom"].as_int64();
    const int max_tries_per_kmer = json_data["init_max_tries_per_kmer"].as_int64(); 
    const int max_tries_per_seed = json_data["init_max_tries_per_seed"].as_int64(); 
    const int max_n_backtracks = json_data["init_max_n_backtracks"].as_int64();
    const int max_n_restarts = json_data["init_max_n_restarts"].as_int64(); 
    const int n_bins = json_data["n_bins_fene_cdf"].as_int64();

    // Initialize random number generator 
    boost::random::mt19937 rng(seed); 
    boost::random::uniform_01<> uniform_dist;

    // Pre-compute FENE bond length CDF 
    Matrix<double, Dynamic, 2> bond_length_cdf = getFeneCDF<double>(
        lj_params["eps"], lj_params["sigma"], fene_params["K"], fene_params["R0"],
        kT, n_bins
    );

    // Generate an initial melt configuration, regardless of whether a previous
    // run is to be continued
    std::cout << "Generating " << n_chains << " chains of length "
              << length << " ...\n";  
    PolymerMeltConfiguration<double> melt_config = generateKMerMelt<double>(
        length, n_chains, lj_params, fene_params, angle_mode, angle_params,
        dihedral_params, intra_collision_threshold, inter_collision_threshold,
        max_tries_per_atom, max_tries_per_kmer, max_tries_per_seed,
        max_n_backtracks, max_n_restarts, rng, uniform_dist, xmax, ymax, zmax,
        bond_length_cdf, Units::NANO, 300, true
    );

    // Parse additional input parameters to pass into LAMMPS
    const int ncores = std::stoi(argv[4]);  
    const double mass = json_data["monomer_mass"].as_double();
    const double dt = json_data["dt"].as_double(); 
    const double damp = json_data["damp"].as_double();
    const double t_final_soft = json_data["t_final_soft"].as_double();
    const double t_final_preeq = json_data["t_final_preeq"].as_double(); 
    const double dt_per_dump = json_data["dt_per_dump"].as_double();  

    // Write the melt configuration to a LAMMPS initial data file
    std::stringstream ss; 
    ss << "Melt of " << n_chains << " polymers of length " << length; 
    std::string header = ss.str();
    ss.str(std::string()); 
    ss.clear(); 
    ss << outprefix << "_init.data"; 
    std::string init_filename = ss.str();  
    melt_config.writeLammps(
        init_filename, lj_params, fene_params, angle_mode, angle_params,
        dihedral_params, header, mass
    ); 

    // Run LAMMPS with a soft potential to relax the configuration
    //
    // Prepare an input file for LAMMPS 
    ss.str(std::string()); 
    ss.clear(); 
    ss << outprefix << "_soft_paths.txt"; 
    std::string soft_input_filename = ss.str(); 
    std::ofstream outfile(soft_input_filename);
    boost::random::uniform_int_distribution<> lammps_seed_dist(1, 999); 
    outfile << init_filename << std::endl
            << outprefix << "_soft.lammpstrj" << std::endl
            << outprefix << "_resolved.data" << std::endl
            << outprefix << "_lj_coeffs.data" << std::endl
            << dt << std::endl
            << damp << std::endl
            << static_cast<int>(t_final_soft / dt) << std::endl
            << static_cast<int>((t_final_preeq - t_final_soft) / dt) << std::endl
            << static_cast<int>(dt_per_dump / dt) << std::endl
            << lammps_seed_dist(rng) << std::endl
            << lammps_seed_dist(rng) << std::endl;
    outfile.close();

    // Run LAMMPS
    //
    // There are three possible modes:
    // 1) No angle or dihedral potential (random coils with excluded volume
    //    interactions 
    // 2) Gaussian angle potential and harmonic (or trivial) dihedral potential
    // 3) Gaussian angle potential and Fourier dihedral potential 
    std::string lammps_script_filename;
    if (angle_mode == AngleMode::COSINE && angle_params["K"] == 0)
    {
        lammps_script_filename = "random_coil_soft.lammps";
    } 
    else if (angle_mode == AngleMode::GAUSSIAN)
    {
        if (dihedral_params.find("delta") == dihedral_params.end())
            lammps_script_filename = "bimodal_angles_gaussian_soft.lammps";
        else 
            lammps_script_filename = "bimodal_angles_gaussian_fourier_soft.lammps"; 
    }
    else
    {
        throw std::runtime_error(
            "Invalid combination of angle and dihedral potentials"
        );
    } 
    ss.str(std::string()); 
    ss.clear();
    ss << "mpirun -np " << ncores << " lmp -i " << lammps_script_filename
       << " -v VARS " << soft_input_filename;
    std::string cmd = ss.str(); 
    std::system(cmd.c_str());

    return 0; 
}
