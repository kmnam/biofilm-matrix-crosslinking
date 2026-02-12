/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     2/10/2026
 */

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include "../include/utils.hpp"
#include "../include/polymerConfiguration.hpp"
#include "../include/cbmc.hpp"

int main(int argc, char** argv)
{
    // Parse input json file 
    std::string json_filename = argv[1];
    boost::json::object json_data = parseConfigFile(json_filename).as_object();

    // Define potential and sampling parameters 
    Matrix<double, 3, 1> r0 = Matrix<double, 3, 1>::Zero();
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
    const double collision_threshold = json_data["init_collision_threshold"].as_double(); 
    const int max_tries_per_atom = json_data["init_max_tries_per_atom"].as_int64();
    const int max_n_backtracks = json_data["init_max_n_backtracks"].as_int64();
    const int n_candidates = json_data["n_candidates"].as_int64(); 
    const int n_target = json_data["n_target_configs"].as_int64(); 
    const int n_burnin = json_data["n_burnin"].as_int64(); 
    const int mod_collect = json_data["collect_config_every_iter"].as_int64();
    const int mod_write = json_data["write_configs_every_iter"].as_int64();  
    const int max_iter = n_burnin + (n_target - 1) * mod_collect;
    const int max_stall = json_data["max_stall_iter"].as_int64();

    // Fix move probabilities 
    Matrix<double, 3, 1> move_probs; 
    move_probs << json_data["reptation_prob"].as_double(), 
                  json_data["terminal_segment_move_prob"].as_double(), 
                  json_data["internal_segment_move_prob"].as_double();  

    // Parse terminal and internal segment lengths
    const int terminal_segment_length = json_data["terminal_segment_length"].as_int64(); 
    const int internal_segment_length = json_data["internal_segment_length"].as_int64();

    // Parse internal move parameters (with default values as follows)
    InternalMoveGenerationMode mode;  
    try
    {
        internal_move_params["tangent_stepsize"] = json_data["tangent_stepsize"].as_double();
    }
    catch (boost::wrapexcept<boost::system::system_error>& e)
    {
        internal_move_params["tangent_stepsize"] = 0.1 * lj_params["sigma"]; 
    }
    try
    {
        internal_move_params["mode"] = json_data["internal_move_generation_mode"].as_int64();
    }
    catch (boost::wrapexcept<boost::system::system_error>& e)
    {
        internal_move_params["mode"] = 0; 
    }
    mode = static_cast<InternalMoveGenerationMode>(internal_move_params["mode"]); 
    std::cout << internal_move_params["mode"] << std::endl;   
    try
    {
        internal_move_params["n_attempts"] = json_data["internal_move_n_attempts"].as_double(); 
    }
    catch (boost::wrapexcept<boost::system::system_error>& e)
    {
        if (mode == InternalMoveGenerationMode::FIXED_ATTEMPTS)
            internal_move_params["n_attempts"] = n_candidates; 
        else 
            internal_move_params["n_attempts"] = 2 * n_candidates; 
    }
    std::cout << internal_move_params["n_attempts"] << std::endl; 
    try
    { 
        internal_move_params["dx"] = json_data["dx"].as_double(); 
    }
    catch (boost::wrapexcept<boost::system::system_error>& e)
    {
        internal_move_params["dx"] = 1e-8; 
    }
    try
    {
        internal_move_params["newton_tol"] = json_data["newton_tol"].as_double(); 
    }
    catch (boost::wrapexcept<boost::system::system_error>& e)
    {
        internal_move_params["newton_tol"] = 1e-5;
    }
    try
    {
        internal_move_params["min_newton_stepsize"]
            = json_data["min_newton_stepsize"].as_double(); 
    }
    catch (boost::wrapexcept<boost::system::system_error>& e)
    {
        internal_move_params["min_newton_stepsize"] = 1e-4; 
    }
    try
    {
        internal_move_params["armijo_const"] = json_data["armijo_const"].as_double(); 
    }
    catch (boost::wrapexcept<boost::system::system_error>& e)
    {
        internal_move_params["armijo_const"] = 1e-4;
    }

    // Initialize random number generator 
    const int seed = std::stoi(argv[3]);
    boost::random::mt19937 rng(seed); 
    boost::random::uniform_01<> uniform_dist;  

    // Generate an initial configuration
    PolymerConfiguration<double> config = generateKMer<double>(
        length, lj_params, fene_params, angle_mode, angle_params,
        dihedral_params, r0, collision_threshold, max_tries_per_atom, 
        max_n_backtracks, rng, uniform_dist, Units::NANO, 300
    );

    // Initialize output file 
    std::string outfilename = argv[2]; 
    std::ofstream outfile(outfilename);

    // Initialize CBMC sampler 
    const double neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"]; 
    PolymerCBMCSampler<double> sampler(
        config, lj_params, neighbor_threshold, fene_params, angle_mode, 
        angle_params, dihedral_params, rng
    );

    // Determine if a prior run is to be continued
    Matrix<double, Dynamic, Dynamic> ensemble_coords; 
    if (argc == 6 && strcmp(argv[4], "-c") == 0)
    {
        std::string prev_filename = argv[5];
        ensemble_coords = sampler.run(prev_filename, n_target - 1, outfile, true);  
    }
    else    // Otherwise, start a new run
    {
        ensemble_coords = sampler.run(
            n_candidates, internal_move_params, move_probs, terminal_segment_length,
            internal_segment_length, max_iter, n_burnin, mod_collect, mod_write,
            max_stall, outfile, true
        );
    } 
    const int n_ensemble = ensemble_coords.rows();  

    // Calculate the persistence length and write to file
    PolymerEnsemble<double> ensemble; 
    for (int i = 0; i < n_ensemble; ++i)
    {
        Matrix<double, Dynamic, 3> coords_i(length, 3); 
        for (int j = 0; j < length; ++j)
            coords_i.row(j) = ensemble_coords(i, Eigen::seqN(3 * j, 3)); 
        PolymerConfiguration<double> config(coords_i, Units::NANO, 300);
        ensemble.push_back(config);  
    } 
    double persist_length = getPersistenceLength<double>(ensemble);
    outfile << "## PERSISTENCE_LENGTH\t" << persist_length << std::endl;  

    return 0; 
}
