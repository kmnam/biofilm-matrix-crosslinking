/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     5/20/2026
 */

#include <iostream>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/random.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/utils.hpp"
#include "../../include/polymerConfiguration.hpp"
#include "../../include/cbmc.hpp"

using namespace Eigen;

using std::min; 

/**
 * Tests for PolymerCBMCSampler::generateReptationMoves(). 
 */
TEST_CASE("Tests for reptation move generation", "[generateReptationMoves()]")
{
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;
    const double tol = 1e-8;  

    // Set up potential and sampling parameters
    Matrix<double, 3, 1> r0 = Matrix<double, 3, 1>::Zero();
    std::unordered_map<std::string, double> lj_params, 
                                            fene_params,
                                            random_params,  
                                            cosine_params,
                                            gaussian_params,
                                            nodihedral_params,
                                            dihedral_params;
    double kT = 1.380649e-2 * 300;
    lj_params["eps"] = kT; 
    lj_params["sigma"] = 0.9;
    fene_params["K"] = 30 * kT; 
    fene_params["R0"] = 1.5;
    random_params["K"] = 0; 
    random_params["theta0"] = boost::math::constants::pi<double>();
    cosine_params["K"] = 20 * kT;
    cosine_params["theta0"] = 160 * boost::math::constants::pi<double>() / 180;
    gaussian_params["A1"] = 0.9; 
    gaussian_params["A2"] = 0.1;
    gaussian_params["w1"] = 0.2236067977;    // = 1/sqrt(20) 
    gaussian_params["w2"] = 0.2236067977; 
    gaussian_params["theta1"] = 160 * boost::math::constants::pi<double>() / 180; 
    gaussian_params["theta2"] = 90 * boost::math::constants::pi<double>() / 180;
    nodihedral_params["K"] = 0;  
    dihedral_params["K"] = 0.5 * kT;
    const double collision_threshold = 0.1;
    const int max_tries_per_atom = 50;
    const int max_n_backtracks = 50;  
    Matrix<double, Dynamic, 2> bond_length_cdf = getFeneCDF<double>(
        lj_params["eps"], lj_params["sigma"], fene_params["K"], fene_params["R0"],
        kT, 10000
    ); 

    // --------------------------------------------------------------- //
    // Reptation moves on 10-mer with a cosine angle potential and no 
    // dihedral potential 
    // --------------------------------------------------------------- //
    const int length = 10; 
    PolymerConfiguration<double> config = generateKMer<double>(
        length, lj_params, fene_params, AngleMode::COSINE, cosine_params,
        nodihedral_params, r0, collision_threshold, max_tries_per_atom,
        max_n_backtracks, rng, uniform_dist, bond_length_cdf
    );
    Matrix<double, Dynamic, 3> coords = config.getSegment(0, length);  
    REQUIRE(config.getLength() == length);
    REQUIRE(coords.rows() == length);

    // Initialize sampler instance
    double neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"]; 
    PolymerCBMCSampler<double> sampler_cosine(
        config, lj_params, neighbor_threshold, fene_params, AngleMode::COSINE,
        cosine_params, nodihedral_params, rng
    );  

    // Try generating 50 reptation moves at the head
    const int n_candidates = 50;  
    auto result = sampler_cosine.generateReptationMoves(
        ReptationDirection::HEAD, n_candidates
    );
    Matrix<double, Dynamic, Dynamic> r_new = result.first; 
    Matrix<double, Dynamic, 1> energy_diffs = result.second;
    REQUIRE(r_new.rows() == n_candidates);
    REQUIRE(r_new.cols() == 3);  
    REQUIRE(energy_diffs.size() == n_candidates); 

    // Check that each new atom has a valid distance to the 0-th atom
    for (int i = 0; i < n_candidates; ++i) 
        REQUIRE((r_new.row(i) - coords.row(0)).norm() < fene_params["R0"]); 

    // Check the reptation non-bonded energy difference
    double nonbonded_energy_orig = config.getNonbondedEnergy(lj_params, neighbor_threshold, true); 
    for (int i = 0; i < n_candidates; ++i)
    {
        PolymerConfiguration<double> config_reptated(config);
        config_reptated.reptateTowardsHead(r_new.row(i)); 
        double nonbonded_energy_new = config_reptated.getNonbondedEnergy(
            lj_params, neighbor_threshold, true
        ); 
        REQUIRE_THAT(
            energy_diffs(i),
            Catch::Matchers::WithinAbs(nonbonded_energy_new - nonbonded_energy_orig, tol)
        ); 
    }

    // Try generating 50 reptation moves at the tail 
    result = sampler_cosine.generateReptationMoves(
        ReptationDirection::TAIL, n_candidates
    );
    r_new = result.first; 
    energy_diffs = result.second;
    REQUIRE(r_new.rows() == n_candidates);
    REQUIRE(r_new.cols() == 3);  
    REQUIRE(energy_diffs.size() == n_candidates); 

    // Check that the new atom has a valid distance to the 0-th atom
    int tail_idx = length - 1; 
    for (int i = 0; i < n_candidates; ++i)
        REQUIRE((r_new.row(i) - coords.row(tail_idx)).norm() < fene_params["R0"]); 

    // Check the reptation non-bonded energy difference
    for (int i = 0; i < n_candidates; ++i)
    {
        PolymerConfiguration<double> config_reptated(config);
        config_reptated.reptateTowardsTail(r_new.row(i)); 
        double nonbonded_energy_new = config_reptated.getNonbondedEnergy(
            lj_params, neighbor_threshold, true
        ); 
        REQUIRE_THAT(
            energy_diffs(i),
            Catch::Matchers::WithinAbs(nonbonded_energy_new - nonbonded_energy_orig, tol)
        ); 
    }

    // --------------------------------------------------------------- //
    // Reptation moves on 10-mer with a cosine angle potential and a  
    // harmonic dihedral potential 
    // --------------------------------------------------------------- //
    config = generateKMer<double>(
        length, lj_params, fene_params, AngleMode::COSINE, cosine_params,
        dihedral_params, r0, collision_threshold, max_tries_per_atom,
        max_n_backtracks, rng, uniform_dist, bond_length_cdf
    );
    coords = config.getSegment(0, length);  
    REQUIRE(config.getLength() == length);
    REQUIRE(coords.rows() == length);

    // Initialize sampler instance
    PolymerCBMCSampler<double> sampler_cosine_dihedral(
        config, lj_params, neighbor_threshold, fene_params, AngleMode::COSINE,
        cosine_params, dihedral_params, rng
    );  

    // Try generating 50 reptation moves at the head
    result = sampler_cosine_dihedral.generateReptationMoves(
        ReptationDirection::HEAD, n_candidates
    );
    r_new = result.first; 
    energy_diffs = result.second;
    REQUIRE(r_new.rows() == n_candidates);
    REQUIRE(r_new.cols() == 3);  
    REQUIRE(energy_diffs.size() == n_candidates); 

    // Check that each new atom has a valid distance to the 0-th atom
    for (int i = 0; i < n_candidates; ++i) 
        REQUIRE((r_new.row(i) - coords.row(0)).norm() < fene_params["R0"]); 

    // Check the reptation non-bonded energy difference
    nonbonded_energy_orig = config.getNonbondedEnergy(lj_params, neighbor_threshold, true); 
    for (int i = 0; i < n_candidates; ++i)
    {
        PolymerConfiguration<double> config_reptated(config);
        config_reptated.reptateTowardsHead(r_new.row(i)); 
        double nonbonded_energy_new = config_reptated.getNonbondedEnergy(
            lj_params, neighbor_threshold, true
        ); 
        REQUIRE_THAT(
            energy_diffs(i),
            Catch::Matchers::WithinAbs(nonbonded_energy_new - nonbonded_energy_orig, tol)
        ); 
    }

    // Try generating 50 reptation moves at the tail 
    result = sampler_cosine_dihedral.generateReptationMoves(
        ReptationDirection::TAIL, n_candidates
    );
    r_new = result.first; 
    energy_diffs = result.second;
    REQUIRE(r_new.rows() == n_candidates);
    REQUIRE(r_new.cols() == 3);  
    REQUIRE(energy_diffs.size() == n_candidates); 

    // Check that the new atom has a valid distance to the 0-th atom
    tail_idx = length - 1; 
    for (int i = 0; i < n_candidates; ++i)
        REQUIRE((r_new.row(i) - coords.row(tail_idx)).norm() < fene_params["R0"]); 

    // Check the reptation non-bonded energy difference
    for (int i = 0; i < n_candidates; ++i)
    {
        PolymerConfiguration<double> config_reptated(config);
        config_reptated.reptateTowardsTail(r_new.row(i)); 
        double nonbonded_energy_new = config_reptated.getNonbondedEnergy(
            lj_params, neighbor_threshold, true
        ); 
        REQUIRE_THAT(
            energy_diffs(i),
            Catch::Matchers::WithinAbs(nonbonded_energy_new - nonbonded_energy_orig, tol)
        ); 
    }

    // --------------------------------------------------------------- //
    // Reptation moves on 10-mer with a Gaussian angle potential and a 
    // harmonic dihedral potential 
    // --------------------------------------------------------------- //
    config = generateKMer<double>(
        length, lj_params, fene_params, AngleMode::GAUSSIAN, gaussian_params,
        dihedral_params, r0, collision_threshold, max_tries_per_atom,
        max_n_backtracks, rng, uniform_dist, bond_length_cdf
    );
    coords = config.getSegment(0, length);  
    REQUIRE(config.getLength() == length);
    REQUIRE(coords.rows() == length); 

    // Initialize sampler instance
    PolymerCBMCSampler<double> sampler_gaussian(
        config, lj_params, neighbor_threshold, fene_params, AngleMode::GAUSSIAN,
        gaussian_params, dihedral_params, rng
    );  

    // Try generating 50 reptation moves at the head 
    result = sampler_gaussian.generateReptationMoves(
        ReptationDirection::HEAD, n_candidates
    );
    r_new = result.first; 
    energy_diffs = result.second;
    REQUIRE(r_new.rows() == n_candidates);
    REQUIRE(r_new.cols() == 3); 
    REQUIRE(energy_diffs.size() == n_candidates); 

    // Check that the new atom has a valid distance to the 0-th atom
    for (int i = 0; i < n_candidates; ++i)
        REQUIRE((r_new.row(i) - coords.row(0)).norm() < fene_params["R0"]); 

    // Check the reptation non-bonded energy difference
    nonbonded_energy_orig = config.getNonbondedEnergy(lj_params, neighbor_threshold, true); 
    for (int i = 0; i < n_candidates; ++i)
    {
        PolymerConfiguration<double> config_reptated(config);
        config_reptated.reptateTowardsHead(r_new.row(i)); 
        double nonbonded_energy_new = config_reptated.getNonbondedEnergy(
            lj_params, neighbor_threshold, true
        ); 
        REQUIRE_THAT(
            energy_diffs(i),
            Catch::Matchers::WithinAbs(nonbonded_energy_new - nonbonded_energy_orig, tol)
        ); 
    }

    // Try generating 50 reptation moves at the tail
    result = sampler_gaussian.generateReptationMoves(
        ReptationDirection::TAIL, n_candidates
    );
    r_new = result.first; 
    energy_diffs = result.second;
    REQUIRE(r_new.rows() == n_candidates);
    REQUIRE(r_new.cols() == 3);  
    REQUIRE(energy_diffs.size() == n_candidates); 

    // Check that the new atom has a valid distance to the last atom
    tail_idx = length - 1;  
    for (int i = 0; i < n_candidates; ++i)
        REQUIRE((r_new.row(i) - coords.row(tail_idx)).norm() < fene_params["R0"]); 

    // Check the reptation non-bonded energy difference
    for (int i = 0; i < n_candidates; ++i)
    {
        PolymerConfiguration<double> config_reptated(config);
        config_reptated.reptateTowardsHead(r_new.row(i)); 
        double nonbonded_energy_new = config_reptated.getNonbondedEnergy(
            lj_params, neighbor_threshold, true
        ); 
        REQUIRE_THAT(
            energy_diffs(i),
            Catch::Matchers::WithinAbs(nonbonded_energy_new - nonbonded_energy_orig, tol)
        ); 
    }
}

/**
 * Tests for reptation in the PolymerCBMCSampler class via moveOnce(). 
 */
TEST_CASE("Tests for reptation", "[moveOnce()]")
{
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;
    const double tol = 1e-8;  

    // Set up potential and sampling parameters
    Matrix<double, 3, 1> r0 = Matrix<double, 3, 1>::Zero();
    std::unordered_map<std::string, double> lj_params, 
                                            fene_params, 
                                            random_params, 
                                            cosine_params,
                                            gaussian_params,
                                            nodihedral_params, 
                                            dihedral_params;
    double kT = 1.380649e-2 * 300;
    lj_params["eps"] = kT; 
    lj_params["sigma"] = 0.9;
    fene_params["K"] = 9 * kT; 
    fene_params["R0"] = 1.5;

    // Define a null cosine potential (to mimic random coils with excluded 
    // volume interactions)
    random_params["K"] = 0.0; 
    random_params["theta0"] = boost::math::constants::pi<double>();

    // Make cosine potential soft to allow for reptation candidates that are
    // close to the original configuration  
    cosine_params["K"] = 0.5 * kT;
    cosine_params["theta0"] = 160 * boost::math::constants::pi<double>() / 180;

    // Similarly make Gaussian potential soft to allow for reptation candidates
    // that are close to the original configuration 
    gaussian_params["A1"] = 0.9; 
    gaussian_params["A2"] = 0.1;
    gaussian_params["w1"] = 2.0;     // Standard deviations of 1 for each component
    gaussian_params["w2"] = 2.0;
    gaussian_params["theta1"] = 160 * boost::math::constants::pi<double>() / 180; 
    gaussian_params["theta2"] = 90 * boost::math::constants::pi<double>() / 180;

    // Define dihedral potential parameters 
    nodihedral_params["K"] = 0;  
    dihedral_params["K"] = 0.5 * kT;

    // Define additional CBMC parameters 
    const double collision_threshold = 0.9;    // Slightly less than 2^(1/6) * sigma ~ 1.01 
    const int max_tries_per_atom = 50;
    const int max_n_backtracks = 50;  
    Matrix<double, Dynamic, 2> bond_length_cdf = getFeneCDF<double>(
        lj_params["eps"], lj_params["sigma"], fene_params["K"], fene_params["R0"],
        kT, 10000
    );

    // --------------------------------------------------------------- //
    // Reptation moves on 10-mer with no angle or dihedral potentials 
    // --------------------------------------------------------------- //
    const int length = 10; 
    PolymerConfiguration<double> config = generateKMer<double>(
        length, lj_params, fene_params, AngleMode::COSINE, random_params, 
        nodihedral_params, r0, collision_threshold, max_tries_per_atom, 
        max_n_backtracks, rng, uniform_dist, bond_length_cdf
    );
    Matrix<double, Dynamic, 3> coords = config.getSegment(0, length);  
    REQUIRE(config.getLength() == length);
    REQUIRE(coords.rows() == length); 

    // Initialize sampler instance
    double neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"]; 
    PolymerCBMCSampler<double> sampler_random(
        config, lj_params, neighbor_threshold, fene_params, AngleMode::COSINE,
        random_params, nodihedral_params, rng
    ); 

    // Try reptating by choosing from 10000 reptation candidate moves ... 
    //
    // First generate the moves  
    const int n_candidates = 10000; 
    PolymerConfiguration<double> config_reptated(config);  
    auto result = sampler_random.moveOnce(n_candidates, CBMCMoveType::REPTATION, 0);
    Matrix<double, Dynamic, Dynamic> forward_moves = std::get<0>(result); 
    Matrix<double, Dynamic, Dynamic> reverse_moves = std::get<1>(result); 
    int move_idx = std::get<2>(result); 
    double prob_accept = std::get<3>(result); 
    CBMCMoveResult accepted_move = std::get<4>(result);
    ReptationDirection direction = static_cast<ReptationDirection>(
        std::get<5>(result).at("direction")
    );

    // Check that there are the correct number of forward/reverse moves 
    REQUIRE(forward_moves.rows() == n_candidates);
    REQUIRE(forward_moves.cols() == 3);  
    REQUIRE(reverse_moves.rows() == n_candidates);
    REQUIRE(reverse_moves.cols() == 3);
    REQUIRE((move_idx >= 0 && move_idx < n_candidates));
    REQUIRE((prob_accept >= 0 && prob_accept <= 1));

    // Generate the reptated configuration 
    if (direction == ReptationDirection::HEAD)
        config_reptated.reptateTowardsHead(forward_moves.row(move_idx)); 
    else 
        config_reptated.reptateTowardsTail(forward_moves.row(move_idx));

    // Check that each new atom has a valid distance to the terminal atom 
    // at the appropriate end 
    for (int i = 0; i < n_candidates; ++i)
    {
        if (direction == ReptationDirection::HEAD)
        { 
            int forward_idx = 0;    // Forward move should involve new atom bonded to 0
            int reverse_idx = 8;    // Reverse move should involve new atom bonded to 8
            REQUIRE((forward_moves.row(i) - coords.row(forward_idx)).norm() < fene_params["R0"]);
            REQUIRE((reverse_moves.row(i) - coords.row(reverse_idx)).norm() < fene_params["R0"]); 
        }
        else 
        {
            int forward_idx = 9;    // Forward move should involve new atom bonded to 9
            int reverse_idx = 1;    // Reverse move should involve new atom bonded to 1
            REQUIRE((forward_moves.row(i) - coords.row(forward_idx)).norm() < fene_params["R0"]);
            REQUIRE((reverse_moves.row(i) - coords.row(reverse_idx)).norm() < fene_params["R0"]); 
        }
    }

    // Check that the 0-th reverse move is reversion to the original configuration
    if (direction == ReptationDirection::HEAD)
    {
        // The new atom should match the 9-th atom in the original configuration 
        REQUIRE_THAT(
            (reverse_moves.row(0) - coords.row(9)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    } 
    else 
    {
        // The new atom should match the 0-th atom in the original configuration
        REQUIRE_THAT(
            (reverse_moves.row(0) - coords.row(0)).norm(), 
            Catch::Matchers::WithinAbs(0, tol)
        );
    }

    // Re-calculate the Boltzmann weights of the individual moves from scratch ...
    ReptationDirection reverse_direction = (
        direction == ReptationDirection::HEAD ? ReptationDirection::TAIL
        : ReptationDirection::HEAD
    ); 
    Matrix<double, Dynamic, 1> weights_forward(n_candidates),
                               weights_reverse(n_candidates);

    // For each forward move ...  
    for (int i = 0; i < n_candidates; ++i)
    {
        // Generate the reptated configuration from the original configuration 
        PolymerConfiguration<double> config2(config);
        if (direction == ReptationDirection::HEAD)
            config2.reptateTowardsHead(forward_moves.row(i)); 
        else 
            config2.reptateTowardsTail(forward_moves.row(i));

        // Calculate the non-bonded energy difference between the two 
        // configurations  
        double diff = (
            config2.getNonbondedEnergy(lj_params, neighbor_threshold, true) - 
            config.getNonbondedEnergy(lj_params, neighbor_threshold, true)
        );

        // Calculate the corresponding Boltzmann weight  
        weights_forward(i) = exp(-diff / kT);
    }

    // For each reverse move ... 
    for (int i = 0; i < n_candidates; ++i)
    {
        // Generate a new reptated configuration from the *reptated*
        // configuration, in the reverse direction 
        PolymerConfiguration<double> config2(config_reptated); 
        if (direction == ReptationDirection::HEAD)
            config2.reptateTowardsTail(reverse_moves.row(i)); 
        else 
            config2.reptateTowardsHead(reverse_moves.row(i));

        // Calculate the non-bonded energy difference between the two 
        // configurations  
        double diff = (
            config2.getNonbondedEnergy(lj_params, neighbor_threshold, true) - 
            config_reptated.getNonbondedEnergy(lj_params, neighbor_threshold, true)
        );

        // Calculate the corresponding Boltzmann weight 
        weights_reverse(i) = exp(-diff / kT);
    }

    // Calculate the Rosenbluth factors
    double forward_rosenbluth = weights_forward.sum();
    double reverse_rosenbluth = weights_reverse.sum();

    // Check that the ratio of Rosenbluth factors is equal to the acceptance 
    // probability (if it is <= 1)
    std::cout << "- Forward Rosenbluth (no angle potential, no dihedral potential): "
              << forward_rosenbluth << std::endl; 
    std::cout << "- Reverse Rosenbluth (no angle potential, no dihedral potential): "
              << reverse_rosenbluth << std::endl; 
    std::cout << "- Acceptance probability: " << prob_accept << std::endl;  
    REQUIRE_THAT(
        min(forward_rosenbluth / reverse_rosenbluth, 1.0),
        Catch::Matchers::WithinAbs(prob_accept, tol)
    );

    // Check that, if the acceptance probability is 1, the chosen move was taken 
    if (prob_accept == 1)
        REQUIRE(accepted_move == CBMCMoveResult::ACCEPT); 

    // Check that, if the chosen move was taken, the resulting configuration is
    // as expected
    Matrix<double, Dynamic, 3> coords_reptated = config_reptated.getSegment(0, length); 
    Matrix<double, Dynamic, 3> coords_result = sampler_random.getCoords();  
    if (accepted_move == CBMCMoveResult::ACCEPT)
    {
        // Move was taken 
        for (int i = 0; i < length; ++i)
        {
            REQUIRE_THAT(
                (coords_result.row(i) - coords_reptated.row(i)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 
    }
    else 
    {
        // Move was not taken 
        for (int i = 0; i < length; ++i)
        {
            REQUIRE_THAT(
                (coords_result.row(i) - coords.row(i)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 
    } 

    // --------------------------------------------------------------- //
    // Reptation moves on 10-mer with a cosine angle potential and no 
    // dihedral potential 
    // --------------------------------------------------------------- //
    config = generateKMer<double>(
        length, lj_params, fene_params, AngleMode::COSINE, cosine_params,
        nodihedral_params, r0, collision_threshold, max_tries_per_atom, 
        max_n_backtracks, rng, uniform_dist, bond_length_cdf
    );
    coords = config.getSegment(0, length);  
    REQUIRE(config.getLength() == length);
    REQUIRE(coords.rows() == length); 

    // Initialize sampler instance
    PolymerCBMCSampler<double> sampler_cosine(
        config, lj_params, neighbor_threshold, fene_params, AngleMode::COSINE,
        cosine_params, nodihedral_params, rng
    ); 

    // Try reptating by choosing from 10000 reptation candidate moves ... 
    //
    // First generate the moves  
    config_reptated = config;
    result = sampler_cosine.moveOnce(n_candidates, CBMCMoveType::REPTATION, 0);
    forward_moves = std::get<0>(result); 
    reverse_moves = std::get<1>(result); 
    move_idx = std::get<2>(result); 
    prob_accept = std::get<3>(result); 
    accepted_move = std::get<4>(result);
    direction = static_cast<ReptationDirection>(std::get<5>(result).at("direction"));

    // Check that there are the correct number of forward/reverse moves 
    REQUIRE(forward_moves.rows() == n_candidates);
    REQUIRE(forward_moves.cols() == 3);  
    REQUIRE(reverse_moves.rows() == n_candidates);
    REQUIRE(reverse_moves.cols() == 3);
    REQUIRE((move_idx >= 0 && move_idx < n_candidates));
    REQUIRE((prob_accept >= 0 && prob_accept <= 1));

    // Generate the reptated configuration 
    if (direction == ReptationDirection::HEAD)
        config_reptated.reptateTowardsHead(forward_moves.row(move_idx)); 
    else 
        config_reptated.reptateTowardsTail(forward_moves.row(move_idx));

    // Check that each new atom has a valid distance to the terminal atom 
    // at the appropriate end 
    for (int i = 0; i < n_candidates; ++i)
    {
        if (direction == ReptationDirection::HEAD)
        { 
            int forward_idx = 0;    // Forward move should involve new atom bonded to 0
            int reverse_idx = 8;    // Reverse move should involve new atom bonded to 8
            REQUIRE((forward_moves.row(i) - coords.row(forward_idx)).norm() < fene_params["R0"]);
            REQUIRE((reverse_moves.row(i) - coords.row(reverse_idx)).norm() < fene_params["R0"]); 
        }
        else 
        {
            int forward_idx = 9;    // Forward move should involve new atom bonded to 9
            int reverse_idx = 1;    // Reverse move should involve new atom bonded to 1
            REQUIRE((forward_moves.row(i) - coords.row(forward_idx)).norm() < fene_params["R0"]);
            REQUIRE((reverse_moves.row(i) - coords.row(reverse_idx)).norm() < fene_params["R0"]); 
        }
    }

    // Check that the 0-th reverse move is reversion to the original configuration
    if (direction == ReptationDirection::HEAD)
    {
        // The new atom should match the 9-th atom in the original configuration 
        REQUIRE_THAT(
            (reverse_moves.row(0) - coords.row(9)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    } 
    else 
    {
        // The new atom should match the 0-th atom in the original configuration
        REQUIRE_THAT(
            (reverse_moves.row(0) - coords.row(0)).norm(), 
            Catch::Matchers::WithinAbs(0, tol)
        );
    }

    // Re-calculate the Boltzmann weights of the individual moves from scratch ...
    reverse_direction = (
        direction == ReptationDirection::HEAD ? ReptationDirection::TAIL
        : ReptationDirection::HEAD
    ); 
    weights_forward = Matrix<double, Dynamic, 1>::Zero(n_candidates); 
    weights_reverse = Matrix<double, Dynamic, 1>::Zero(n_candidates); 

    // For each forward move ...  
    for (int i = 0; i < n_candidates; ++i)
    {
        // Generate the reptated configuration from the original configuration 
        PolymerConfiguration<double> config2(config);
        if (direction == ReptationDirection::HEAD)
            config2.reptateTowardsHead(forward_moves.row(i)); 
        else 
            config2.reptateTowardsTail(forward_moves.row(i));

        // Calculate the non-bonded energy difference between the two 
        // configurations  
        double diff = (
            config2.getNonbondedEnergy(lj_params, neighbor_threshold, true) - 
            config.getNonbondedEnergy(lj_params, neighbor_threshold, true)
        );

        // Calculate the corresponding Boltzmann weight  
        weights_forward(i) = exp(-diff / kT);
    }

    // For each reverse move ... 
    for (int i = 0; i < n_candidates; ++i)
    {
        // Generate a new reptated configuration from the *reptated*
        // configuration, in the reverse direction 
        PolymerConfiguration<double> config2(config_reptated); 
        if (direction == ReptationDirection::HEAD)
            config2.reptateTowardsTail(reverse_moves.row(i)); 
        else 
            config2.reptateTowardsHead(reverse_moves.row(i));

        // Calculate the non-bonded energy difference between the two 
        // configurations  
        double diff = (
            config2.getNonbondedEnergy(lj_params, neighbor_threshold, true) - 
            config_reptated.getNonbondedEnergy(lj_params, neighbor_threshold, true)
        );

        // Calculate the corresponding Boltzmann weight 
        weights_reverse(i) = exp(-diff / kT);
    }

    // Calculate the Rosenbluth factors
    forward_rosenbluth = weights_forward.sum();
    reverse_rosenbluth = weights_reverse.sum();

    // Check that the ratio of Rosenbluth factors is equal to the acceptance 
    // probability (if it is <= 1)
    std::cout << "- Forward Rosenbluth (cosine angle potential, no dihedral potential): "
              << forward_rosenbluth << std::endl; 
    std::cout << "- Reverse Rosenbluth (cosine angle potential, no dihedral potential): "
              << reverse_rosenbluth << std::endl; 
    std::cout << "- Acceptance probability: " << prob_accept << std::endl;  
    REQUIRE_THAT(
        min(forward_rosenbluth / reverse_rosenbluth, 1.0),
        Catch::Matchers::WithinAbs(prob_accept, tol)
    );

    // Check that, if the acceptance probability is 1, the chosen move was taken 
    if (prob_accept == 1)
        REQUIRE(accepted_move == CBMCMoveResult::ACCEPT); 

    // Check that, if the chosen move was taken, the resulting configuration is
    // as expected
    coords_reptated = config_reptated.getSegment(0, length); 
    coords_result = sampler_cosine.getCoords();  
    if (accepted_move == CBMCMoveResult::ACCEPT)
    {
        // Move was taken 
        for (int i = 0; i < length; ++i)
        {
            REQUIRE_THAT(
                (coords_result.row(i) - coords_reptated.row(i)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 
    }
    else 
    {
        // Move was not taken 
        for (int i = 0; i < length; ++i)
        {
            REQUIRE_THAT(
                (coords_result.row(i) - coords.row(i)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 
    } 

    // --------------------------------------------------------------- //
    // Reptation moves on 10-mer with a cosine angle potential and a 
    // harmonic dihedral potential 
    // --------------------------------------------------------------- //
    config = generateKMer<double>(
        length, lj_params, fene_params, AngleMode::COSINE, cosine_params,
        dihedral_params, r0, collision_threshold, max_tries_per_atom,
        max_n_backtracks, rng, uniform_dist, bond_length_cdf
    );
    coords = config.getSegment(0, length);  
    REQUIRE(config.getLength() == length);
    REQUIRE(coords.rows() == length); 

    // Initialize sampler instance
    PolymerCBMCSampler<double> sampler_cosine_dihedral(
        config, lj_params, neighbor_threshold, fene_params, AngleMode::COSINE,
        cosine_params, dihedral_params, rng
    );  

    // Try reptating by choosing from 10000 reptation candidate moves ...
    //
    // First generate the moves  
    config_reptated = config; 
    result = sampler_cosine_dihedral.moveOnce(n_candidates, CBMCMoveType::REPTATION, 0);
    forward_moves = std::get<0>(result); 
    reverse_moves = std::get<1>(result); 
    move_idx = std::get<2>(result); 
    prob_accept = std::get<3>(result); 
    accepted_move = std::get<4>(result);
    direction = static_cast<ReptationDirection>(std::get<5>(result).at("direction"));

    // Check that there are the correct number of forward/reverse moves 
    REQUIRE(forward_moves.rows() == n_candidates);
    REQUIRE(forward_moves.cols() == 3);  
    REQUIRE(reverse_moves.rows() == n_candidates);
    REQUIRE(reverse_moves.cols() == 3);
    REQUIRE((move_idx >= 0 && move_idx < n_candidates));
    REQUIRE((prob_accept >= 0 && prob_accept <= 1));

    // Generate the reptated configuration
    if (direction == ReptationDirection::HEAD)
        config_reptated.reptateTowardsHead(forward_moves.row(move_idx)); 
    else 
        config_reptated.reptateTowardsTail(forward_moves.row(move_idx));

    // Check that each new atom has a valid distance to the terminal atom 
    // at the appropriate end 
    for (int i = 0; i < n_candidates; ++i)
    {
        if (direction == ReptationDirection::HEAD)
        {
            int forward_idx = 0;    // Forward move should involve new atom bonded to 0
            int reverse_idx = 8;    // Reverse move should involve new atom bonded to 8
            REQUIRE((forward_moves.row(i) - coords.row(forward_idx)).norm() < fene_params["R0"]);
            REQUIRE((reverse_moves.row(i) - coords.row(reverse_idx)).norm() < fene_params["R0"]); 
        }
        else 
        {
            int forward_idx = 9;    // Forward move should involve new atom bonded to 9
            int reverse_idx = 1;    // Reverse move should involve new atom bonded to 1
            REQUIRE((forward_moves.row(i) - coords.row(forward_idx)).norm() < fene_params["R0"]);
            REQUIRE((reverse_moves.row(i) - coords.row(reverse_idx)).norm() < fene_params["R0"]); 
        }
    }

    // Check that the 0-th reverse move is reversion to the original configuration
    if (direction == ReptationDirection::HEAD)
    {
        // The new atom should match the 9-th atom in the original configuration 
        REQUIRE_THAT(
            (reverse_moves.row(0) - coords.row(9)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    }
    else
    { 
        // The new atom should match the 0-th atom in the original configuration 
        REQUIRE_THAT(
            (reverse_moves.row(0) - coords.row(0)).norm(), 
            Catch::Matchers::WithinAbs(0, tol)
        );
    }

    // Re-calculate the Boltzmann weights of the individual moves from scratch ...
    reverse_direction = (
        direction == ReptationDirection::HEAD ? ReptationDirection::TAIL
        : ReptationDirection::HEAD
    ); 
    weights_forward = Matrix<double, Dynamic, 1>::Zero(n_candidates); 
    weights_reverse = Matrix<double, Dynamic, 1>::Zero(n_candidates); 

    // For each forward move ... 
    for (int i = 0; i < n_candidates; ++i)
    {
        // Generate the reptated configuration from the original configuration 
        PolymerConfiguration<double> config2(config); 
        if (direction == ReptationDirection::HEAD)
            config2.reptateTowardsHead(forward_moves.row(i)); 
        else 
            config2.reptateTowardsTail(forward_moves.row(i)); 
        
        // Calculate the non-bonded energy difference between the two 
        // configurations 
        double diff = (
            config2.getNonbondedEnergy(lj_params, neighbor_threshold, true) - 
            config.getNonbondedEnergy(lj_params, neighbor_threshold, true)
        ); 

        // Calculate the corresponding Boltzmann weight 
        weights_forward(i) = exp(-diff / kT);
    }

    // For each reverse move ... 
    for (int i = 0; i < n_candidates; ++i)
    {
        // Generate a new reptated configuration from the *reptated* 
        // configuration, in the reverse direction  
        PolymerConfiguration<double> config2(config_reptated); 
        if (direction == ReptationDirection::HEAD)
            config2.reptateTowardsTail(reverse_moves.row(i)); 
        else 
            config2.reptateTowardsHead(reverse_moves.row(i));

        // Calculate the non-bonded energy difference between the two 
        // configurations  
        double diff = (
            config2.getNonbondedEnergy(lj_params, neighbor_threshold, true) - 
            config_reptated.getNonbondedEnergy(lj_params, neighbor_threshold, true)
        );

        // Calculate the corresponding Boltzmann weight
        weights_reverse(i) = exp(-diff / kT);
    }

    // Calculate the Rosenbluth factors 
    forward_rosenbluth = weights_forward.sum(); 
    reverse_rosenbluth = weights_reverse.sum();

    // Check that the ratio of Rosenbluth factors is equal to the acceptance 
    // probability (if it is <= 1)
    std::cout << "- Forward Rosenbluth (cosine angle potential, harmonic dihedral potential): "
              << forward_rosenbluth << std::endl; 
    std::cout << "- Reverse Rosenbluth (cosine angle potential, harmonic dihedral potential): "
              << reverse_rosenbluth << std::endl; 
    std::cout << "- Acceptance probability: " << prob_accept << std::endl;  
    REQUIRE_THAT(
        min(forward_rosenbluth / reverse_rosenbluth, 1.0),
        Catch::Matchers::WithinAbs(prob_accept, tol)
    );

    // Check that, if the acceptance probability is 1, the chosen move was taken 
    if (prob_accept == 1)
        REQUIRE(accepted_move == CBMCMoveResult::ACCEPT); 

    // Check that, if the chosen move was taken, the resulting configuration is
    // as expected
    coords_reptated = config_reptated.getSegment(0, length); 
    coords_result = sampler_cosine_dihedral.getCoords();  
    if (accepted_move == CBMCMoveResult::ACCEPT)
    {
        // Move was taken 
        for (int i = 0; i < length; ++i)
        {
            REQUIRE_THAT(
                (coords_result.row(i) - coords_reptated.row(i)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 
    }
    else 
    {
        // Move was not taken 
        for (int i = 0; i < length; ++i)
        {
            REQUIRE_THAT(
                (coords_result.row(i) - coords.row(i)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 
    } 

    // ------------------------------------------------------------------ // 
    // Reptation moves on 10-mer with a Gaussian angle potential and a 
    // harmonic dihedral potential
    // ------------------------------------------------------------------ //
    config = generateKMer<double>(
        length, lj_params, fene_params, AngleMode::GAUSSIAN, gaussian_params,
        dihedral_params, r0, collision_threshold, max_tries_per_atom,
        max_n_backtracks, rng, uniform_dist, bond_length_cdf
    );
    coords = config.getSegment(0, length);  
    REQUIRE(config.getLength() == length);
    REQUIRE(coords.rows() == length);

    // Initialize sampler instance
    PolymerCBMCSampler<double> sampler_gaussian(
        config, lj_params, neighbor_threshold, fene_params, AngleMode::GAUSSIAN,
        gaussian_params, dihedral_params, rng
    );  

    // Try reptating by choosing from 10000 reptation candidate moves ...
    //
    // First generate the moves  
    config_reptated = config; 
    result = sampler_gaussian.moveOnce(n_candidates, CBMCMoveType::REPTATION, 0);
    forward_moves = std::get<0>(result); 
    reverse_moves = std::get<1>(result); 
    move_idx = std::get<2>(result); 
    prob_accept = std::get<3>(result); 
    accepted_move = std::get<4>(result);
    direction = static_cast<ReptationDirection>(std::get<5>(result).at("direction"));

    // Check that there are the correct number of forward/reverse moves 
    REQUIRE(forward_moves.rows() == n_candidates);
    REQUIRE(forward_moves.cols() == 3);  
    REQUIRE(reverse_moves.rows() == n_candidates);
    REQUIRE(reverse_moves.cols() == 3); 
    REQUIRE((move_idx >= 0 && move_idx < n_candidates));
    REQUIRE((prob_accept >= 0 && prob_accept <= 1)); 

    // Generate the reptated configuration 
    if (direction == ReptationDirection::HEAD)
        config_reptated.reptateTowardsHead(forward_moves.row(move_idx)); 
    else 
        config_reptated.reptateTowardsTail(forward_moves.row(move_idx));

    // Check that each new atom has a valid distance to the terminal atom 
    // at the appropriate end 
    for (int i = 0; i < n_candidates; ++i)
    {
        if (direction == ReptationDirection::HEAD)
        {
            int forward_idx = 0;    // Forward move should involve new atom bonded to 0
            int reverse_idx = 8;    // Reverse move should involve new atom bonded to 8
            REQUIRE((forward_moves.row(i) - coords.row(forward_idx)).norm() < fene_params["R0"]);
            REQUIRE((reverse_moves.row(i) - coords.row(reverse_idx)).norm() < fene_params["R0"]); 
        }
        else 
        {
            int forward_idx = 9;    // Forward move should involve new atom bonded to 9
            int reverse_idx = 1;    // Reverse move should involve new atom bonded to 1
            REQUIRE((forward_moves.row(i) - coords.row(forward_idx)).norm() < fene_params["R0"]);
            REQUIRE((reverse_moves.row(i) - coords.row(reverse_idx)).norm() < fene_params["R0"]); 
        }
    }

    // Check that the 0-th reverse move is reversion to the original configuration
    if (direction == ReptationDirection::HEAD)
    {
        // The new atom should match the 9-th atom in the original configuration 
        REQUIRE_THAT(
            (reverse_moves.row(0) - coords.row(9)).norm(), 
            Catch::Matchers::WithinAbs(0, tol)
        );
    } 
    else
    {
        // The new atom should match the 0-th atom in the original configuration  
        REQUIRE_THAT(
            (reverse_moves.row(0) - coords.row(0)).norm(), 
            Catch::Matchers::WithinAbs(0, tol)
        );
    }

    // Re-calculate the Boltzmann weights of the individual moves from scratch ... 
    reverse_direction = (
        direction == ReptationDirection::HEAD ? ReptationDirection::TAIL
        : ReptationDirection::HEAD
    );
    weights_forward = Matrix<double, Dynamic, 1>::Zero(n_candidates); 
    weights_reverse = Matrix<double, Dynamic, 1>::Zero(n_candidates); 

    // For each forward move ... 
    for (int i = 0; i < n_candidates; ++i)
    {
        // Generate the reptated configuration from the original configuration 
        PolymerConfiguration<double> config2(config);
        if (direction == ReptationDirection::HEAD)
            config2.reptateTowardsHead(forward_moves.row(i)); 
        else 
            config2.reptateTowardsTail(forward_moves.row(i));

        // Calculate the non-bonded energy difference between the two 
        // configurations  
        double diff = (
            config2.getNonbondedEnergy(lj_params, neighbor_threshold, true) - 
            config.getNonbondedEnergy(lj_params, neighbor_threshold, true)
        ); 

        // Calculate the corresponding Boltzmann weight 
        weights_forward(i) = exp(-diff / kT);
    }

    // For each reverse move ... 
    for (int i = 0; i < n_candidates; ++i)
    {
        // Generate a new reptated configuration from the *reptated*
        // configuration, in the reverse direction 
        PolymerConfiguration<double> config2(config_reptated); 
        if (direction == ReptationDirection::HEAD)
            config2.reptateTowardsTail(reverse_moves.row(i)); 
        else 
            config2.reptateTowardsHead(reverse_moves.row(i));

        // Calculate the non-bonded energy difference between the two 
        // configurations  
        double diff = (
            config2.getNonbondedEnergy(lj_params, neighbor_threshold, true) - 
            config_reptated.getNonbondedEnergy(lj_params, neighbor_threshold, true)
        );

        // Calculate the corresponding Boltzmann weight  
        weights_reverse(i) = exp(-diff / kT);
    }

    // Calculate the Rosenbluth factors 
    forward_rosenbluth = weights_forward.sum(); 
    reverse_rosenbluth = weights_reverse.sum();

    // Check that the ratio of Rosenbluth factors is equal to the acceptance 
    // probability (if it is <= 1)
    std::cout << "- Forward Rosenbluth (Gaussian angle potential, harmonic dihedral potential): "
              << forward_rosenbluth << std::endl; 
    std::cout << "- Reverse Rosenbluth (Gaussian angle potential, harmonic dihedral potential): "
              << reverse_rosenbluth << std::endl; 
    std::cout << "- Acceptance probability: " << prob_accept << std::endl;  
    REQUIRE_THAT(
        min(forward_rosenbluth / reverse_rosenbluth, 1.0),
        Catch::Matchers::WithinAbs(prob_accept, tol)
    );

    // Check that, if the acceptance probability is 1, the chosen move was taken 
    if (prob_accept == 1)
        REQUIRE(accepted_move == CBMCMoveResult::ACCEPT); 

    // Check that, if the chosen move was taken, the resulting configuration is
    // as expected
    coords_reptated = config_reptated.getSegment(0, length);
    coords_result = sampler_gaussian.getCoords(); 
    if (accepted_move == CBMCMoveResult::ACCEPT)
    {
        // Move was taken 
        for (int i = 0; i < length; ++i)
        {
            REQUIRE_THAT(
                (coords_result.row(i) - coords_reptated.row(i)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 
    }
    else 
    {
        // Move was not taken 
        for (int i = 0; i < length; ++i)
        {
            REQUIRE_THAT(
                (coords_result.row(i) - coords.row(i)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 
    } 
}

