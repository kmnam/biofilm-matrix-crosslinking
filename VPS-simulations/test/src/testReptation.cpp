/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     2/1/2026
 */

#include <iostream>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/random.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/utils.hpp"
#include "../../include/cbmc.hpp"

using namespace Eigen;

/**
 * Tests for generateReptationMoves(). 
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
                                            cosine_params,
                                            gaussian_params,
                                            dihedral_params;
    double kT = 1.380649e-2 * 300;
    lj_params["eps"] = kT; 
    lj_params["sigma"] = 0.9;
    fene_params["K"] = 30 * kT; 
    fene_params["R0"] = 1.5;
    cosine_params["K"] = 20 * kT;
    cosine_params["theta0"] = 160 * boost::math::constants::pi<double>() / 180;
    gaussian_params["A1"] = 0.9; 
    gaussian_params["A2"] = 0.1;
    gaussian_params["w1"] = 0.2236067977;    // = 1/sqrt(20) 
    gaussian_params["w2"] = 0.2236067977; 
    gaussian_params["theta1"] = 160 * boost::math::constants::pi<double>() / 180; 
    gaussian_params["theta2"] = 90 * boost::math::constants::pi<double>() / 180; 
    dihedral_params["K"] = 10 * kT;
    const double collision_threshold = 0.1;
    const int max_tries_per_atom = 50;
    const int max_n_backtracks = 50;  

    // Generate a 10-mer with a cosine angle potential
    PolymerConfiguration<double> config = generateKMer<double, 10>(
        lj_params, fene_params, AngleMode::COSINE, cosine_params, dihedral_params,
        r0, collision_threshold, max_tries_per_atom, max_n_backtracks, rng, 
        uniform_dist
    );
    Matrix<double, Dynamic, 3> coords = config.getSegment(0, 10);  
    REQUIRE(config.getLength() == 10);
    REQUIRE(coords.rows() == 10);  

    // Try generating 50 reptation moves at the head
    int n_candidates = 50;  
    double neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"]; 
    auto result = generateReptationMoves<double>(
        config, ReptationDirection::HEAD, n_candidates, rng, uniform_dist,
        lj_params, neighbor_threshold, fene_params, AngleMode::COSINE,
        cosine_params, dihedral_params 
    );
    Matrix<double, Dynamic, 3> r_new = result.first; 
    Matrix<double, Dynamic, 1> energy_diffs = result.second;
    REQUIRE(r_new.rows() == n_candidates); 
    REQUIRE(energy_diffs.size() == n_candidates); 

    // Check that each new atom has a valid distance to the 0-th atom
    for (int i = 0; i < n_candidates; ++i) 
        REQUIRE((r_new.row(i) - coords.row(0)).norm() < fene_params["R0"]); 

    // Check the reptation non-bonded energy difference
    for (int i = 0; i < n_candidates; ++i)
        REQUIRE_THAT(
            energy_diffs(i),
            Catch::Matchers::WithinAbs(
                config.getReptationNonbondedEnergyDifference(
                    ReptationDirection::HEAD, r_new.row(i), lj_params,
                    neighbor_threshold
                ), 
                tol
            )  
        );

    // Try generating 50 reptation moves at the tail 
    result = generateReptationMoves<double>(
        config, ReptationDirection::TAIL, n_candidates, rng, uniform_dist,
        lj_params, neighbor_threshold, fene_params, AngleMode::COSINE,
        cosine_params, dihedral_params 
    );
    r_new = result.first; 
    energy_diffs = result.second;
    REQUIRE(r_new.rows() == n_candidates); 
    REQUIRE(energy_diffs.size() == n_candidates); 

    // Check that the new atom has a valid distance to the 0-th atom
    for (int i = 0; i < n_candidates; ++i)
        REQUIRE((r_new.row(i) - coords.row(9)).norm() < fene_params["R0"]); 

    // Check the reptation non-bonded energy difference
    for (int i = 0; i < n_candidates; ++i)
        REQUIRE_THAT(
            energy_diffs(i),
            Catch::Matchers::WithinAbs(
                config.getReptationNonbondedEnergyDifference(
                    ReptationDirection::TAIL, r_new.row(i), lj_params,
                    neighbor_threshold
                ), 
                tol
            )  
        );

    // Generate a 10-mer with a dual Gaussian mixture angle potential
    config = generateKMer<double, 10>(
        lj_params, fene_params, AngleMode::GAUSSIAN, gaussian_params,
        dihedral_params, r0, collision_threshold, max_tries_per_atom,
        max_n_backtracks, rng, uniform_dist
    );
    coords = config.getSegment(0, 10);  
    REQUIRE(config.getLength() == 10);
    REQUIRE(coords.rows() == 10);  

    // Try generating one reptation move at the head 
    result = generateReptationMoves<double>(
        config, ReptationDirection::HEAD, n_candidates, rng, uniform_dist,
        lj_params, neighbor_threshold, fene_params, AngleMode::GAUSSIAN,
        gaussian_params, dihedral_params
    );
    r_new = result.first; 
    energy_diffs = result.second;
    REQUIRE(r_new.rows() == n_candidates); 
    REQUIRE(energy_diffs.size() == n_candidates); 

    // Check that the new atom has a valid distance to the 0-th atom
    for (int i = 0; i < n_candidates; ++i)
        REQUIRE((r_new.row(i) - coords.row(0)).norm() < fene_params["R0"]); 

    // Check the reptation non-bonded energy difference
    for (int i = 0; i < n_candidates; ++i)
        REQUIRE_THAT(
            energy_diffs(i),
            Catch::Matchers::WithinAbs(
                config.getReptationNonbondedEnergyDifference(
                    ReptationDirection::HEAD, r_new.row(i), lj_params,
                    neighbor_threshold
                ), 
                tol
            )  
        );

    // Try generating one reptation move at the tail
    result = generateReptationMoves<double>(
        config, ReptationDirection::TAIL, n_candidates, rng, uniform_dist,
        lj_params, neighbor_threshold, fene_params, AngleMode::GAUSSIAN,
        gaussian_params, dihedral_params
    );
    r_new = result.first; 
    energy_diffs = result.second;
    REQUIRE(r_new.rows() == n_candidates); 
    REQUIRE(energy_diffs.size() == n_candidates); 

    // Check that the new atom has a valid distance to the 0-th atom
    for (int i = 0; i < n_candidates; ++i)
        REQUIRE((r_new.row(i) - coords.row(9)).norm() < fene_params["R0"]); 

    // Check the reptation non-bonded energy difference
    for (int i = 0; i < n_candidates; ++i)
        REQUIRE_THAT(
            energy_diffs(i),
            Catch::Matchers::WithinAbs(
                config.getReptationNonbondedEnergyDifference(
                    ReptationDirection::TAIL, r_new.row(i), lj_params,
                    neighbor_threshold
                ), 
                tol
            )  
        );
}

