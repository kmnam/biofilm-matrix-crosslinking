/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     2/2/2026
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

    // Try generating 50 reptation moves at the head 
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

    // Try generating 50 reptation moves at the tail
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

/**
 * Tests for reptate(). 
 */
TEST_CASE("Tests for reptation", "[reptate()]")
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

    // Try reptating
    int n_candidates = 50; 
    double neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"];
    PolymerConfiguration<double> config2(config), config3(config);  
    auto result = reptate<double>(
        config2, n_candidates, rng, uniform_dist, lj_params, neighbor_threshold, 
        fene_params, AngleMode::COSINE, cosine_params, dihedral_params
    );
    ReptationDirection direction = std::get<0>(result);  
    Matrix<double, Dynamic, 3> forward_moves = std::get<1>(result); 
    Matrix<double, Dynamic, 3> reverse_moves = std::get<2>(result); 
    int move_idx = std::get<3>(result); 
    double prob_accept = std::get<4>(result); 
    bool accepted_move = std::get<5>(result);
    if (direction == ReptationDirection::HEAD)
        config3.reptateTowardsHead(forward_moves.row(move_idx)); 
    else 
        config3.reptateTowardsTail(forward_moves.row(move_idx));
    REQUIRE(forward_moves.rows() == n_candidates); 
    REQUIRE(reverse_moves.rows() == n_candidates);
    REQUIRE((move_idx >= 0 && move_idx < n_candidates));
    REQUIRE((prob_accept >= 0 && prob_accept <= 1));  

    // Check that each new atom has a valid distance to the terminal atom 
    // at the appropriate end 
    for (int i = 0; i < n_candidates; ++i)
    {
        if (direction == ReptationDirection::HEAD)
        { 
            REQUIRE((forward_moves.row(i) - coords.row(0)).norm() < fene_params["R0"]);
            REQUIRE((reverse_moves.row(i) - coords.row(8)).norm() < fene_params["R0"]); 
        }
        else 
        {
            REQUIRE((forward_moves.row(i) - coords.row(9)).norm() < fene_params["R0"]);
            REQUIRE((reverse_moves.row(i) - coords.row(1)).norm() < fene_params["R0"]); 
        }
    }

    // Check that the reverse move corresponding to the chosen move is
    // reversion to the original configuration
    if (direction == ReptationDirection::HEAD)
        REQUIRE_THAT(
            (reverse_moves.row(move_idx) - coords.row(9)).norm(), 
            Catch::Matchers::WithinAbs(0, tol)
        ); 
    else 
        REQUIRE_THAT(
            (reverse_moves.row(move_idx) - coords.row(0)).norm(), 
            Catch::Matchers::WithinAbs(0, tol)
        );

    // Re-calculate the Rosenbluth weights ...
    ReptationDirection reverse_direction = (
        direction == ReptationDirection::HEAD ? ReptationDirection::TAIL
        : ReptationDirection::HEAD
    ); 
    Matrix<double, Dynamic, 1> weights_forward(n_candidates),
                               weights_reverse(n_candidates);  
    for (int i = 0; i < n_candidates; ++i)
    {
        double diff1 = config.getReptationNonbondedEnergyDifference(
            direction, forward_moves.row(i), lj_params, neighbor_threshold
        );
        double diff2 = config3.getReptationNonbondedEnergyDifference(
            reverse_direction, reverse_moves.row(i), lj_params, neighbor_threshold
        );
        weights_forward(i) = exp(-diff1 / kT);
        weights_reverse(i) = exp(-diff2 / kT);
    }
    double forward_rosenbluth = weights_forward.sum(); 
    double reverse_rosenbluth = weights_reverse.sum();

    // Check that the ratio of Rosenbluth factors is equal to the acceptance 
    // probability  
    REQUIRE_THAT(
        forward_rosenbluth / reverse_rosenbluth,
        Catch::Matchers::WithinAbs(prob_accept, tol)
    );

    // Check that, if the acceptance probability is 1, the chosen move was taken 
    if (prob_accept == 1)
        REQUIRE(accepted_move); 

    // Check that, if the chosen move was taken, the resulting configuration is
    // as expected
    Matrix<double, Dynamic, 3> coords2 = config2.getSegment(0, 10);
    Matrix<double, Dynamic, 3> coords3 = config3.getSegment(0, 10);  
    if (accepted_move)
    {
        // Move was taken 
        for (int i = 0; i < 10; ++i)
        {
            REQUIRE_THAT(
                (coords2.row(i) - coords3.row(i)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 
    }
    else 
    {
        // Move was not taken 
        for (int i = 0; i < 10; ++i)
        {
            REQUIRE_THAT(
                (coords2.row(i) - coords.row(i)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 
    } 

    // Generate a 10-mer with a dual Gaussian mixture angle potential
    PolymerConfiguration<double> config4 = generateKMer<double, 10>(
        lj_params, fene_params, AngleMode::GAUSSIAN, gaussian_params,
        dihedral_params, r0, collision_threshold, max_tries_per_atom,
        max_n_backtracks, rng, uniform_dist
    );
    Matrix<double, Dynamic, 3> coords4 = config4.getSegment(0, 10);  
    REQUIRE(config4.getLength() == 10);
    REQUIRE(coords4.rows() == 10); 

    // Try reptating
    PolymerConfiguration<double> config5(config4), config6(config4);  
    result = reptate<double>(
        config5, n_candidates, rng, uniform_dist, lj_params, neighbor_threshold, 
        fene_params, AngleMode::GAUSSIAN, gaussian_params, dihedral_params
    );
    direction = std::get<0>(result);  
    forward_moves = std::get<1>(result); 
    reverse_moves = std::get<2>(result); 
    move_idx = std::get<3>(result); 
    prob_accept = std::get<4>(result); 
    accepted_move = std::get<5>(result);
    if (direction == ReptationDirection::HEAD)
        config6.reptateTowardsHead(forward_moves.row(move_idx)); 
    else 
        config6.reptateTowardsTail(forward_moves.row(move_idx));
    REQUIRE(forward_moves.rows() == n_candidates); 
    REQUIRE(reverse_moves.rows() == n_candidates);
    REQUIRE((move_idx >= 0 && move_idx < n_candidates));
    REQUIRE((prob_accept >= 0 && prob_accept <= 1));  

    // Check that each new atom has a valid distance to the terminal atom 
    // at the appropriate end 
    for (int i = 0; i < n_candidates; ++i)
    {
        if (direction == ReptationDirection::HEAD)
        { 
            REQUIRE((forward_moves.row(i) - coords4.row(0)).norm() < fene_params["R0"]);
            REQUIRE((reverse_moves.row(i) - coords4.row(8)).norm() < fene_params["R0"]); 
        }
        else 
        {
            REQUIRE((forward_moves.row(i) - coords4.row(9)).norm() < fene_params["R0"]);
            REQUIRE((reverse_moves.row(i) - coords4.row(1)).norm() < fene_params["R0"]); 
        }
    }

    // Check that the reverse move corresponding to the chosen move is
    // reversion to the original configuration
    if (direction == ReptationDirection::HEAD)
        REQUIRE_THAT(
            (reverse_moves.row(move_idx) - coords4.row(9)).norm(), 
            Catch::Matchers::WithinAbs(0, tol)
        ); 
    else 
        REQUIRE_THAT(
            (reverse_moves.row(move_idx) - coords4.row(0)).norm(), 
            Catch::Matchers::WithinAbs(0, tol)
        );

    // Re-calculate the Rosenbluth weights ...
    reverse_direction = (
        direction == ReptationDirection::HEAD ? ReptationDirection::TAIL
        : ReptationDirection::HEAD
    ); 
    for (int i = 0; i < n_candidates; ++i)
    {
        double diff1 = config4.getReptationNonbondedEnergyDifference(
            direction, forward_moves.row(i), lj_params, neighbor_threshold
        );
        double diff2 = config6.getReptationNonbondedEnergyDifference(
            reverse_direction, reverse_moves.row(i), lj_params, neighbor_threshold
        );
        weights_forward(i) = exp(-diff1 / kT);
        weights_reverse(i) = exp(-diff2 / kT);
    }
    forward_rosenbluth = weights_forward.sum(); 
    reverse_rosenbluth = weights_reverse.sum();

    // Check that the ratio of Rosenbluth factors is equal to the acceptance 
    // probability  
    REQUIRE_THAT(
        forward_rosenbluth / reverse_rosenbluth,
        Catch::Matchers::WithinAbs(prob_accept, tol)
    );

    // Check that, if the acceptance probability is 1, the chosen move was taken 
    if (prob_accept == 1)
        REQUIRE(accepted_move); 

    // Check that, if the chosen move was taken, the resulting configuration is
    // as expected
    Matrix<double, Dynamic, 3> coords5 = config5.getSegment(0, 10);
    Matrix<double, Dynamic, 3> coords6 = config6.getSegment(0, 10);  
    if (accepted_move)
    {
        // Move was taken 
        for (int i = 0; i < 10; ++i)
        {
            REQUIRE_THAT(
                (coords5.row(i) - coords6.row(i)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 
    }
    else 
    {
        // Move was not taken 
        for (int i = 0; i < 10; ++i)
        {
            REQUIRE_THAT(
                (coords5.row(i) - coords4.row(i)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 
    } 
}

