/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     2/11/2026
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

/**
 * Tests for PolymerCBMCSampler::generateTerminalSegmentMoves(). 
 */
TEST_CASE("Tests for terminal segment move generation", "[generateTerminalSegmentMoves()]")
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

    // Initialize sampler instance
    double neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"]; 
    PolymerCBMCSampler<double> sampler_cosine(
        config, lj_params, neighbor_threshold, fene_params, AngleMode::COSINE,
        cosine_params, dihedral_params, rng
    );  
    
    // Try generating 50 3-atom terminal segment moves at the head
    int n_candidates = 50;  
    auto result = sampler_cosine.generateTerminalSegmentMoves(
        3, TerminalSegmentEnd::HEAD, n_candidates
    );
    Matrix<double, Dynamic, Dynamic> r_new = result.first; 
    Matrix<double, Dynamic, 1> energy_diffs = result.second;
    REQUIRE(r_new.rows() == n_candidates);
    REQUIRE(r_new.cols() == 9);  
    REQUIRE(energy_diffs.size() == n_candidates); 

    // Check that each new segment has atoms with valid distances to their 
    // neighbors 
    for (int i = 0; i < n_candidates; ++i)
    {
        Matrix<double, 3, 1> r1 = coords.row(3); 
        Matrix<double, 3, 1> r2 = r_new(i, Eigen::seqN(6, 3));
        Matrix<double, 3, 1> r3 = r_new(i, Eigen::seqN(3, 3)); 
        Matrix<double, 3, 1> r4 = r_new(i, Eigen::seqN(0, 3)); 
        REQUIRE((r2 - r1).norm() < fene_params["R0"]);
        REQUIRE((r3 - r2).norm() < fene_params["R0"]);
        REQUIRE((r4 - r3).norm() < fene_params["R0"]);
    } 

    // Check the non-bonded energy differences
    for (int i = 0; i < n_candidates; ++i)
    {
        Matrix<double, Dynamic, 3> segment_i(3, 3); 
        for (int j = 0; j < 3; ++j)
            segment_i.row(j) = r_new(i, Eigen::seqN(3 * j, 3)); 
        REQUIRE_THAT(
            energy_diffs(i),
            Catch::Matchers::WithinAbs(
                config.getSegmentReplacementNonbondedEnergyDifference(
                    segment_i, 0, lj_params, neighbor_threshold
                ), 
                tol
            )  
        );
    }

    // Try generating 50 3-atom terminal segment moves at the tail 
    result = sampler_cosine.generateTerminalSegmentMoves(
        3, TerminalSegmentEnd::TAIL, n_candidates
    );
    r_new = result.first; 
    energy_diffs = result.second;
    REQUIRE(r_new.rows() == n_candidates);
    REQUIRE(r_new.cols() == 9);  
    REQUIRE(energy_diffs.size() == n_candidates); 

    // Check that each new segment has atoms with valid distances to their 
    // neighbors 
    for (int i = 0; i < n_candidates; ++i)
    {
        Matrix<double, 3, 1> r1 = coords.row(6); 
        Matrix<double, 3, 1> r2 = r_new(i, Eigen::seqN(0, 3));
        Matrix<double, 3, 1> r3 = r_new(i, Eigen::seqN(3, 3)); 
        Matrix<double, 3, 1> r4 = r_new(i, Eigen::seqN(6, 3)); 
        REQUIRE((r2 - r1).norm() < fene_params["R0"]);
        REQUIRE((r3 - r2).norm() < fene_params["R0"]);
        REQUIRE((r4 - r3).norm() < fene_params["R0"]);
    } 

    // Check the non-bonded energy differences
    for (int i = 0; i < n_candidates; ++i)
    {
        Matrix<double, Dynamic, 3> segment_i(3, 3); 
        for (int j = 0; j < 3; ++j)
            segment_i.row(j) = r_new(i, Eigen::seqN(3 * j, 3)); 
        REQUIRE_THAT(
            energy_diffs(i),
            Catch::Matchers::WithinAbs(
                config.getSegmentReplacementNonbondedEnergyDifference(
                    segment_i, 7, lj_params, neighbor_threshold
                ), 
                tol
            )  
        );
    }

    // Generate a 10-mer with a dual Gaussian mixture angle potential
    config = generateKMer<double, 10>(
        lj_params, fene_params, AngleMode::GAUSSIAN, gaussian_params,
        dihedral_params, r0, collision_threshold, max_tries_per_atom,
        max_n_backtracks, rng, uniform_dist
    );
    coords = config.getSegment(0, 10);  
    REQUIRE(config.getLength() == 10);
    REQUIRE(coords.rows() == 10); 

    // Initialize sampler instance
    PolymerCBMCSampler<double> sampler_gaussian(
        config, lj_params, neighbor_threshold, fene_params, AngleMode::GAUSSIAN,
        gaussian_params, dihedral_params, rng
    );  

    // Try generating 50 3-atom terminal segment moves at the head
    result = sampler_gaussian.generateTerminalSegmentMoves(
        3, TerminalSegmentEnd::HEAD, n_candidates
    );
    r_new = result.first; 
    energy_diffs = result.second;
    REQUIRE(r_new.rows() == n_candidates);
    REQUIRE(r_new.cols() == 9);  
    REQUIRE(energy_diffs.size() == n_candidates); 

    // Check that each new segment has atoms with valid distances to their 
    // neighbors 
    for (int i = 0; i < n_candidates; ++i)
    {
        Matrix<double, 3, 1> r1 = coords.row(3); 
        Matrix<double, 3, 1> r2 = r_new(i, Eigen::seqN(6, 3));
        Matrix<double, 3, 1> r3 = r_new(i, Eigen::seqN(3, 3)); 
        Matrix<double, 3, 1> r4 = r_new(i, Eigen::seqN(0, 3)); 
        REQUIRE((r2 - r1).norm() < fene_params["R0"]);
        REQUIRE((r3 - r2).norm() < fene_params["R0"]);
        REQUIRE((r4 - r3).norm() < fene_params["R0"]);
    } 

    // Check the non-bonded energy differences
    for (int i = 0; i < n_candidates; ++i)
    {
        Matrix<double, Dynamic, 3> segment_i(3, 3); 
        for (int j = 0; j < 3; ++j)
            segment_i.row(j) = r_new(i, Eigen::seqN(3 * j, 3)); 
        REQUIRE_THAT(
            energy_diffs(i),
            Catch::Matchers::WithinAbs(
                config.getSegmentReplacementNonbondedEnergyDifference(
                    segment_i, 0, lj_params, neighbor_threshold
                ), 
                tol
            )  
        );
    }

    // Try generating 50 3-atom terminal segment moves at the tail 
    result = sampler_gaussian.generateTerminalSegmentMoves(
        3, TerminalSegmentEnd::TAIL, n_candidates
    );
    r_new = result.first; 
    energy_diffs = result.second;
    REQUIRE(r_new.rows() == n_candidates);
    REQUIRE(r_new.cols() == 9);
    REQUIRE(energy_diffs.size() == n_candidates); 

    // Check that each new segment has atoms with valid distances to their 
    // neighbors 
    for (int i = 0; i < n_candidates; ++i)
    {
        Matrix<double, 3, 1> r1 = coords.row(6); 
        Matrix<double, 3, 1> r2 = r_new(i, Eigen::seqN(0, 3));
        Matrix<double, 3, 1> r3 = r_new(i, Eigen::seqN(3, 3)); 
        Matrix<double, 3, 1> r4 = r_new(i, Eigen::seqN(6, 3)); 
        REQUIRE((r2 - r1).norm() < fene_params["R0"]);
        REQUIRE((r3 - r2).norm() < fene_params["R0"]);
        REQUIRE((r4 - r3).norm() < fene_params["R0"]);
    } 

    // Check the non-bonded energy differences
    for (int i = 0; i < n_candidates; ++i)
    {
        Matrix<double, Dynamic, 3> segment_i(3, 3); 
        for (int j = 0; j < 3; ++j)
            segment_i.row(j) = r_new(i, Eigen::seqN(3 * j, 3)); 
        REQUIRE_THAT(
            energy_diffs(i),
            Catch::Matchers::WithinAbs(
                config.getSegmentReplacementNonbondedEnergyDifference(
                    segment_i, 7, lj_params, neighbor_threshold
                ), 
                tol
            )  
        );
    }
}

/**
 * Tests for terminal segment moves in the PolymerCBMCSampler class via 
 * moveOnce(). 
 */
TEST_CASE("Tests for terminal segment moves", "[moveOnce()]")
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

    // Initialize sampler instance
    double neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"]; 
    PolymerCBMCSampler<double> sampler_cosine(
        config, lj_params, neighbor_threshold, fene_params, AngleMode::COSINE,
        cosine_params, dihedral_params, rng
    );  

    // Try moving a terminal 3-atom segment by choosing from 50 candidate 
    // moves 
    int n_candidates = 50; 
    PolymerConfiguration<double> config_moved(config);
    auto result_cosine = sampler_cosine.moveOnce(
        n_candidates, CBMCMoveType::TERMINAL_SEGMENT, 3, {}
    ); 
    Matrix<double, Dynamic, Dynamic> forward_moves = std::get<0>(result_cosine); 
    Matrix<double, Dynamic, Dynamic> reverse_moves = std::get<1>(result_cosine); 
    int move_idx = std::get<2>(result_cosine); 
    double prob_accept = std::get<3>(result_cosine);
    CBMCMoveResult accepted_move = std::get<4>(result_cosine);
    TerminalSegmentEnd direction = static_cast<TerminalSegmentEnd>(
        std::get<5>(result_cosine).at("terminal_end")
    ); 
    REQUIRE(forward_moves.rows() == n_candidates);
    REQUIRE(forward_moves.cols() == 9);  
    REQUIRE(reverse_moves.rows() == n_candidates);
    REQUIRE(reverse_moves.cols() == 9); 
    REQUIRE((move_idx >= 0 && move_idx < n_candidates));
    REQUIRE((prob_accept >= 0 && prob_accept <= 1)); 
    REQUIRE(accepted_move != CBMCMoveResult::NONE);  

    // Generate the modified configuration (in case the move was not accepted)
    Matrix<double, Dynamic, 3> move_segment(3, 3); 
    for (int i = 0; i < 3; ++i)
        move_segment.row(i) = forward_moves(move_idx, Eigen::seqN(3 * i, 3)); 
    if (direction == TerminalSegmentEnd::HEAD)
        config_moved.replaceSegment(move_segment, 0); 
    else 
        config_moved.replaceSegment(move_segment, 7);

    // Check that each new atom has a valid distance to the terminal atom 
    // at the appropriate end 
    for (int i = 0; i < n_candidates; ++i)
    {
        if (direction == TerminalSegmentEnd::HEAD)
        {
            REQUIRE((move_segment.row(2) - coords.row(3)).norm() < fene_params["R0"]); 
            REQUIRE((move_segment.row(1) - move_segment.row(2)).norm() < fene_params["R0"]); 
            REQUIRE((move_segment.row(0) - move_segment.row(1)).norm() < fene_params["R0"]); 
        }
        else 
        {
            REQUIRE((move_segment.row(0) - coords.row(6)).norm() < fene_params["R0"]); 
            REQUIRE((move_segment.row(1) - move_segment.row(0)).norm() < fene_params["R0"]); 
            REQUIRE((move_segment.row(2) - move_segment.row(1)).norm() < fene_params["R0"]); 
        }
    }

    // Check that the 0-th reverse move is reversion to the original configuration
    Matrix<double, Dynamic, 3> reverse_move_segment(3, 3); 
    for (int i = 0; i < 3; ++i)
        reverse_move_segment.row(i) = reverse_moves(0, Eigen::seqN(3 * i, 3)); 
    if (direction == TerminalSegmentEnd::HEAD)
    {
        for (int i = 0; i < 3; ++i)
            REQUIRE_THAT(
                (reverse_move_segment.row(i) - coords.row(i)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            );
    } 
    else 
    {
        for (int i = 0; i < 3; ++i)
            REQUIRE_THAT(
                (reverse_move_segment.row(i) - coords.row(7 + i)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            );
    }

    // Re-calculate the Rosenbluth weights ...
    Matrix<double, Dynamic, 1> weights_forward(n_candidates),
                               weights_reverse(n_candidates);  
    for (int i = 0; i < n_candidates; ++i)
    {
        Matrix<double, Dynamic, 3> forward_segment(3, 3), reverse_segment(3, 3);
        for (int j = 0; j < 3; ++j)
        {
            forward_segment.row(j) = forward_moves(i, Eigen::seqN(3 * j, 3));
            reverse_segment.row(j) = reverse_moves(i, Eigen::seqN(3 * j, 3));  
        } 
        double diff1 = config.getSegmentReplacementNonbondedEnergyDifference(
            forward_segment, (direction == TerminalSegmentEnd::HEAD ? 0 : 7), 
            lj_params, neighbor_threshold
        );
        double diff2 = config_moved.getSegmentReplacementNonbondedEnergyDifference(
            reverse_segment, (direction == TerminalSegmentEnd::HEAD ? 0 : 7),
            lj_params, neighbor_threshold
        );
        weights_forward(i) = exp(-diff1 / kT);
        weights_reverse(i) = exp(-diff2 / kT);
    }
    double forward_rosenbluth = weights_forward.sum(); 
    double reverse_rosenbluth = weights_reverse.sum();

    // Check that the ratio of Rosenbluth factors is equal to the acceptance 
    // probability  
    REQUIRE_THAT(
        min(1.0, forward_rosenbluth / reverse_rosenbluth),
        Catch::Matchers::WithinAbs(prob_accept, tol)
    );

    // Check that, if the acceptance probability is 1, the chosen move was taken 
    if (prob_accept == 1)
        REQUIRE(accepted_move == CBMCMoveResult::ACCEPT); 

    // Check that, if the chosen move was taken, the resulting configuration is
    // as expected
    Matrix<double, Dynamic, 3> coords_moved = config_moved.getSegment(0, 10); 
    Matrix<double, Dynamic, 3> coords_result = sampler_cosine.getCoords();  
    if (accepted_move == CBMCMoveResult::ACCEPT)
    {
        // Move was taken 
        for (int i = 0; i < 10; ++i)
        {
            REQUIRE_THAT(
                (coords_result.row(i) - coords_moved.row(i)).norm(),
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
                (coords_result.row(i) - coords.row(i)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 
    } 
    
    // Generate a 10-mer with a dual Gaussian mixture angle potential
    PolymerConfiguration<double> config2 = generateKMer<double, 10>(
        lj_params, fene_params, AngleMode::GAUSSIAN, gaussian_params,
        dihedral_params, r0, collision_threshold, max_tries_per_atom,
        max_n_backtracks, rng, uniform_dist
    );
    Matrix<double, Dynamic, 3> coords2 = config2.getSegment(0, 10);  
    REQUIRE(config2.getLength() == 10);
    REQUIRE(coords2.rows() == 10);

    // Initialize sampler instance
    PolymerCBMCSampler<double> sampler_gaussian(
        config2, lj_params, neighbor_threshold, fene_params, AngleMode::GAUSSIAN,
        gaussian_params, dihedral_params, rng
    );  

    // Try moving a terminal 5-atom segment
    PolymerConfiguration<double> config2_moved(config2);
    auto result_gaussian = sampler_gaussian.moveOnce(
        n_candidates, CBMCMoveType::TERMINAL_SEGMENT, 5, {}
    );
    forward_moves = std::get<0>(result_gaussian); 
    reverse_moves = std::get<1>(result_gaussian); 
    move_idx = std::get<2>(result_gaussian); 
    prob_accept = std::get<3>(result_gaussian); 
    accepted_move = std::get<4>(result_gaussian);
    direction = static_cast<TerminalSegmentEnd>(
        std::get<5>(result_gaussian).at("terminal_end")
    );
    REQUIRE(forward_moves.rows() == n_candidates);
    REQUIRE(forward_moves.cols() == 15);  
    REQUIRE(reverse_moves.rows() == n_candidates);
    REQUIRE(reverse_moves.cols() == 15); 
    REQUIRE((move_idx >= 0 && move_idx < n_candidates));
    REQUIRE((prob_accept >= 0 && prob_accept <= 1)); 
    REQUIRE(accepted_move != CBMCMoveResult::NONE);  

    // Generate the modified configuration (in case the move was not accepted)
    move_segment.resize(5, 3); 
    for (int i = 0; i < 5; ++i)
        move_segment.row(i) = forward_moves(move_idx, Eigen::seqN(3 * i, 3)); 
    if (direction == TerminalSegmentEnd::HEAD)
        config2_moved.replaceSegment(move_segment, 0); 
    else 
        config2_moved.replaceSegment(move_segment, 5); 

    // Check that each new atom has a valid distance to the terminal atom 
    // at the appropriate end 
    for (int i = 0; i < n_candidates; ++i)
    {
        if (direction == TerminalSegmentEnd::HEAD)
        {
            REQUIRE((move_segment.row(4) - coords2.row(5)).norm() < fene_params["R0"]); 
            REQUIRE((move_segment.row(3) - move_segment.row(4)).norm() < fene_params["R0"]); 
            REQUIRE((move_segment.row(2) - move_segment.row(3)).norm() < fene_params["R0"]); 
            REQUIRE((move_segment.row(1) - move_segment.row(2)).norm() < fene_params["R0"]); 
            REQUIRE((move_segment.row(0) - move_segment.row(1)).norm() < fene_params["R0"]); 
        }
        else 
        {
            REQUIRE((move_segment.row(0) - coords2.row(4)).norm() < fene_params["R0"]); 
            REQUIRE((move_segment.row(1) - move_segment.row(0)).norm() < fene_params["R0"]); 
            REQUIRE((move_segment.row(2) - move_segment.row(1)).norm() < fene_params["R0"]);
            REQUIRE((move_segment.row(3) - move_segment.row(2)).norm() < fene_params["R0"]); 
            REQUIRE((move_segment.row(4) - move_segment.row(3)).norm() < fene_params["R0"]); 
        }
    }

    // Check that the 0-th reverse move is reversion to the original configuration
    reverse_move_segment.resize(5, 3); 
    for (int i = 0; i < 5; ++i)
        reverse_move_segment.row(i) = reverse_moves(0, Eigen::seqN(3 * i, 3)); 
    if (direction == TerminalSegmentEnd::HEAD)
    {
        for (int i = 0; i < 5; ++i)
            REQUIRE_THAT(
                (reverse_move_segment.row(i) - coords2.row(i)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            );
    } 
    else 
    {
        for (int i = 0; i < 5; ++i)
            REQUIRE_THAT(
                (reverse_move_segment.row(i) - coords2.row(5 + i)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            );
    }

    // Re-calculate the Rosenbluth weights ...
    for (int i = 0; i < n_candidates; ++i)
    {
        Matrix<double, Dynamic, 3> forward_segment(5, 3), reverse_segment(5, 3);
        for (int j = 0; j < 5; ++j)
        {
            forward_segment.row(j) = forward_moves(i, Eigen::seqN(3 * j, 3));
            reverse_segment.row(j) = reverse_moves(i, Eigen::seqN(3 * j, 3));  
        } 
        double diff1 = config2.getSegmentReplacementNonbondedEnergyDifference(
            forward_segment, (direction == TerminalSegmentEnd::HEAD ? 0 : 5), 
            lj_params, neighbor_threshold
        );
        double diff2 = config2_moved.getSegmentReplacementNonbondedEnergyDifference(
            reverse_segment, (direction == TerminalSegmentEnd::HEAD ? 0 : 5),
            lj_params, neighbor_threshold
        );
        weights_forward(i) = exp(-diff1 / kT);
        weights_reverse(i) = exp(-diff2 / kT);
    }
    forward_rosenbluth = weights_forward.sum(); 
    reverse_rosenbluth = weights_reverse.sum();

    // Check that the ratio of Rosenbluth factors is equal to the acceptance 
    // probability  
    REQUIRE_THAT(
        min(1.0, forward_rosenbluth / reverse_rosenbluth),
        Catch::Matchers::WithinAbs(prob_accept, tol)
    );

    // Check that, if the acceptance probability is 1, the chosen move was taken 
    if (prob_accept == 1)
        REQUIRE(accepted_move == CBMCMoveResult::ACCEPT); 

    // Check that, if the chosen move was taken, the resulting configuration is
    // as expected
    Matrix<double, Dynamic, 3> coords2_moved = config2_moved.getSegment(0, 10);
    Matrix<double, Dynamic, 3> coords2_result = sampler_gaussian.getCoords(); 
    if (accepted_move == CBMCMoveResult::ACCEPT)
    {
        // Move was taken 
        for (int i = 0; i < 10; ++i)
        {
            REQUIRE_THAT(
                (coords2_result.row(i) - coords2_moved.row(i)).norm(),
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
                (coords2_result.row(i) - coords2.row(i)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 
    }
}

