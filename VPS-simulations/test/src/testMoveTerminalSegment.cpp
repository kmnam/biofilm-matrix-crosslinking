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
 * Tests for generateTerminalSegmentMoves(). 
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

    // Try generating 50 3-atom terminal segment moves at the head
    int n_candidates = 50;  
    double neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"]; 
    auto result = generateTerminalSegmentMoves<double, 3>(
        config, TerminalSegmentEnd::HEAD, n_candidates, rng, uniform_dist,
        lj_params, neighbor_threshold, fene_params, AngleMode::COSINE,
        cosine_params, dihedral_params 
    );
    Matrix<double, Dynamic, 9> r_new = result.first; 
    Matrix<double, Dynamic, 1> energy_diffs = result.second;
    REQUIRE(r_new.rows() == n_candidates); 
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
    result = generateTerminalSegmentMoves<double, 3>(
        config, TerminalSegmentEnd::TAIL, n_candidates, rng, uniform_dist,
        lj_params, neighbor_threshold, fene_params, AngleMode::COSINE,
        cosine_params, dihedral_params 
    );
    r_new = result.first; 
    energy_diffs = result.second;
    REQUIRE(r_new.rows() == n_candidates); 
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

    // Try generating 50 3-atom terminal segment moves at the head
    result = generateTerminalSegmentMoves<double, 3>(
        config, TerminalSegmentEnd::HEAD, n_candidates, rng, uniform_dist,
        lj_params, neighbor_threshold, fene_params, AngleMode::GAUSSIAN,
        gaussian_params, dihedral_params
    );
    r_new = result.first; 
    energy_diffs = result.second;
    REQUIRE(r_new.rows() == n_candidates); 
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
    result = generateTerminalSegmentMoves<double, 3>(
        config, TerminalSegmentEnd::TAIL, n_candidates, rng, uniform_dist,
        lj_params, neighbor_threshold, fene_params, AngleMode::GAUSSIAN,
        gaussian_params, dihedral_params
    );
    r_new = result.first; 
    energy_diffs = result.second;
    REQUIRE(r_new.rows() == n_candidates); 
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
 * Tests for moveTerminalSegment(). 
 */
TEST_CASE("Tests for terminal segment moves", "[moveTerminalSegment()]")
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

    // Try moving a terminal 3-atom segment
    int n_candidates = 50; 
    double neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"];
    PolymerConfiguration<double> config2(config), config3(config);  
    auto result_cosine = moveTerminalSegment<double, 3>(
        config2, n_candidates, rng, uniform_dist, lj_params, neighbor_threshold, 
        fene_params, AngleMode::COSINE, cosine_params, dihedral_params
    );
    TerminalSegmentEnd direction = std::get<0>(result_cosine);  
    Matrix<double, Dynamic, 9> forward_moves = std::get<1>(result_cosine); 
    Matrix<double, Dynamic, 9> reverse_moves = std::get<2>(result_cosine); 
    int move_idx = std::get<3>(result_cosine); 
    double prob_accept = std::get<4>(result_cosine); 
    bool accepted_move = std::get<5>(result_cosine);
    Matrix<double, Dynamic, 3> move_segment(3, 3); 
    for (int i = 0; i < 3; ++i)
        move_segment.row(i) = forward_moves(move_idx, Eigen::seqN(3 * i, 3)); 
    if (direction == TerminalSegmentEnd::HEAD)
        config3.replaceSegment(move_segment, 0); 
    else 
        config3.replaceSegment(move_segment, 7); 
    REQUIRE(forward_moves.rows() == n_candidates); 
    REQUIRE(reverse_moves.rows() == n_candidates);
    REQUIRE((move_idx >= 0 && move_idx < n_candidates));
    REQUIRE((prob_accept >= 0 && prob_accept <= 1));  

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

    // Check that the reverse move corresponding to the chosen move is
    // reversion to the original configuration
    Matrix<double, Dynamic, 3> reverse_move_segment(3, 3); 
    for (int i = 0; i < 3; ++i)
        reverse_move_segment.row(i) = reverse_moves(move_idx, Eigen::seqN(3 * i, 3)); 
    if (direction == TerminalSegmentEnd::HEAD)
    {
        // In this case, the reverse move concerns the tail segment 
        for (int i = 0; i < 3; ++i)
            REQUIRE_THAT(
                (reverse_move_segment.row(i) - coords.row(7 + i)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            );
    } 
    else 
    {
        // In this case, the reverse move concerns the head segment
        for (int i = 0; i < 3; ++i)
            REQUIRE_THAT(
                (reverse_move_segment.row(i) - coords.row(i)).norm(), 
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
        double diff2 = config3.getSegmentReplacementNonbondedEnergyDifference(
            reverse_segment, (direction == TerminalSegmentEnd::HEAD ? 7 : 0),
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

    // Try moving a terminal 5-atom segment
    PolymerConfiguration<double> config5(config4), config6(config4);  
    auto result_gaussian = moveTerminalSegment<double, 5>(
        config5, n_candidates, rng, uniform_dist, lj_params, neighbor_threshold, 
        fene_params, AngleMode::GAUSSIAN, gaussian_params, dihedral_params
    );
    direction = std::get<0>(result_gaussian);  
    Matrix<double, Dynamic, 15> forward_moves2 = std::get<1>(result_gaussian); 
    Matrix<double, Dynamic, 15> reverse_moves2 = std::get<2>(result_gaussian); 
    move_idx = std::get<3>(result_gaussian); 
    prob_accept = std::get<4>(result_gaussian); 
    accepted_move = std::get<5>(result_gaussian);
    move_segment.resize(5, 3); 
    for (int i = 0; i < 5; ++i)
        move_segment.row(i) = forward_moves2(move_idx, Eigen::seqN(3 * i, 3)); 
    if (direction == TerminalSegmentEnd::HEAD)
        config6.replaceSegment(move_segment, 0); 
    else 
        config6.replaceSegment(move_segment, 5); 
    REQUIRE(forward_moves2.rows() == n_candidates); 
    REQUIRE(reverse_moves2.rows() == n_candidates);
    REQUIRE((move_idx >= 0 && move_idx < n_candidates));
    REQUIRE((prob_accept >= 0 && prob_accept <= 1));  

    // Check that each new atom has a valid distance to the terminal atom 
    // at the appropriate end 
    for (int i = 0; i < n_candidates; ++i)
    {
        if (direction == TerminalSegmentEnd::HEAD)
        {
            REQUIRE((move_segment.row(4) - coords4.row(5)).norm() < fene_params["R0"]); 
            REQUIRE((move_segment.row(3) - move_segment.row(4)).norm() < fene_params["R0"]); 
            REQUIRE((move_segment.row(2) - move_segment.row(3)).norm() < fene_params["R0"]); 
            REQUIRE((move_segment.row(1) - move_segment.row(2)).norm() < fene_params["R0"]); 
            REQUIRE((move_segment.row(0) - move_segment.row(1)).norm() < fene_params["R0"]); 
        }
        else 
        {
            REQUIRE((move_segment.row(0) - coords4.row(4)).norm() < fene_params["R0"]); 
            REQUIRE((move_segment.row(1) - move_segment.row(0)).norm() < fene_params["R0"]); 
            REQUIRE((move_segment.row(2) - move_segment.row(1)).norm() < fene_params["R0"]);
            REQUIRE((move_segment.row(3) - move_segment.row(2)).norm() < fene_params["R0"]); 
            REQUIRE((move_segment.row(4) - move_segment.row(3)).norm() < fene_params["R0"]); 
        }
    }

    // Check that the reverse move corresponding to the chosen move is
    // reversion to the original configuration
    reverse_move_segment.resize(5, 3); 
    for (int i = 0; i < 5; ++i)
        reverse_move_segment.row(i) = reverse_moves2(move_idx, Eigen::seqN(3 * i, 3)); 
    if (direction == TerminalSegmentEnd::HEAD)
    {
        // In this case, the reverse move concerns the tail segment 
        for (int i = 0; i < 5; ++i)
            REQUIRE_THAT(
                (reverse_move_segment.row(i) - coords4.row(5 + i)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            );
    } 
    else 
    {
        // In this case, the reverse move concerns the head segment
        for (int i = 0; i < 5; ++i)
            REQUIRE_THAT(
                (reverse_move_segment.row(i) - coords4.row(i)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            );
    }

    // Re-calculate the Rosenbluth weights ...
    for (int i = 0; i < n_candidates; ++i)
    {
        Matrix<double, Dynamic, 3> forward_segment(5, 3), reverse_segment(5, 3);
        for (int j = 0; j < 5; ++j)
        {
            forward_segment.row(j) = forward_moves2(i, Eigen::seqN(3 * j, 3));
            reverse_segment.row(j) = reverse_moves2(i, Eigen::seqN(3 * j, 3));  
        } 
        double diff1 = config4.getSegmentReplacementNonbondedEnergyDifference(
            forward_segment, (direction == TerminalSegmentEnd::HEAD ? 0 : 5), 
            lj_params, neighbor_threshold
        );
        double diff2 = config6.getSegmentReplacementNonbondedEnergyDifference(
            reverse_segment, (direction == TerminalSegmentEnd::HEAD ? 5 : 0),
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

