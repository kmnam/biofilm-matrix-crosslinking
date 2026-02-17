/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     2/16/2026
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
 * Tests for PolymerCBMCSampler::generateMultimerReptationMoves(). 
 */
TEST_CASE(
    "Tests for multimer reptation move generation",
    "[generateMultimerReptationMoves()]"
)
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

    // Try generating 50 3-mer reptation moves at the head
    int n_candidates = 50; 
    int n_reptate = 3;  
    auto result = sampler_cosine.generateMultimerReptationMoves(
        ReptationDirection::HEAD, n_reptate, n_candidates
    );
    Matrix<double, Dynamic, Dynamic> r_new = result.first; 
    Matrix<double, Dynamic, 1> energy_diffs = result.second;
    REQUIRE(r_new.rows() == n_candidates);
    REQUIRE(r_new.cols() == 3 * n_reptate);  
    REQUIRE(energy_diffs.size() == n_candidates); 

    // Check that the last atom in each new segment has a valid distance 
    // to the 0-th atom 
    for (int i = 0; i < n_candidates; ++i) 
        REQUIRE((r_new(i, Eigen::seqN(6, 3)) - coords.row(0)).norm() < fene_params["R0"]); 

    // Check the reptation non-bonded energy difference
    for (int i = 0; i < n_candidates; ++i)
    {
        Matrix<double, Dynamic, 3> segment_i(n_reptate, 3); 
        for (int j = 0; j < n_reptate; ++j)
            segment_i.row(j) = r_new(i, Eigen::seqN(3 * j, 3)); 
        REQUIRE_THAT(
            energy_diffs(i),
            Catch::Matchers::WithinAbs(
                config.getMultimerReptationNonbondedEnergyDifference(
                    ReptationDirection::HEAD, segment_i, lj_params,
                    neighbor_threshold
                ), 
                tol
            )  
        );
    }

    // Try generating 50 3-mer reptation moves at the tail 
    result = sampler_cosine.generateMultimerReptationMoves(
        ReptationDirection::TAIL, n_reptate, n_candidates
    );
    r_new = result.first; 
    energy_diffs = result.second;
    REQUIRE(r_new.rows() == n_candidates);
    REQUIRE(r_new.cols() == 3 * n_reptate);  
    REQUIRE(energy_diffs.size() == n_candidates); 

    // Check that the 0-th atom in each new segment has a valid distance to 
    // the final atom 
    for (int i = 0; i < n_candidates; ++i)
        REQUIRE((r_new(i, Eigen::seqN(0, 3)) - coords.row(9)).norm() < fene_params["R0"]); 

    // Check the reptation non-bonded energy difference
    for (int i = 0; i < n_candidates; ++i)
    {
        Matrix<double, Dynamic, 3> segment_i(n_reptate, 3); 
        for (int j = 0; j < n_reptate; ++j)
            segment_i.row(j) = r_new(i, Eigen::seqN(3 * j, 3)); 
        REQUIRE_THAT(
            energy_diffs(i),
            Catch::Matchers::WithinAbs(
                config.getMultimerReptationNonbondedEnergyDifference(
                    ReptationDirection::TAIL, segment_i, lj_params,
                    neighbor_threshold
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

    // Try generating 50 4-mer reptation moves at the head
    n_reptate = 4; 
    result = sampler_gaussian.generateMultimerReptationMoves(
        ReptationDirection::HEAD, n_reptate, n_candidates
    );
    r_new = result.first; 
    energy_diffs = result.second;
    REQUIRE(r_new.rows() == n_candidates);
    REQUIRE(r_new.cols() == 3 * n_reptate); 
    REQUIRE(energy_diffs.size() == n_candidates); 

    // Check that the last atom in each new segment has a valid distance 
    // to the 0-th atom 
    for (int i = 0; i < n_candidates; ++i) 
        REQUIRE((r_new(i, Eigen::seqN(9, 3)) - coords.row(0)).norm() < fene_params["R0"]);

    // Check the reptation non-bonded energy difference
    for (int i = 0; i < n_candidates; ++i)
    {
        Matrix<double, Dynamic, 3> segment_i(n_reptate, 3); 
        for (int j = 0; j < n_reptate; ++j)
            segment_i.row(j) = r_new(i, Eigen::seqN(3 * j, 3)); 
        REQUIRE_THAT(
            energy_diffs(i),
            Catch::Matchers::WithinAbs(
                config.getMultimerReptationNonbondedEnergyDifference(
                    ReptationDirection::HEAD, segment_i, lj_params,
                    neighbor_threshold
                ), 
                tol
            )  
        );
    }

    // Try generating 50 4-mer reptation moves at the tail
    result = sampler_gaussian.generateMultimerReptationMoves(
        ReptationDirection::TAIL, n_reptate, n_candidates
    );
    r_new = result.first; 
    energy_diffs = result.second;
    REQUIRE(r_new.rows() == n_candidates);
    REQUIRE(r_new.cols() == 3 * n_reptate);  
    REQUIRE(energy_diffs.size() == n_candidates); 

    // Check that the 0-th atom in each new segment has a valid distance to 
    // the final atom 
    for (int i = 0; i < n_candidates; ++i)
        REQUIRE((r_new(i, Eigen::seqN(0, 3)) - coords.row(9)).norm() < fene_params["R0"]); 

    // Check the reptation non-bonded energy difference
    for (int i = 0; i < n_candidates; ++i)
    {
        Matrix<double, Dynamic, 3> segment_i(n_reptate, 3); 
        for (int j = 0; j < n_reptate; ++j)
            segment_i.row(j) = r_new(i, Eigen::seqN(3 * j, 3)); 
        REQUIRE_THAT(
            energy_diffs(i),
            Catch::Matchers::WithinAbs(
                config.getMultimerReptationNonbondedEnergyDifference(
                    ReptationDirection::TAIL, segment_i, lj_params,
                    neighbor_threshold
                ), 
                tol
            )  
        );
    }
}

/**
 * Tests for multimer reptation in the PolymerCBMCSampler class via moveOnce(). 
 */
TEST_CASE("Tests for multimer reptation", "[moveOnce()]")
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

    // Try reptating by choosing from 50 3-mer reptation candidate moves 
    int n_candidates = 50;
    int n_reptate = 3;  
    PolymerConfiguration<double> config_reptated(config);  
    auto result = sampler_cosine.moveOnce(
        n_candidates, CBMCMoveType::MULTIMER_REPTATION, n_reptate, {}
    ); 
    Matrix<double, Dynamic, Dynamic> forward_moves = std::get<0>(result); 
    Matrix<double, Dynamic, Dynamic> reverse_moves = std::get<1>(result); 
    int move_idx = std::get<2>(result); 
    double prob_accept = std::get<3>(result); 
    CBMCMoveResult accepted_move = std::get<4>(result);
    ReptationDirection direction = static_cast<ReptationDirection>(
        std::get<5>(result).at("direction")
    );
    REQUIRE(forward_moves.rows() == n_candidates);
    REQUIRE(forward_moves.cols() == 3 * n_reptate);  
    REQUIRE(reverse_moves.rows() == n_candidates);
    REQUIRE(reverse_moves.cols() == 3 * n_reptate);
    REQUIRE((move_idx >= 0 && move_idx < n_candidates));
    REQUIRE((prob_accept >= 0 && prob_accept <= 1));
    REQUIRE(accepted_move != CBMCMoveResult::NONE); 

    // Generate the reptated configuration (in case the reptation move was not 
    // accepted) 
    Matrix<double, Dynamic, 3> chosen_segment(n_reptate, 3); 
    for (int i = 0; i < n_reptate; ++i)
        chosen_segment.row(i) = forward_moves(move_idx, Eigen::seqN(3 * i, 3)); 
    if (direction == ReptationDirection::HEAD)
        config_reptated.reptateTowardsHeadMultimer(chosen_segment); 
    else 
        config_reptated.reptateTowardsTailMultimer(chosen_segment); 

    // Check that the terminal atom at the appropriate end of each segment 
    // has a valid distance to the corresponding terminal atom in the original
    // configuration
    for (int i = 0; i < n_candidates; ++i)
    {
        if (direction == ReptationDirection::HEAD)
        {
            int term_idx1 = n_reptate - 1;
            int term_idx2 = 0;
            Matrix<double, 3, 1> r1 = forward_moves(i, Eigen::seqN(3 * term_idx1, 3)); 
            Matrix<double, 3, 1> r2 = coords.row(term_idx2); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }
        else 
        {
            int term_idx1 = 0;
            int term_idx2 = 9;
            Matrix<double, 3, 1> r1 = forward_moves(i, Eigen::seqN(3 * term_idx1, 3)); 
            Matrix<double, 3, 1> r2 = coords.row(term_idx2); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }
    }

    // Do the same for the reverse moves 
    for (int i = 0; i < n_candidates; ++i)
    {
        if (direction == ReptationDirection::HEAD)   // Reverse moves are at the tail
        {
            int term_idx1 = 0; 
            int term_idx2 = 9 - n_reptate;    // New terminal index after reptation towards the head
            Matrix<double, 3, 1> r1 = reverse_moves(i, Eigen::seqN(3 * term_idx1, 3)); 
            Matrix<double, 3, 1> r2 = coords.row(term_idx2); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }
        else                                         // Reverse moves are at the head  
        {
            int term_idx1 = n_reptate - 1;
            int term_idx2 = n_reptate;        // New terminal index after reptation towards the tail
            Matrix<double, 3, 1> r1 = reverse_moves(i, Eigen::seqN(3 * term_idx1, 3)); 
            Matrix<double, 3, 1> r2 = coords.row(term_idx2); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }
    }
    
    // Check that the reverse move corresponding to the chosen move is
    // reversion to the original configuration
    Matrix<double, Dynamic, 3> segment_reversion(n_reptate, 3);
    for (int i = 0; i < n_reptate; ++i)
        segment_reversion.row(i) = reverse_moves(0, Eigen::seqN(3 * i, 3));  
    if (direction == ReptationDirection::HEAD)
        REQUIRE_THAT(
            (segment_reversion - coords(Eigen::seqN(9 - n_reptate, n_reptate), Eigen::all)).norm(), 
            Catch::Matchers::WithinAbs(0, tol)
        );
    else
        REQUIRE_THAT(
            (segment_reversion - coords(Eigen::seqN(0, n_reptate), Eigen::all)).norm(), 
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
        Matrix<double, Dynamic, 3> segment_forward(n_reptate, 3), 
                                   segment_reverse(n_reptate, 3); 
        for (int j = 0; j < n_reptate; ++j)
        {
            segment_forward.row(j) = forward_moves(i, Eigen::seqN(3 * j, 3)); 
            segment_reverse.row(j) = reverse_moves(i, Eigen::seqN(3 * j, 3)); 
        }
        double diff1 = config.getMultimerReptationNonbondedEnergyDifference(
            direction, segment_forward, lj_params, neighbor_threshold
        );
        double diff2 = config_reptated.getMultimerReptationNonbondedEnergyDifference(
            reverse_direction, segment_reverse, lj_params, neighbor_threshold
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
        REQUIRE(accepted_move == CBMCMoveResult::ACCEPT); 

    // Check that, if the chosen move was taken, the resulting configuration is
    // as expected
    Matrix<double, Dynamic, 3> coords_reptated = config_reptated.getSegment(0, 10); 
    Matrix<double, Dynamic, 3> coords_result = sampler_cosine.getCoords();  
    if (accepted_move == CBMCMoveResult::ACCEPT)
    {
        // Move was taken 
        for (int i = 0; i < 10; ++i)
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

    // Try reptating by choosing from 50 3-mer reptation candidate moves 
    PolymerConfiguration<double> config2_original(config2), config2_reptated(config2);  
    result = sampler_gaussian.moveOnce(
        n_candidates, CBMCMoveType::MULTIMER_REPTATION, n_reptate, {}
    ); 
    forward_moves = std::get<0>(result); 
    reverse_moves = std::get<1>(result); 
    move_idx = std::get<2>(result); 
    prob_accept = std::get<3>(result); 
    accepted_move = std::get<4>(result);
    direction = static_cast<ReptationDirection>(std::get<5>(result).at("direction"));
    REQUIRE(forward_moves.rows() == n_candidates);
    REQUIRE(forward_moves.cols() == 3 * n_reptate);  
    REQUIRE(reverse_moves.rows() == n_candidates);
    REQUIRE(reverse_moves.cols() == 3 * n_reptate); 
    REQUIRE((move_idx >= 0 && move_idx < n_candidates));
    REQUIRE((prob_accept >= 0 && prob_accept <= 1)); 
    REQUIRE(accepted_move != CBMCMoveResult::NONE); 

    // Generate the reptated configuration (in case the reptation move was not 
    // accepted)
    chosen_segment = Matrix<double, Dynamic, 3>::Zero(n_reptate, 3); 
    for (int i = 0; i < n_reptate; ++i)
        chosen_segment.row(i) = forward_moves(move_idx, Eigen::seqN(3 * i, 3)); 
    if (direction == ReptationDirection::HEAD)
        config2_reptated.reptateTowardsHeadMultimer(chosen_segment); 
    else 
        config2_reptated.reptateTowardsTailMultimer(chosen_segment);

    // Check that the terminal atom at the appropriate end of each segment 
    // has a valid distance to the corresponding terminal atom in the original
    // configuration
    for (int i = 0; i < n_candidates; ++i)
    {
        if (direction == ReptationDirection::HEAD)
        {
            int term_idx1 = n_reptate - 1;
            int term_idx2 = 0;
            Matrix<double, 3, 1> r1 = forward_moves(i, Eigen::seqN(3 * term_idx1, 3)); 
            Matrix<double, 3, 1> r2 = coords2.row(term_idx2); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }
        else 
        {
            int term_idx1 = 0;
            int term_idx2 = 9;
            Matrix<double, 3, 1> r1 = forward_moves(i, Eigen::seqN(3 * term_idx1, 3)); 
            Matrix<double, 3, 1> r2 = coords2.row(term_idx2); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }
    }

    // Do the same for the reverse moves 
    for (int i = 0; i < n_candidates; ++i)
    {
        if (direction == ReptationDirection::HEAD)   // Reverse moves are at the tail
        {
            int term_idx1 = 0; 
            int term_idx2 = 9 - n_reptate;    // New terminal index after reptation towards the head
            Matrix<double, 3, 1> r1 = reverse_moves(i, Eigen::seqN(3 * term_idx1, 3)); 
            Matrix<double, 3, 1> r2 = coords2.row(term_idx2); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }
        else                                         // Reverse moves are at the head  
        {
            int term_idx1 = n_reptate - 1;
            int term_idx2 = n_reptate;        // New terminal index after reptation towards the tail
            Matrix<double, 3, 1> r1 = reverse_moves(i, Eigen::seqN(3 * term_idx1, 3)); 
            Matrix<double, 3, 1> r2 = coords2.row(term_idx2); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }
    }

    // Check that the reverse move corresponding to the chosen move is
    // reversion to the original configuration
    segment_reversion = Matrix<double, Dynamic, 3>::Zero(n_reptate, 3);
    for (int i = 0; i < n_reptate; ++i)
        segment_reversion.row(i) = reverse_moves(0, Eigen::seqN(3 * i, 3));  
    if (direction == ReptationDirection::HEAD)
        REQUIRE_THAT(
            (segment_reversion - coords2(Eigen::seqN(9 - n_reptate, n_reptate), Eigen::all)).norm(), 
            Catch::Matchers::WithinAbs(0, tol)
        ); 
    else 
        REQUIRE_THAT(
            (segment_reversion - coords2(Eigen::seqN(0, n_reptate), Eigen::all)).norm(), 
            Catch::Matchers::WithinAbs(0, tol)
        );

    // Re-calculate the Rosenbluth weights ...
    reverse_direction = (
        direction == ReptationDirection::HEAD ? ReptationDirection::TAIL
        : ReptationDirection::HEAD
    ); 
    for (int i = 0; i < n_candidates; ++i)
    {
        Matrix<double, Dynamic, 3> segment_forward(n_reptate, 3), 
                                   segment_reverse(n_reptate, 3); 
        for (int j = 0; j < n_reptate; ++j)
        {
            segment_forward.row(j) = forward_moves(i, Eigen::seqN(3 * j, 3)); 
            segment_reverse.row(j) = reverse_moves(i, Eigen::seqN(3 * j, 3)); 
        }
        double diff1 = config2.getMultimerReptationNonbondedEnergyDifference(
            direction, segment_forward, lj_params, neighbor_threshold
        );
        double diff2 = config2_reptated.getMultimerReptationNonbondedEnergyDifference(
            reverse_direction, segment_reverse, lj_params, neighbor_threshold
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
        REQUIRE(accepted_move == CBMCMoveResult::ACCEPT); 

    // Check that, if the chosen move was taken, the resulting configuration is
    // as expected
    Matrix<double, Dynamic, 3> coords2_reptated = config2_reptated.getSegment(0, 10);
    Matrix<double, Dynamic, 3> coords2_result = sampler_gaussian.getCoords(); 
    if (accepted_move == CBMCMoveResult::ACCEPT)
    {
        // Move was taken 
        for (int i = 0; i < 10; ++i)
        {
            REQUIRE_THAT(
                (coords2_result.row(i) - coords2_reptated.row(i)).norm(),
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

