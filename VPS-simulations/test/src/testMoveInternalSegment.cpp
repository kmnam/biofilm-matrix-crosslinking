/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     2/11/2026
 */

#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/random.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/utils.hpp"
#include "../../include/polymerConfiguration.hpp"
#include "../../include/cbmc.hpp"

using std::isinf; 

using namespace Eigen;

/**
 * Tests for PolymerCBMCSampler::generateInternalSegmentMoves(). 
 */
TEST_CASE("Tests for internal segment move generation", "[generateInternalSegmentMoves()]")
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
    
    // Try generating 50 4-atom internal segment moves for the segment
    // [3, 4, 5, 6], using precisely 50 attempts 
    int n_attempts = 50; 
    double tangent_stepsize = 0.1; 
    auto result = sampler_cosine.generateInternalSegmentMoves(
        n_attempts, n_attempts, 4, 3, tangent_stepsize,
        InternalMoveGenerationMode::FIXED_ATTEMPTS, true
    );
    Matrix<double, Dynamic, Dynamic> r_new = result.first; 
    Matrix<double, Dynamic, 1> energy_diffs = result.second;
    REQUIRE(r_new.rows() <= n_attempts); 
    REQUIRE(r_new.cols() == 12);  
    REQUIRE(energy_diffs.size() == r_new.rows());

    // Check that each new segment is correctly clamped at the endpoints 
    for (int i = 0; i < r_new.rows(); ++i)
    {
        Matrix<double, 3, 1> r1 = r_new(i, Eigen::seqN(0, 3)); 
        Matrix<double, 3, 1> r2 = r_new(i, Eigen::seqN(9, 3));
        REQUIRE_THAT((r1.transpose() - coords.row(3)).norm(), Catch::Matchers::WithinAbs(0, tol)); 
        REQUIRE_THAT((r2.transpose() - coords.row(6)).norm(), Catch::Matchers::WithinAbs(0, tol));
    }

    // Check that each new segment differs from the original segment along
    // the interior 
    int n_moves_match = 0; 
    for (int i = 0; i < r_new.rows(); ++i)
    {
        bool found_mismatch = false; 
        for (int j = 1; j < 3; ++j)
        {
            Matrix<double, 3, 1> r1 = r_new(i, Eigen::seqN(3 * j, 3));
            Matrix<double, 3, 1> r2 = coords.row(3 + j); 
            if ((r1 - r2).norm() > tol)
            {
                found_mismatch = true; 
                break;
            } 
        }
        if (!found_mismatch)
            n_moves_match++; 
    }
    REQUIRE(n_moves_match == 0); 

    // Check that, for any new segment along which there is a bond that is 
    // too long, the corresponding energy difference is infinite
    Matrix<int, Dynamic, 1> found_long_bond = Matrix<int, Dynamic, 1>::Zero(r_new.rows()); 
    for (int i = 0; i < r_new.rows(); ++i)
    {
        for (int j = 0; j < 3; ++j)     // 3 bonds to check 
        {
            Matrix<double, 3, 1> r1 = r_new(i, Eigen::seqN(3 * j, 3));
            Matrix<double, 3, 1> r2 = r_new(i, Eigen::seqN(3 * (j + 1), 3)); 
            if ((r2 - r1).norm() >= fene_params["R0"])
            {
                found_long_bond(i) = 1; 
                break;  
            }
        }
        if (found_long_bond(i))
            REQUIRE(isinf(energy_diffs(i)));
        else 
            REQUIRE(!isinf(energy_diffs(i))); 
    } 

    // Check the (finite) energy differences
    for (int i = 0; i < r_new.rows(); ++i)
    {
        if (!found_long_bond(i))
        {
            Matrix<double, Dynamic, 3> segment_i(4, 3); 
            for (int j = 0; j < 4; ++j)
                segment_i.row(j) = r_new(i, Eigen::seqN(3 * j, 3)); 
            REQUIRE_THAT(
                energy_diffs(i),
                Catch::Matchers::WithinAbs(
                    config.getSegmentReplacementEnergyDifference(
                        segment_i, 3, lj_params, neighbor_threshold,
                        fene_params, AngleMode::COSINE, cosine_params, 
                        dihedral_params
                    ), 
                    tol
                )  
            );
        }
    }

    // Try generating 50 4-atom internal segment moves for the segment
    // [3, 4, 5, 6], using 100 attempts 
    n_attempts = 100;
    int n_candidates = 50; 
    result = sampler_cosine.generateInternalSegmentMoves(
        n_attempts, n_candidates, 4, 3, tangent_stepsize,
        InternalMoveGenerationMode::FIXED_CANDIDATES, true
    );
    r_new = result.first; 
    energy_diffs = result.second;
    REQUIRE(r_new.rows() <= n_candidates); 
    REQUIRE(r_new.cols() == 12);  
    REQUIRE(energy_diffs.size() == r_new.rows());

    // Check that each new segment is correctly clamped at the endpoints 
    for (int i = 0; i < r_new.rows(); ++i)
    {
        Matrix<double, 3, 1> r1 = r_new(i, Eigen::seqN(0, 3)); 
        Matrix<double, 3, 1> r2 = r_new(i, Eigen::seqN(9, 3));
        REQUIRE_THAT((r1.transpose() - coords.row(3)).norm(), Catch::Matchers::WithinAbs(0, tol)); 
        REQUIRE_THAT((r2.transpose() - coords.row(6)).norm(), Catch::Matchers::WithinAbs(0, tol));
    }

    // Check that each new segment differs from the original segment along
    // the interior 
    n_moves_match = 0; 
    for (int i = 0; i < r_new.rows(); ++i)
    {
        bool found_mismatch = false; 
        for (int j = 1; j < 3; ++j)
        {
            Matrix<double, 3, 1> r1 = r_new(i, Eigen::seqN(3 * j, 3));
            Matrix<double, 3, 1> r2 = coords.row(3 + j); 
            if ((r1 - r2).norm() > tol)
            {
                found_mismatch = true; 
                break;
            } 
        }
        if (!found_mismatch)
            n_moves_match++; 
    }
    REQUIRE(n_moves_match == 0); 

    // Check that, for any new segment along which there is a bond that is 
    // too long, the corresponding energy difference is infinite
    found_long_bond = Matrix<int, Dynamic, 1>::Zero(r_new.rows()); 
    for (int i = 0; i < r_new.rows(); ++i)
    {
        for (int j = 0; j < 3; ++j)     // 3 bonds to check 
        {
            Matrix<double, 3, 1> r1 = r_new(i, Eigen::seqN(3 * j, 3));
            Matrix<double, 3, 1> r2 = r_new(i, Eigen::seqN(3 * (j + 1), 3)); 
            if ((r2 - r1).norm() >= fene_params["R0"])
            {
                found_long_bond(i) = 1; 
                break;  
            }
        }
        if (found_long_bond(i))
            REQUIRE(isinf(energy_diffs(i)));
        else 
            REQUIRE(!isinf(energy_diffs(i))); 
    } 

    // Check the (finite) energy differences
    for (int i = 0; i < r_new.rows(); ++i)
    {
        if (!found_long_bond(i))
        {
            Matrix<double, Dynamic, 3> segment_i(4, 3); 
            for (int j = 0; j < 4; ++j)
                segment_i.row(j) = r_new(i, Eigen::seqN(3 * j, 3)); 
            REQUIRE_THAT(
                energy_diffs(i),
                Catch::Matchers::WithinAbs(
                    config.getSegmentReplacementEnergyDifference(
                        segment_i, 3, lj_params, neighbor_threshold,
                        fene_params, AngleMode::COSINE, cosine_params, 
                        dihedral_params
                    ), 
                    tol
                )  
            );
        }
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

    // Try generating 50 5-atom internal segment moves for the segment
    // [1, 2, 3, 4, 5] with a new seed (for later comparison)
    sampler_gaussian.seed(1234567890); 
    result = sampler_gaussian.generateInternalSegmentMoves(
        n_attempts, n_attempts, 5, 1, tangent_stepsize, 
        InternalMoveGenerationMode::FIXED_ATTEMPTS, true
    );
    r_new = result.first; 
    energy_diffs = result.second;
    REQUIRE(r_new.rows() <= n_attempts); 
    REQUIRE(r_new.cols() == 15);  
    REQUIRE(energy_diffs.size() == r_new.rows()); 

    // Check that each new segment is correctly clamped at the endpoints 
    for (int i = 0; i < r_new.rows(); ++i)
    {
        Matrix<double, 3, 1> r1 = r_new(i, Eigen::seqN(0, 3)); 
        Matrix<double, 3, 1> r2 = r_new(i, Eigen::seqN(12, 3));
        REQUIRE_THAT((r1.transpose() - coords.row(1)).norm(), Catch::Matchers::WithinAbs(0, tol)); 
        REQUIRE_THAT((r2.transpose() - coords.row(5)).norm(), Catch::Matchers::WithinAbs(0, tol));
    }

    // Check that each new segment differs from the original segment along
    // the interior 
    n_moves_match = 0; 
    for (int i = 0; i < r_new.rows(); ++i)
    {
        bool found_mismatch = false; 
        for (int j = 1; j < 3; ++j)
        {
            Matrix<double, 3, 1> r1 = r_new(i, Eigen::seqN(3 * j, 3));
            Matrix<double, 3, 1> r2 = coords.row(3 + j); 
            if ((r1 - r2).norm() > tol)
            {
                found_mismatch = true; 
                break;
            } 
        }
        if (!found_mismatch)
            n_moves_match++; 
    }
    REQUIRE(n_moves_match == 0); 

    // Check that, for any new segment along which there is a bond that is 
    // too long, the corresponding energy difference is infinite
    found_long_bond = Matrix<int, Dynamic, 1>::Zero(r_new.rows()); 
    for (int i = 0; i < r_new.rows(); ++i)
    {
        for (int j = 0; j < 4; ++j)    // 4 bonds to check
        {
            Matrix<double, 3, 1> r1 = r_new(i, Eigen::seqN(3 * j, 3));
            Matrix<double, 3, 1> r2 = r_new(i, Eigen::seqN(3 * (j + 1), 3)); 
            if ((r2 - r1).norm() >= fene_params["R0"])
            {
                found_long_bond(i) = 1; 
                break;  
            }
        }
        if (found_long_bond(i))
            REQUIRE(isinf(energy_diffs(i)));
        else 
            REQUIRE(!isinf(energy_diffs(i))); 
    } 

    // Check the (finite) energy differences
    for (int i = 0; i < r_new.rows(); ++i)
    {
        if (!found_long_bond(i))
        {
            Matrix<double, Dynamic, 3> segment_i(5, 3); 
            for (int j = 0; j < 5; ++j)
                segment_i.row(j) = r_new(i, Eigen::seqN(3 * j, 3)); 
            REQUIRE_THAT(
                energy_diffs(i),
                Catch::Matchers::WithinAbs(
                    config.getSegmentReplacementEnergyDifference(
                        segment_i, 1, lj_params, neighbor_threshold,
                        fene_params, AngleMode::GAUSSIAN, gaussian_params, 
                        dihedral_params
                    ), 
                    tol
                )  
            );
        }
    }

    // Try generating a new set of 50 5-atom internal segment moves for the
    // segment [1, 2, 3, 4, 5] with the same seed, but with a smaller initial
    // tangent stepsize 
    sampler_gaussian.seed(1234567890); 
    result = sampler_gaussian.generateInternalSegmentMoves(
        n_attempts, n_attempts, 5, 1, tangent_stepsize / 2,
        InternalMoveGenerationMode::FIXED_ATTEMPTS, true
    );
    Matrix<double, Dynamic, Dynamic> r_new_small = result.first; 
    Matrix<double, Dynamic, 1> energy_diffs_small = result.second;
    REQUIRE(r_new_small.rows() <= n_attempts);
    REQUIRE(r_new_small.cols() == 15);  
    REQUIRE(energy_diffs_small.size() == r_new_small.rows()); 

    // Check that each new segment is correctly clamped at the endpoints 
    for (int i = 0; i < r_new_small.rows(); ++i)
    {
        Matrix<double, 3, 1> r1 = r_new_small(i, Eigen::seqN(0, 3)); 
        Matrix<double, 3, 1> r2 = r_new_small(i, Eigen::seqN(12, 3));
        REQUIRE_THAT((r1.transpose() - coords.row(1)).norm(), Catch::Matchers::WithinAbs(0, tol)); 
        REQUIRE_THAT((r2.transpose() - coords.row(5)).norm(), Catch::Matchers::WithinAbs(0, tol));
    }

    // Check that each new segment differs from the original segment along
    // the interior 
    int n_moves_match_small = 0; 
    for (int i = 0; i < r_new.rows(); ++i)
    {
        bool found_mismatch = false; 
        for (int j = 1; j < 3; ++j)
        {
            Matrix<double, 3, 1> r1 = r_new_small(i, Eigen::seqN(3 * j, 3));
            Matrix<double, 3, 1> r2 = coords.row(3 + j); 
            if ((r1 - r2).norm() > tol)
            {
                found_mismatch = true; 
                break;
            } 
        }
        if (!found_mismatch)
            n_moves_match_small++; 
    }
    REQUIRE(n_moves_match_small == 0); 

    // Check that the total perturbation of each segment is smaller, given 
    // that every perturbation was successful for both stepsizes
    if (r_new.rows() == n_attempts && r_new_small.rows() == n_attempts)
    {
        for (int i = 0; i < n_attempts; ++i)
        {
            // Get the sum of the perturbations of all atoms along
            // each segment 
            double sum1 = 0; 
            double sum2 = 0;
            for (int j = 0; j < 5; ++j)
            {
                Matrix<double, 3, 1> r1 = coords.row(1 + j); 
                Matrix<double, 3, 1> r2 = r_new(i, Eigen::seqN(3 * j, 3)); 
                Matrix<double, 3, 1> r3 = r_new_small(i, Eigen::seqN(3 * j, 3)); 
                sum1 += (r1 - r2).norm(); 
                sum2 += (r1 - r3).norm(); 
            }
            REQUIRE(sum1 >= sum2); 
        }
    }
}

/**
 * Tests for internal segment moves in the PolymerCBMCSampler class via 
 * moveOnce(). 
 */
TEST_CASE("Tests for internal segment moves", "[moveOnce()]")
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

    // Try moving a randomly chosen 4-atom internal segment by choosing from 
    // 50 candidate moves 
    int n_attempts = 50; 
    std::unordered_map<std::string, double> internal_move_params; 
    internal_move_params["tangent_stepsize"] = 0.1;
    internal_move_params["mode"] = 0;  
    PolymerConfiguration<double> config_moved(config);
    auto result_cosine = sampler_cosine.moveOnce(
        n_attempts, CBMCMoveType::INTERNAL_SEGMENT, 4, internal_move_params
    ); 
    Matrix<double, Dynamic, Dynamic> forward_moves = std::get<0>(result_cosine); 
    Matrix<double, Dynamic, Dynamic> reverse_moves = std::get<1>(result_cosine); 
    int move_idx = std::get<2>(result_cosine); 
    double prob_accept = std::get<3>(result_cosine);
    CBMCMoveResult accepted_move = std::get<4>(result_cosine);
    int segment_idx = static_cast<int>(std::get<5>(result_cosine).at("segment_idx"));
    bool proposed_new_move = static_cast<bool>(
        std::get<5>(result_cosine).at("proposed_new_move")
    );

    // Check that the two sets of moves are nonempty 
    REQUIRE(forward_moves.rows() > 0); 
    REQUIRE(forward_moves.rows() <= n_attempts);
    REQUIRE(reverse_moves.rows() > 0); 
    REQUIRE(reverse_moves.rows() <= n_attempts); 
    if (!proposed_new_move)
    {
        REQUIRE(forward_moves.rows() == 1);
        REQUIRE(reverse_moves.rows() == 1);
    }
    REQUIRE(forward_moves.cols() == 12); 
    REQUIRE(reverse_moves.cols() == 12); 

    // If only the null move was returned, then the move index and acceptance 
    // probability are fixed
    if (!proposed_new_move)
    {
        REQUIRE(move_idx == 0); 
        REQUIRE(prob_accept == 1);
    }
    else 
    { 
        REQUIRE((move_idx >= 0 && move_idx < forward_moves.rows())); 
        REQUIRE((prob_accept >= 0 && prob_accept <= 1));
    }

    // Check the chosen segment index 
    REQUIRE((segment_idx >= 1 && segment_idx <= 5)); 

    // If only the null move was returned, check the forward and reverse moves
    if (!proposed_new_move)
    {
        for (int i = 0; i < 4; ++i)
        {
            Matrix<double, 3, 1> r1 = forward_moves(0, Eigen::seqN(3 * i, 3)); 
            Matrix<double, 3, 1> r2 = reverse_moves(0, Eigen::seqN(3 * i, 3)); 
            REQUIRE_THAT(
                (r1.transpose() - coords.row(segment_idx + i)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
            REQUIRE_THAT(
                (r2.transpose() - coords.row(segment_idx + i)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        }
    }
    // If non-null moves were returned, check that each forward and reverse
    // move introduces a segment configuration that is correctly clamped at
    // the endpoints
    else 
    { 
        for (int i = 0; i < forward_moves.rows(); ++i)
        {
            Matrix<double, 3, 1> r1 = forward_moves(i, Eigen::seqN(0, 3)); 
            Matrix<double, 3, 1> r2 = forward_moves(i, Eigen::seqN(9, 3));
            REQUIRE_THAT(
                (r1.transpose() - coords.row(segment_idx)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
            REQUIRE_THAT(
                (r2.transpose() - coords.row(segment_idx + 3)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            );
        }
        for (int i = 0; i < reverse_moves.rows(); ++i)
        {
            Matrix<double, 3, 1> r3 = reverse_moves(i, Eigen::seqN(0, 3)); 
            Matrix<double, 3, 1> r4 = reverse_moves(i, Eigen::seqN(9, 3));
            REQUIRE_THAT(
                (r3.transpose() - coords.row(segment_idx)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
            REQUIRE_THAT(
                (r4.transpose() - coords.row(segment_idx + 3)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            );
        }
    }

    // Now, assume that non-null moves were proposed ... 
    if (proposed_new_move)
    {
        // Generate the modified configuration (in case the move was not accepted)
        Matrix<double, Dynamic, 3> move_segment(4, 3); 
        for (int i = 0; i < 4; ++i)
            move_segment.row(i) = forward_moves(move_idx, Eigen::seqN(3 * i, 3)); 
        config_moved.replaceSegment(move_segment, segment_idx);

        // If the chosen candidate move incurs an infinite energy change, that 
        // means that every candidate move must do the same
        Matrix<double, Dynamic, 1> forward_diffs(forward_moves.rows()); 
        for (int i = 0; i < forward_moves.rows(); ++i)
        { 
            Matrix<double, Dynamic, 3> move_segment_i(4, 3); 
            for (int j = 0; j < 4; ++j)
                move_segment_i.row(j) = forward_moves(i, Eigen::seqN(3 * j, 3)); 
            forward_diffs(i) = config.getSegmentReplacementEnergyDifference(
                move_segment_i, segment_idx, lj_params, neighbor_threshold,
                fene_params, AngleMode::COSINE, cosine_params, dihedral_params
            ); 
        } 
        if (isinf(forward_diffs(move_idx)))
            REQUIRE(forward_diffs.array().isInf().all()); 

        // Check that each new atom has a valid distance to the terminal atom 
        // at the appropriate end, unless the corresponding energy difference 
        // is infinite  
        REQUIRE_THAT(
            (move_segment.row(0) - coords.row(segment_idx)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
        if (!isinf(forward_diffs(move_idx)))
        {
            // Check bond lengths along the segment 
            REQUIRE((move_segment.row(1) - move_segment.row(0)).norm() < fene_params["R0"]);
            REQUIRE((move_segment.row(2) - move_segment.row(1)).norm() < fene_params["R0"]); 
            REQUIRE((move_segment.row(3) - move_segment.row(2)).norm() < fene_params["R0"]); 
        } 
        REQUIRE_THAT(
            (move_segment.row(3) - coords.row(segment_idx + 3)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );

        // Check that the 0-th reverse move is reversion to the original
        // configuration
        for (int i = 0; i < 4; ++i)
        {
            Matrix<double, 3, 1> r1 = reverse_moves(0, Eigen::seqN(3 * i, 3));
            Matrix<double, 3, 1> r2 = coords.row(segment_idx + i);  
            REQUIRE_THAT((r1 - r2).norm(), Catch::Matchers::WithinAbs(0, tol));
        }

        // Re-calculate the Rosenbluth weights ...
        Matrix<double, Dynamic, 1> weights_forward(forward_moves.rows()), 
                                   weights_reverse(reverse_moves.rows()); 
        for (int i = 0; i < forward_moves.rows(); ++i)
        {
            Matrix<double, Dynamic, 3> forward_segment(4, 3);
            for (int j = 0; j < 4; ++j)
                forward_segment.row(j) = forward_moves(i, Eigen::seqN(3 * j, 3));
            double diff = config.getSegmentReplacementEnergyDifference(
                forward_segment, segment_idx, lj_params, neighbor_threshold,
                fene_params, AngleMode::COSINE, cosine_params, dihedral_params
            );
            weights_forward(i) = exp(-diff / kT);
        }
        for (int i = 0; i < reverse_moves.rows(); ++i)
        {
            Matrix<double, Dynamic, 3> reverse_segment(4, 3);
            for (int j = 0; j < 4; ++j)
                reverse_segment.row(j) = reverse_moves(i, Eigen::seqN(3 * j, 3));
            double diff = config_moved.getSegmentReplacementEnergyDifference(
                reverse_segment, segment_idx, lj_params, neighbor_threshold,
                fene_params, AngleMode::COSINE, cosine_params, dihedral_params
            );
            weights_reverse(i) = exp(-diff / kT);
        }
        double forward_rosenbluth = weights_forward.sum(); 
        double reverse_rosenbluth = weights_reverse.sum();
        REQUIRE_THAT(
            weights_forward(move_idx) * weights_reverse(0),
            Catch::Matchers::WithinAbs(1.0, tol)
        ); 

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

    // Try moving a randomly chosen 5-atom internal segment by choosing from 
    // 50 candidate moves 
    PolymerConfiguration<double> config2_moved(config2);
    auto result_gaussian = sampler_gaussian.moveOnce(
        n_attempts, CBMCMoveType::INTERNAL_SEGMENT, 5, internal_move_params
    ); 
    forward_moves = std::get<0>(result_gaussian);
    reverse_moves = std::get<1>(result_gaussian);
    move_idx = std::get<2>(result_gaussian); 
    prob_accept = std::get<3>(result_gaussian);
    accepted_move = std::get<4>(result_gaussian);
    segment_idx = static_cast<int>(std::get<5>(result_gaussian).at("segment_idx"));
    proposed_new_move = static_cast<bool>(
        std::get<5>(result_gaussian).at("proposed_new_move")
    ); 

    // Check that the two sets of moves are nonempty 
    REQUIRE(forward_moves.rows() > 0); 
    REQUIRE(forward_moves.rows() <= n_attempts);
    REQUIRE(reverse_moves.rows() > 0); 
    REQUIRE(reverse_moves.rows() <= n_attempts); 
    if (!proposed_new_move)
    {
        REQUIRE(forward_moves.rows() == 1);
        REQUIRE(reverse_moves.rows() == 1);
    }
    REQUIRE(forward_moves.rows() == reverse_moves.rows());

    // If only the null move was returned, then the move index and acceptance 
    // probability are fixed
    if (!proposed_new_move)
    {
        REQUIRE(move_idx == 0); 
        REQUIRE(prob_accept == 1); 
    }
    else 
    { 
        REQUIRE((move_idx >= 0 && move_idx < forward_moves.rows())); 
        REQUIRE((prob_accept >= 0 && prob_accept <= 1));
    }

    // Check the chosen segment index 
    REQUIRE((segment_idx >= 1 && segment_idx <= 4));

    // If only the null move was returned, check the forward and reverse moves
    if (!proposed_new_move)
    {
        for (int i = 0; i < 4; ++i)
        {
            Matrix<double, 3, 1> r1 = forward_moves(0, Eigen::seqN(3 * i, 3)); 
            Matrix<double, 3, 1> r2 = reverse_moves(0, Eigen::seqN(3 * i, 3)); 
            REQUIRE_THAT(
                (r1.transpose() - coords2.row(segment_idx + i)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
            REQUIRE_THAT(
                (r2.transpose() - coords2.row(segment_idx + i)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        }
    }
    // If non-null moves were returned, check that each forward and reverse
    // move introduces a segment configuration that is correctly clamped at
    // the endpoints
    else 
    { 
        for (int i = 0; i < forward_moves.rows(); ++i)
        {
            Matrix<double, 3, 1> r1 = forward_moves(i, Eigen::seqN(0, 3)); 
            Matrix<double, 3, 1> r2 = forward_moves(i, Eigen::seqN(12, 3));
            REQUIRE_THAT(
                (r1.transpose() - coords2.row(segment_idx)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
            REQUIRE_THAT(
                (r2.transpose() - coords2.row(segment_idx + 4)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            );
        }
        for (int i = 0; i < reverse_moves.rows(); ++i)
        {
            Matrix<double, 3, 1> r3 = reverse_moves(i, Eigen::seqN(0, 3)); 
            Matrix<double, 3, 1> r4 = reverse_moves(i, Eigen::seqN(12, 3));
            REQUIRE_THAT(
                (r3.transpose() - coords2.row(segment_idx)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
            REQUIRE_THAT(
                (r4.transpose() - coords2.row(segment_idx + 4)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            );
        }
    }

    // Now, assume that non-null moves were proposed ... 
    if (proposed_new_move)
    {
        // Generate the modified configuration (in case the move was not accepted)
        Matrix<double, Dynamic, 3> move_segment(5, 3); 
        for (int i = 0; i < 5; ++i)
            move_segment.row(i) = forward_moves(move_idx, Eigen::seqN(3 * i, 3)); 
        config2_moved.replaceSegment(move_segment, segment_idx);

        // If the chosen candidate move incurs an infinite energy change, that 
        // means that every candidate move must do the same
        Matrix<double, Dynamic, 1> forward_diffs(forward_moves.rows()); 
        for (int i = 0; i < forward_moves.rows(); ++i)
        { 
            Matrix<double, Dynamic, 3> move_segment_i(5, 3); 
            for (int j = 0; j < 5; ++j)
                move_segment_i.row(j) = forward_moves(i, Eigen::seqN(3 * j, 3)); 
            forward_diffs(i) = config2.getSegmentReplacementEnergyDifference(
                move_segment_i, segment_idx, lj_params, neighbor_threshold,
                fene_params, AngleMode::GAUSSIAN, gaussian_params, dihedral_params
            ); 
        } 
        if (isinf(forward_diffs(move_idx)))
            REQUIRE(forward_diffs.array().isInf().all()); 

        // Check that each new atom has a valid distance to the terminal atom 
        // at the appropriate end, unless the corresponding energy difference 
        // is infinite  
        REQUIRE_THAT(
            (move_segment.row(0) - coords2.row(segment_idx)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
        if (!isinf(forward_diffs(move_idx)))
        {
            // Check bond lengths along the segment 
            REQUIRE((move_segment.row(1) - move_segment.row(0)).norm() < fene_params["R0"]);
            REQUIRE((move_segment.row(2) - move_segment.row(1)).norm() < fene_params["R0"]); 
            REQUIRE((move_segment.row(3) - move_segment.row(2)).norm() < fene_params["R0"]);
            REQUIRE((move_segment.row(4) - move_segment.row(3)).norm() < fene_params["R0"]);  
        } 
        REQUIRE_THAT(
            (move_segment.row(4) - coords2.row(segment_idx + 4)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );

        // Check that the 0-th reverse move is reversion to the original
        // configuration
        for (int i = 0; i < 5; ++i)
        {
            Matrix<double, 3, 1> r1 = reverse_moves(0, Eigen::seqN(3 * i, 3));
            Matrix<double, 3, 1> r2 = coords2.row(segment_idx + i);  
            REQUIRE_THAT((r1 - r2).norm(), Catch::Matchers::WithinAbs(0, tol));
        } 

        // Re-calculate the Rosenbluth weights ...
        Matrix<double, Dynamic, 1> weights_forward(forward_moves.rows()), 
                                   weights_reverse(reverse_moves.rows()); 
        for (int i = 0; i < forward_moves.rows(); ++i)
        {
            Matrix<double, Dynamic, 3> forward_segment(5, 3);
            for (int j = 0; j < 5; ++j)
                forward_segment.row(j) = forward_moves(i, Eigen::seqN(3 * j, 3));
            double diff = config2.getSegmentReplacementEnergyDifference(
                forward_segment, segment_idx, lj_params, neighbor_threshold,
                fene_params, AngleMode::GAUSSIAN, gaussian_params, dihedral_params
            );
            weights_forward(i) = exp(-diff / kT);
        }
        for (int i = 0; i < reverse_moves.rows(); ++i)
        {
            Matrix<double, Dynamic, 3> reverse_segment(5, 3);
            for (int j = 0; j < 5; ++j)
                reverse_segment.row(j) = reverse_moves(i, Eigen::seqN(3 * j, 3));
            double diff = config2_moved.getSegmentReplacementEnergyDifference(
                reverse_segment, segment_idx, lj_params, neighbor_threshold,
                fene_params, AngleMode::GAUSSIAN, gaussian_params, dihedral_params
            );
            weights_reverse(i) = exp(-diff / kT);
        }
        double forward_rosenbluth = weights_forward.sum(); 
        double reverse_rosenbluth = weights_reverse.sum();
        REQUIRE_THAT(
            weights_forward(move_idx) * weights_reverse(0),
            Catch::Matchers::WithinAbs(1.0, tol)
        ); 

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
}

