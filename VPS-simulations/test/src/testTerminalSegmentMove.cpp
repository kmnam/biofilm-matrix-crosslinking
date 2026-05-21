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

/**
 * Tests for PolymerCBMCSampler::generateForwardTerminalSegmentMove(). 
 */
TEST_CASE(
    "Tests for terminal segment move generation",
    "[generateForwardTerminalSegmentMove()]"
)
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

    // Make cosine potential soft to allow for terminal segment move candidates
    // that are close to the original configuration  
    cosine_params["K"] = 0.5 * kT;
    cosine_params["theta0"] = 160 * boost::math::constants::pi<double>() / 180;

    // Similarly make Gaussian potential soft to allow for terminal segment 
    // move candidates that are close to the original configuration 
    gaussian_params["A1"] = 0.9; 
    gaussian_params["A2"] = 0.1;
    gaussian_params["w1"] = 2.0;     // Standard deviations of 1 for each component
    gaussian_params["w2"] = 2.0;
    gaussian_params["theta1"] = 160 * boost::math::constants::pi<double>() / 180; 
    gaussian_params["theta2"] = 90 * boost::math::constants::pi<double>() / 180;

    // Define dihedral potential parameters 
    nodihedral_params["K"] = 0;  
    dihedral_params["K"] = 0.5 * kT;

    // Define additional parameters for initialization and sampling
    const double collision_threshold = 0.9;    // Slightly less than 2^(1/6) * sigma ~ 1.01 
    const int max_tries_per_atom = 50;
    const int max_n_backtracks = 50;  
    Matrix<double, Dynamic, 2> bond_length_cdf = getFeneCDF<double>(
        lj_params["eps"], lj_params["sigma"], fene_params["K"], fene_params["R0"],
        kT, 10000
    );

    // --------------------------------------------------------------- //
    // Terminal segment moves on 10-mer with no angle or dihedral potentials 
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

    // Try generating a collection of 3-mer terminal segment moves at the head
    const int n_moves = 50; 
    const int n_candidates = 50;
    int segment_length = 3; 
    for (int i = 0; i < n_moves; ++i)
    {
        auto result = sampler_random.generateForwardTerminalSegmentMove(
            segment_length, TerminalSegmentEnd::HEAD, n_candidates
        );
        Matrix<double, Dynamic, Dynamic> candidates = std::get<0>(result); 
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);  
        double log_forward_rosenbluth = std::get<2>(result);
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * segment_length);  
        REQUIRE(r_new.rows() == segment_length); 

        // Check that the chosen 3-mer terminal segment move indeed features
        // among the proposed atom positions
        for (int j = 0; j < segment_length; ++j)
        {
            bool found_r_new_j = false;
            Matrix<double, 3, 1> r_new_j = r_new.row(j);  
            for (int k = 0; k < n_candidates; ++k)
            {
                if ((r_new_j.transpose() - candidates(k, Eigen::seqN(3 * j, 3))).norm() < tol)
                {
                    found_r_new_j = true; 
                    break; 
                }
            }
            REQUIRE(found_r_new_j); 
        }
        
        // Check that the bonds have valid lengths
        //
        // The first atom should be bonded to the 3rd atom in the original
        // configuration 
        REQUIRE((r_new.row(0) - coords.row(3)).norm() < fene_params["R0"]); 

        // Each subsequent pair of atoms should also be separated by valid 
        // bond lengths 
        for (int j = 1; j < segment_length; ++j)
        {
            Matrix<double, 3, 1> r1 = r_new.row(j - 1); 
            Matrix<double, 3, 1> r2 = r_new.row(j); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < segment_length; ++j)
        {
            Matrix<double, 3, 1> predecessor; 
            if (j == 0)
                predecessor = coords.row(3); 
            else 
                predecessor = r_new.row(j - 1);
            for (int k = 0; k < n_candidates; ++k)
            {
                double r = (predecessor.transpose() - candidates(k, Eigen::seqN(3 * j, 3))).norm();
                REQUIRE(r < fene_params["R0"]); 
            } 
        }

        // Check the Rosenbluth factor of the proposed move ... 
        //
        // For each proposed atom at each position, get the total non-bonded 
        // interaction energy between that atom and:
        // 1) every atom in the original configuration that survives the move
        // 2) the previous atoms along the segment
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_forward_rosenbluth_ = 0;  
        for (int j = 0; j < segment_length; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are moving the head terminal segment, run through
                // all atoms 3, ..., 9 in the original configuration, which 
                // are the atoms that survive the move 
                //
                // Omit atom 3 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = segment_length + 1; m < length; ++m)
                {
                    double r = (r_curr - coords.row(m).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    double r = (r_curr - coords.row(segment_length).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the segment (except for the immediately preceding atom)
                for (int m = 0; m < j - 1; ++m)
                {
                    double r = (r_curr - r_new.row(m).transpose()).norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Get the Boltzmann weight and increment the Rosenbluth weight 
                double boltzmann = exp(-residual_ijk / kT);
                rosenbluth_j += boltzmann; 
            }

            // Multiply the Rosenbluth weights to get the Rosenbluth factor 
            log_forward_rosenbluth_ += log(rosenbluth_j); 
        }
        REQUIRE_THAT(
            log_forward_rosenbluth, 
            Catch::Matchers::WithinAbs(log_forward_rosenbluth_, tol)
        );
    }

    // Try generating a collection of 3-mer terminal segment moves at the tail
    for (int i = 0; i < n_moves; ++i)
    {
        auto result = sampler_random.generateForwardTerminalSegmentMove(
            segment_length, TerminalSegmentEnd::TAIL, n_candidates
        );
        Matrix<double, Dynamic, Dynamic> candidates = std::get<0>(result); 
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);  
        double log_forward_rosenbluth = std::get<2>(result);
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * segment_length); 
        REQUIRE(r_new.rows() == segment_length); 

        // Check that the chosen 3-mer terminal segment move indeed features
        // among the proposed atom positions
        for (int j = 0; j < segment_length; ++j)
        {
            bool found_r_new_j = false;
            Matrix<double, 3, 1> r_new_j = r_new.row(j);  
            for (int k = 0; k < n_candidates; ++k)
            {
                if ((r_new_j.transpose() - candidates(k, Eigen::seqN(3 * j, 3))).norm() < tol)
                {
                    found_r_new_j = true; 
                    break; 
                }
            }
            REQUIRE(found_r_new_j); 
        }
        
        // Check that the bonds have valid lengths
        //
        // The first atom should be bonded to the 6-th atom in the original
        // configuration
        REQUIRE((r_new.row(0) - coords.row(length - 1 - segment_length)).norm() < fene_params["R0"]); 

        // Each subsequent pair of atoms should also be separated by valid 
        // bond lengths 
        for (int j = 1; j < segment_length; ++j)
        {
            Matrix<double, 3, 1> r1 = r_new.row(j - 1); 
            Matrix<double, 3, 1> r2 = r_new.row(j); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < segment_length; ++j)
        {
            Matrix<double, 3, 1> predecessor; 
            if (j == 0)
                predecessor = coords.row(length - 1 - segment_length); 
            else 
                predecessor = r_new.row(j - 1);
            for (int k = 0; k < n_candidates; ++k)
            {
                double r = (predecessor.transpose() - candidates(k, Eigen::seqN(3 * j, 3))).norm();
                REQUIRE(r < fene_params["R0"]); 
            } 
        }

        // Check the Rosenbluth factor of the proposed move ... 
        //
        // For each proposed atom at each position, get the total non-bonded 
        // interaction energy between that atom and:
        // 1) every atom in the original configuration that survives the move
        // 2) the previous atoms along the segment
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_forward_rosenbluth_ = 0;  
        for (int j = 0; j < segment_length; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are moving the tail terminal segment, run through
                // all atoms 0, ..., 6 in the original configuration, which
                // are the atoms that survive the move 
                //
                // Omit atom 6 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = 0; m < length - 1 - segment_length; ++m)
                {
                    double r = (r_curr - coords.row(m).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    double r = (r_curr - coords.row(length - 1 - segment_length).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the segment (except for the immediately preceding atom)
                for (int m = 0; m < j - 1; ++m)
                {
                    double r = (r_curr - r_new.row(m).transpose()).norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Get the Boltzmann weight and increment the Rosenbluth weight 
                double boltzmann = exp(-residual_ijk / kT);
                rosenbluth_j += boltzmann; 
            }

            // Multiply the Rosenbluth weights to get the Rosenbluth factor 
            log_forward_rosenbluth_ += log(rosenbluth_j); 
        }
        REQUIRE_THAT(
            log_forward_rosenbluth, 
            Catch::Matchers::WithinAbs(log_forward_rosenbluth_, tol)
        );
    }

    // --------------------------------------------------------------- //
    // Terminal segment moves on 10-mer with a cosine angle potential and no 
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

    // Try generating a collection of 3-mer terminal segment moves at the head
    for (int i = 0; i < n_moves; ++i)
    {
        auto result = sampler_cosine.generateForwardTerminalSegmentMove(
            segment_length, TerminalSegmentEnd::HEAD, n_candidates
        );
        Matrix<double, Dynamic, Dynamic> candidates = std::get<0>(result); 
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);  
        double log_forward_rosenbluth = std::get<2>(result);
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * segment_length);  
        REQUIRE(r_new.rows() == segment_length); 

        // Check that the chosen 3-mer terminal segment move indeed features
        // among the proposed atom positions
        for (int j = 0; j < segment_length; ++j)
        {
            bool found_r_new_j = false;
            Matrix<double, 3, 1> r_new_j = r_new.row(j);  
            for (int k = 0; k < n_candidates; ++k)
            {
                if ((r_new_j.transpose() - candidates(k, Eigen::seqN(3 * j, 3))).norm() < tol)
                {
                    found_r_new_j = true; 
                    break; 
                }
            }
            REQUIRE(found_r_new_j); 
        }
        
        // Check that the bonds have valid lengths
        //
        // The first atom should be bonded to the 3rd atom in the original
        // configuration 
        REQUIRE((r_new.row(0) - coords.row(3)).norm() < fene_params["R0"]); 

        // Each subsequent pair of atoms should also be separated by valid 
        // bond lengths 
        for (int j = 1; j < segment_length; ++j)
        {
            Matrix<double, 3, 1> r1 = r_new.row(j - 1); 
            Matrix<double, 3, 1> r2 = r_new.row(j); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < segment_length; ++j)
        {
            Matrix<double, 3, 1> predecessor; 
            if (j == 0)
                predecessor = coords.row(3); 
            else 
                predecessor = r_new.row(j - 1);
            for (int k = 0; k < n_candidates; ++k)
            {
                double r = (predecessor.transpose() - candidates(k, Eigen::seqN(3 * j, 3))).norm();
                REQUIRE(r < fene_params["R0"]); 
            } 
        }

        // Check the Rosenbluth factor of the proposed move ... 
        //
        // For each proposed atom at each position, get the total non-bonded 
        // interaction energy between that atom and:
        // 1) every atom in the original configuration that survives the move
        // 2) the previous atoms along the segment
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_forward_rosenbluth_ = 0;  
        for (int j = 0; j < segment_length; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are moving the head terminal segment, run through
                // all atoms 3, ..., 9 in the original configuration, which 
                // are the atoms that survive the move 
                //
                // Omit atom 3 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = segment_length + 1; m < length; ++m)
                {
                    double r = (r_curr - coords.row(m).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    double r = (r_curr - coords.row(segment_length).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the segment (except for the immediately preceding atom)
                for (int m = 0; m < j - 1; ++m)
                {
                    double r = (r_curr - r_new.row(m).transpose()).norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Get the Boltzmann weight and increment the Rosenbluth weight 
                double boltzmann = exp(-residual_ijk / kT);
                rosenbluth_j += boltzmann; 
            }

            // Multiply the Rosenbluth weights to get the Rosenbluth factor 
            log_forward_rosenbluth_ += log(rosenbluth_j); 
        }
        REQUIRE_THAT(
            log_forward_rosenbluth, 
            Catch::Matchers::WithinAbs(log_forward_rosenbluth_, tol)
        );
    }

    // Try generating a collection of 3-mer terminal segment moves at the tail
    for (int i = 0; i < n_moves; ++i)
    {
        auto result = sampler_cosine.generateForwardTerminalSegmentMove(
            segment_length, TerminalSegmentEnd::TAIL, n_candidates
        );
        Matrix<double, Dynamic, Dynamic> candidates = std::get<0>(result); 
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);  
        double log_forward_rosenbluth = std::get<2>(result);
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * segment_length); 
        REQUIRE(r_new.rows() == segment_length); 

        // Check that the chosen 3-mer terminal segment move indeed features
        // among the proposed atom positions
        for (int j = 0; j < segment_length; ++j)
        {
            bool found_r_new_j = false;
            Matrix<double, 3, 1> r_new_j = r_new.row(j);  
            for (int k = 0; k < n_candidates; ++k)
            {
                if ((r_new_j.transpose() - candidates(k, Eigen::seqN(3 * j, 3))).norm() < tol)
                {
                    found_r_new_j = true; 
                    break; 
                }
            }
            REQUIRE(found_r_new_j); 
        }
        
        // Check that the bonds have valid lengths
        //
        // The first atom should be bonded to the 6-th atom in the original
        // configuration
        REQUIRE((r_new.row(0) - coords.row(length - 1 - segment_length)).norm() < fene_params["R0"]); 

        // Each subsequent pair of atoms should also be separated by valid 
        // bond lengths 
        for (int j = 1; j < segment_length; ++j)
        {
            Matrix<double, 3, 1> r1 = r_new.row(j - 1); 
            Matrix<double, 3, 1> r2 = r_new.row(j); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < segment_length; ++j)
        {
            Matrix<double, 3, 1> predecessor; 
            if (j == 0)
                predecessor = coords.row(length - 1 - segment_length); 
            else 
                predecessor = r_new.row(j - 1);
            for (int k = 0; k < n_candidates; ++k)
            {
                double r = (predecessor.transpose() - candidates(k, Eigen::seqN(3 * j, 3))).norm();
                REQUIRE(r < fene_params["R0"]); 
            } 
        }

        // Check the Rosenbluth factor of the proposed move ... 
        //
        // For each proposed atom at each position, get the total non-bonded 
        // interaction energy between that atom and:
        // 1) every atom in the original configuration that survives the move
        // 2) the previous atoms along the segment
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_forward_rosenbluth_ = 0;  
        for (int j = 0; j < segment_length; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are moving the tail terminal segment, run through
                // all atoms 0, ..., 6 in the original configuration, which
                // are the atoms that survive the move 
                //
                // Omit atom 6 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = 0; m < length - 1 - segment_length; ++m)
                {
                    double r = (r_curr - coords.row(m).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    double r = (r_curr - coords.row(length - 1 - segment_length).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the segment (except for the immediately preceding atom)
                for (int m = 0; m < j - 1; ++m)
                {
                    double r = (r_curr - r_new.row(m).transpose()).norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Get the Boltzmann weight and increment the Rosenbluth weight 
                double boltzmann = exp(-residual_ijk / kT);
                rosenbluth_j += boltzmann; 
            }

            // Multiply the Rosenbluth weights to get the Rosenbluth factor 
            log_forward_rosenbluth_ += log(rosenbluth_j); 
        }
        REQUIRE_THAT(
            log_forward_rosenbluth, 
            Catch::Matchers::WithinAbs(log_forward_rosenbluth_, tol)
        );
    }

    // --------------------------------------------------------------- //
    // Terminal segment moves on 10-mer with a cosine angle potential and a
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

    // Try generating a collection of 3-mer terminal segment moves at the head
    for (int i = 0; i < n_moves; ++i)
    {
        auto result = sampler_cosine_dihedral.generateForwardTerminalSegmentMove(
            segment_length, TerminalSegmentEnd::HEAD, n_candidates
        );
        Matrix<double, Dynamic, Dynamic> candidates = std::get<0>(result); 
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);  
        double log_forward_rosenbluth = std::get<2>(result);
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * segment_length);  
        REQUIRE(r_new.rows() == segment_length); 

        // Check that the chosen 3-mer terminal segment move indeed features
        // among the proposed atom positions
        for (int j = 0; j < segment_length; ++j)
        {
            bool found_r_new_j = false;
            Matrix<double, 3, 1> r_new_j = r_new.row(j);  
            for (int k = 0; k < n_candidates; ++k)
            {
                if ((r_new_j.transpose() - candidates(k, Eigen::seqN(3 * j, 3))).norm() < tol)
                {
                    found_r_new_j = true; 
                    break; 
                }
            }
            REQUIRE(found_r_new_j); 
        }
        
        // Check that the bonds have valid lengths
        //
        // The first atom should be bonded to the 3rd atom in the original
        // configuration 
        REQUIRE((r_new.row(0) - coords.row(3)).norm() < fene_params["R0"]); 

        // Each subsequent pair of atoms should also be separated by valid 
        // bond lengths 
        for (int j = 1; j < segment_length; ++j)
        {
            Matrix<double, 3, 1> r1 = r_new.row(j - 1); 
            Matrix<double, 3, 1> r2 = r_new.row(j); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < segment_length; ++j)
        {
            Matrix<double, 3, 1> predecessor; 
            if (j == 0)
                predecessor = coords.row(3); 
            else 
                predecessor = r_new.row(j - 1);
            for (int k = 0; k < n_candidates; ++k)
            {
                double r = (predecessor.transpose() - candidates(k, Eigen::seqN(3 * j, 3))).norm();
                REQUIRE(r < fene_params["R0"]); 
            } 
        }

        // Check the Rosenbluth factor of the proposed move ... 
        //
        // For each proposed atom at each position, get the total non-bonded 
        // interaction energy between that atom and:
        // 1) every atom in the original configuration that survives the move
        // 2) the previous atoms along the segment
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_forward_rosenbluth_ = 0;  
        for (int j = 0; j < segment_length; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are moving the head terminal segment, run through
                // all atoms 3, ..., 9 in the original configuration, which 
                // are the atoms that survive the move 
                //
                // Omit atom 3 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = segment_length + 1; m < length; ++m)
                {
                    double r = (r_curr - coords.row(m).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    double r = (r_curr - coords.row(segment_length).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the segment (except for the immediately preceding atom)
                for (int m = 0; m < j - 1; ++m)
                {
                    double r = (r_curr - r_new.row(m).transpose()).norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Get the Boltzmann weight and increment the Rosenbluth weight 
                double boltzmann = exp(-residual_ijk / kT);
                rosenbluth_j += boltzmann; 
            }

            // Multiply the Rosenbluth weights to get the Rosenbluth factor 
            log_forward_rosenbluth_ += log(rosenbluth_j); 
        }
        REQUIRE_THAT(
            log_forward_rosenbluth, 
            Catch::Matchers::WithinAbs(log_forward_rosenbluth_, tol)
        );
    }

    // Try generating a collection of 3-mer terminal segment moves at the tail
    for (int i = 0; i < n_moves; ++i)
    {
        auto result = sampler_cosine_dihedral.generateForwardTerminalSegmentMove(
            segment_length, TerminalSegmentEnd::TAIL, n_candidates
        );
        Matrix<double, Dynamic, Dynamic> candidates = std::get<0>(result); 
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);  
        double log_forward_rosenbluth = std::get<2>(result);
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * segment_length); 
        REQUIRE(r_new.rows() == segment_length); 

        // Check that the chosen 3-mer terminal segment move indeed features
        // among the proposed atom positions
        for (int j = 0; j < segment_length; ++j)
        {
            bool found_r_new_j = false;
            Matrix<double, 3, 1> r_new_j = r_new.row(j);  
            for (int k = 0; k < n_candidates; ++k)
            {
                if ((r_new_j.transpose() - candidates(k, Eigen::seqN(3 * j, 3))).norm() < tol)
                {
                    found_r_new_j = true; 
                    break; 
                }
            }
            REQUIRE(found_r_new_j); 
        }
        
        // Check that the bonds have valid lengths
        //
        // The first atom should be bonded to the 6-th atom in the original
        // configuration
        REQUIRE((r_new.row(0) - coords.row(length - 1 - segment_length)).norm() < fene_params["R0"]); 

        // Each subsequent pair of atoms should also be separated by valid 
        // bond lengths 
        for (int j = 1; j < segment_length; ++j)
        {
            Matrix<double, 3, 1> r1 = r_new.row(j - 1); 
            Matrix<double, 3, 1> r2 = r_new.row(j); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < segment_length; ++j)
        {
            Matrix<double, 3, 1> predecessor; 
            if (j == 0)
                predecessor = coords.row(length - 1 - segment_length); 
            else 
                predecessor = r_new.row(j - 1);
            for (int k = 0; k < n_candidates; ++k)
            {
                double r = (predecessor.transpose() - candidates(k, Eigen::seqN(3 * j, 3))).norm();
                REQUIRE(r < fene_params["R0"]); 
            } 
        }

        // Check the Rosenbluth factor of the proposed move ... 
        //
        // For each proposed atom at each position, get the total non-bonded 
        // interaction energy between that atom and:
        // 1) every atom in the original configuration that survives the move
        // 2) the previous atoms along the segment
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_forward_rosenbluth_ = 0;  
        for (int j = 0; j < segment_length; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are moving the tail terminal segment, run through
                // all atoms 0, ..., 6 in the original configuration, which
                // are the atoms that survive the move 
                //
                // Omit atom 6 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = 0; m < length - 1 - segment_length; ++m)
                {
                    double r = (r_curr - coords.row(m).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    double r = (r_curr - coords.row(length - 1 - segment_length).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the segment (except for the immediately preceding atom)
                for (int m = 0; m < j - 1; ++m)
                {
                    double r = (r_curr - r_new.row(m).transpose()).norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Get the Boltzmann weight and increment the Rosenbluth weight 
                double boltzmann = exp(-residual_ijk / kT);
                rosenbluth_j += boltzmann; 
            }

            // Multiply the Rosenbluth weights to get the Rosenbluth factor 
            log_forward_rosenbluth_ += log(rosenbluth_j); 
        }
        REQUIRE_THAT(
            log_forward_rosenbluth, 
            Catch::Matchers::WithinAbs(log_forward_rosenbluth_, tol)
        );
    }

    // --------------------------------------------------------------- //
    // Terminal segment moves on 10-mer with a Gaussian angle potential and a
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

    // Try generating a collection of 3-mer terminal segment moves at the head
    for (int i = 0; i < n_moves; ++i)
    {
        auto result = sampler_gaussian.generateForwardTerminalSegmentMove(
            segment_length, TerminalSegmentEnd::HEAD, n_candidates
        );
        Matrix<double, Dynamic, Dynamic> candidates = std::get<0>(result); 
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);  
        double log_forward_rosenbluth = std::get<2>(result);
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * segment_length);  
        REQUIRE(r_new.rows() == segment_length); 

        // Check that the chosen 3-mer terminal segment move indeed features
        // among the proposed atom positions
        for (int j = 0; j < segment_length; ++j)
        {
            bool found_r_new_j = false;
            Matrix<double, 3, 1> r_new_j = r_new.row(j);  
            for (int k = 0; k < n_candidates; ++k)
            {
                if ((r_new_j.transpose() - candidates(k, Eigen::seqN(3 * j, 3))).norm() < tol)
                {
                    found_r_new_j = true; 
                    break; 
                }
            }
            REQUIRE(found_r_new_j); 
        }
        
        // Check that the bonds have valid lengths
        //
        // The first atom should be bonded to the 3rd atom in the original
        // configuration 
        REQUIRE((r_new.row(0) - coords.row(3)).norm() < fene_params["R0"]); 

        // Each subsequent pair of atoms should also be separated by valid 
        // bond lengths 
        for (int j = 1; j < segment_length; ++j)
        {
            Matrix<double, 3, 1> r1 = r_new.row(j - 1); 
            Matrix<double, 3, 1> r2 = r_new.row(j); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < segment_length; ++j)
        {
            Matrix<double, 3, 1> predecessor; 
            if (j == 0)
                predecessor = coords.row(3); 
            else 
                predecessor = r_new.row(j - 1);
            for (int k = 0; k < n_candidates; ++k)
            {
                double r = (predecessor.transpose() - candidates(k, Eigen::seqN(3 * j, 3))).norm();
                REQUIRE(r < fene_params["R0"]); 
            } 
        }

        // Check the Rosenbluth factor of the proposed move ... 
        //
        // For each proposed atom at each position, get the total non-bonded 
        // interaction energy between that atom and:
        // 1) every atom in the original configuration that survives the move
        // 2) the previous atoms along the segment
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_forward_rosenbluth_ = 0;  
        for (int j = 0; j < segment_length; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are moving the head terminal segment, run through
                // all atoms 3, ..., 9 in the original configuration, which 
                // are the atoms that survive the move 
                //
                // Omit atom 3 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = segment_length + 1; m < length; ++m)
                {
                    double r = (r_curr - coords.row(m).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    double r = (r_curr - coords.row(segment_length).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the segment (except for the immediately preceding atom)
                for (int m = 0; m < j - 1; ++m)
                {
                    double r = (r_curr - r_new.row(m).transpose()).norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Get the Boltzmann weight and increment the Rosenbluth weight 
                double boltzmann = exp(-residual_ijk / kT);
                rosenbluth_j += boltzmann; 
            }

            // Multiply the Rosenbluth weights to get the Rosenbluth factor 
            log_forward_rosenbluth_ += log(rosenbluth_j); 
        }
        REQUIRE_THAT(
            log_forward_rosenbluth, 
            Catch::Matchers::WithinAbs(log_forward_rosenbluth_, tol)
        );
    }

    // Try generating a collection of 3-mer terminal segment moves at the tail
    for (int i = 0; i < n_moves; ++i)
    {
        auto result = sampler_gaussian.generateForwardTerminalSegmentMove(
            segment_length, TerminalSegmentEnd::TAIL, n_candidates
        );
        Matrix<double, Dynamic, Dynamic> candidates = std::get<0>(result); 
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);  
        double log_forward_rosenbluth = std::get<2>(result);
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * segment_length); 
        REQUIRE(r_new.rows() == segment_length); 

        // Check that the chosen 3-mer terminal segment move indeed features
        // among the proposed atom positions
        for (int j = 0; j < segment_length; ++j)
        {
            bool found_r_new_j = false;
            Matrix<double, 3, 1> r_new_j = r_new.row(j);  
            for (int k = 0; k < n_candidates; ++k)
            {
                if ((r_new_j.transpose() - candidates(k, Eigen::seqN(3 * j, 3))).norm() < tol)
                {
                    found_r_new_j = true; 
                    break; 
                }
            }
            REQUIRE(found_r_new_j); 
        }
        
        // Check that the bonds have valid lengths
        //
        // The first atom should be bonded to the 6-th atom in the original
        // configuration
        REQUIRE((r_new.row(0) - coords.row(length - 1 - segment_length)).norm() < fene_params["R0"]); 

        // Each subsequent pair of atoms should also be separated by valid 
        // bond lengths 
        for (int j = 1; j < segment_length; ++j)
        {
            Matrix<double, 3, 1> r1 = r_new.row(j - 1); 
            Matrix<double, 3, 1> r2 = r_new.row(j); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < segment_length; ++j)
        {
            Matrix<double, 3, 1> predecessor; 
            if (j == 0)
                predecessor = coords.row(length - 1 - segment_length); 
            else 
                predecessor = r_new.row(j - 1);
            for (int k = 0; k < n_candidates; ++k)
            {
                double r = (predecessor.transpose() - candidates(k, Eigen::seqN(3 * j, 3))).norm();
                REQUIRE(r < fene_params["R0"]); 
            } 
        }

        // Check the Rosenbluth factor of the proposed move ... 
        //
        // For each proposed atom at each position, get the total non-bonded 
        // interaction energy between that atom and:
        // 1) every atom in the original configuration that survives the move
        // 2) the previous atoms along the segment
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_forward_rosenbluth_ = 0;  
        for (int j = 0; j < segment_length; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are moving the tail terminal segment, run through
                // all atoms 0, ..., 6 in the original configuration, which
                // are the atoms that survive the move 
                //
                // Omit atom 6 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = 0; m < length - 1 - segment_length; ++m)
                {
                    double r = (r_curr - coords.row(m).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    double r = (r_curr - coords.row(length - 1 - segment_length).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the segment (except for the immediately preceding atom)
                for (int m = 0; m < j - 1; ++m)
                {
                    double r = (r_curr - r_new.row(m).transpose()).norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Get the Boltzmann weight and increment the Rosenbluth weight 
                double boltzmann = exp(-residual_ijk / kT);
                rosenbluth_j += boltzmann; 
            }

            // Multiply the Rosenbluth weights to get the Rosenbluth factor 
            log_forward_rosenbluth_ += log(rosenbluth_j); 
        }
        REQUIRE_THAT(
            log_forward_rosenbluth, 
            Catch::Matchers::WithinAbs(log_forward_rosenbluth_, tol)
        );
    }
}

/**
 * Tests for PolymerCBMCSampler::getBackwardTerminalSegmentMoveRosenbluthWeight(). 
 */
TEST_CASE(
    "Tests for backward Rosenbluth factor calculation for terminal segment move", 
    "[getBackwardTerminalSegmentMoveRosenbluthWeight()]"
)
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

    // Make cosine potential soft to allow for terminal segment move candidates
    // that are close to the original configuration  
    cosine_params["K"] = 0.5 * kT;
    cosine_params["theta0"] = 160 * boost::math::constants::pi<double>() / 180;

    // Similarly make Gaussian potential soft to allow for terminal segment 
    // move candidates that are close to the original configuration 
    gaussian_params["A1"] = 0.9; 
    gaussian_params["A2"] = 0.1;
    gaussian_params["w1"] = 2.0;     // Standard deviations of 1 for each component
    gaussian_params["w2"] = 2.0;
    gaussian_params["theta1"] = 160 * boost::math::constants::pi<double>() / 180; 
    gaussian_params["theta2"] = 90 * boost::math::constants::pi<double>() / 180;

    // Define dihedral potential parameters 
    nodihedral_params["K"] = 0;  
    dihedral_params["K"] = 0.5 * kT;

    // Define additional parameters for initialization and sampling
    const double collision_threshold = 0.9;    // Slightly less than 2^(1/6) * sigma ~ 1.01 
    const int max_tries_per_atom = 50;
    const int max_n_backtracks = 50;  
    Matrix<double, Dynamic, 2> bond_length_cdf = getFeneCDF<double>(
        lj_params["eps"], lj_params["sigma"], fene_params["K"], fene_params["R0"],
        kT, 10000
    );

    // --------------------------------------------------------------- //
    // Terminal segment moves on 10-mer with no angle or dihedral potentials 
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

    // Try generating a collection of 3-mer terminal segment moves at the head
    const int n_moves = 50; 
    const int n_candidates = 50;
    int segment_length = 3; 
    for (int i = 0; i < n_moves; ++i)
    {
        // First apply the terminal segment move to generate a new configuration 
        auto result = sampler_random.generateForwardTerminalSegmentMove(
            segment_length, TerminalSegmentEnd::HEAD, n_candidates
        );
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);
        PolymerConfiguration<double> config_moved(config); 
        config_moved.replaceSegment(r_new.colwise().reverse(), 0);
        Matrix<double, Dynamic, 3> coords_moved = config_moved.getSegment(0, length);

        // Check the coordinates of the moved configuration 
        for (int j = 0; j < segment_length; ++j)
        {
            REQUIRE_THAT(
                (coords_moved.row(j) - r_new.row(segment_length - 1 - j)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        }
        for (int j = segment_length; j < length; ++j)
        {
            REQUIRE_THAT(
                (coords_moved.row(j) - coords.row(j)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            );  
        }

        // Now calculate the backward Rosenbluth weights
        auto result2 = sampler_random.getBackwardTerminalSegmentMoveRosenbluthWeight(
            segment_length, TerminalSegmentEnd::HEAD, n_candidates, coords_moved
        );
        Matrix<double, Dynamic, Dynamic> candidates = result2.first; 
        double log_reverse_rosenbluth = result2.second;
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * segment_length); 

        // Check that the 0-th candidate for each position is reversion to 
        // the original configuration
        //
        // This involves comparing candidate atom 0 at each position (0, 1, 2)
        // with atoms 2, 1, 0 in the original configuration
        //
        // Note that the indices are mirrored because we are moving the 
        // head terminal segment 
        for (int j = 0; j < segment_length; ++j)
        { 
            REQUIRE_THAT(
                (candidates(0, Eigen::seqN(3 * j, 3)) - coords.row(segment_length - 1 - j)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < segment_length; ++j)
        {
            Matrix<double, 3, 1> predecessor;

            // If j == 0, then the predecessor is the 3rd atom in the original
            // configuration 
            //
            // If j > 0, then the predecessor is the 2nd or 1st atom in the 
            // original configuration 
            predecessor = coords.row(segment_length - j); 
            for (int k = 0; k < n_candidates; ++k)
            {
                double r = (predecessor.transpose() - candidates(k, Eigen::seqN(3 * j, 3))).norm();
                REQUIRE(r < fene_params["R0"]); 
            } 
        }

        // Check the backward Rosenbluth factor ... 
        //
        // For each proposed atom at each position, get the total non-bonded 
        // interaction energy between that atom and:
        // 1) every atom in the original configuration that survives the move
        // 2) the previous atoms along the segment
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_reverse_rosenbluth_ = 0; 
        for (int j = 0; j < segment_length; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are moving the head terminal segment, run through
                // all atoms 3, ... 9 in the moved configuration, which are
                // the atoms that survive the reverse move 
                //
                // Omit atom 3 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = segment_length + 1; m < length; ++m)
                {
                    double r = (r_curr - coords_moved.row(m).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    double r = (r_curr - coords_moved.row(segment_length).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the reversion segment (except for the immediately preceding
                // atom)
                //
                // Since we are only moving a segment of 3 monomers, this means
                // that, for j = 2, we add in interactions with atom 2 in the
                // original configuration 
                for (int m = 0; m < j - 1; ++m)
                {
                    double r = (r_curr - coords.row(segment_length - 1 - m).transpose()).norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Get the Boltzmann weight and increment the Rosenbluth weight 
                double boltzmann = exp(-residual_ijk / kT);
                rosenbluth_j += boltzmann; 
            }

            // Multiply the Rosenbluth weights to get the Rosenbluth factor 
            log_reverse_rosenbluth_ += log(rosenbluth_j); 
        }
        REQUIRE_THAT(
            log_reverse_rosenbluth, 
            Catch::Matchers::WithinAbs(log_reverse_rosenbluth_, tol)
        );
    }

    // Try generating a collection of 3-mer terminal segment moves at the tail 
    for (int i = 0; i < n_moves; ++i)
    {
        // First apply the terminal segment move to generate a new configuration 
        auto result = sampler_random.generateForwardTerminalSegmentMove(
            segment_length, TerminalSegmentEnd::TAIL, n_candidates
        );
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);
        PolymerConfiguration<double> config_moved(config); 
        config_moved.replaceSegment(r_new, length - segment_length); 
        Matrix<double, Dynamic, 3> coords_moved = config_moved.getSegment(0, length);

        // Check the coordinates of the moved configuration 
        for (int j = 0; j < length - segment_length; ++j)
        {
            REQUIRE_THAT(
                (coords_moved.row(j) - coords.row(j)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        }
        for (int j = length - segment_length; j < length; ++j)
        {
            REQUIRE_THAT(
                (coords_moved.row(j) - r_new.row(j - (length - segment_length))).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            );  
        }

        // Now calculate the backward Rosenbluth weights
        auto result2 = sampler_random.getBackwardTerminalSegmentMoveRosenbluthWeight(
            segment_length, TerminalSegmentEnd::TAIL, n_candidates, coords_moved
        );
        Matrix<double, Dynamic, Dynamic> candidates = result2.first; 
        double log_reverse_rosenbluth = result2.second;
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * segment_length); 

        // Check that the 0-th candidate for each position is reversion to 
        // the original configuration
        //
        // This involves comparing candidate atom 0 at each position (0, 1, 2)
        // with atoms 7, 8, 9 in the original configuration
        for (int j = 0; j < segment_length; ++j)
        { 
            REQUIRE_THAT(
                (candidates(0, Eigen::seqN(3 * j, 3)) - coords.row(length - segment_length + j)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < segment_length; ++j)
        {
            Matrix<double, 3, 1> predecessor;

            // If j == 0, then the predecessor is the 6th atom in the original
            // configuration
            //
            // If j > 0, then the predecessor is the 7th or 8th atom in the 
            // original configuration 
            predecessor = coords.row(length - segment_length - 1 + j); 
            for (int k = 0; k < n_candidates; ++k)
            {
                double r = (predecessor.transpose() - candidates(k, Eigen::seqN(3 * j, 3))).norm();
                REQUIRE(r < fene_params["R0"]); 
            } 
        }

        // Check the backward Rosenbluth factor ... 
        //
        // For each proposed atom at each position, get the total non-bonded 
        // interaction energy between that atom and:
        // 1) every atom in the original configuration that survives the move
        // 2) the previous atoms along the segment
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_reverse_rosenbluth_ = 0; 
        for (int j = 0; j < segment_length; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are moving the tail terminal segment, run through
                // all atoms 0, ..., 6 in the moved configuration, which are
                // the atoms that survive the reverse move
                //
                // Omit atom 6 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = 0; m < length - segment_length - 1; ++m)
                {
                    double r = (r_curr - coords_moved.row(m).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    double r = (r_curr - coords_moved.row(length - segment_length - 1).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the reversion segment (except for the immediately preceding
                // atom)
                //
                // Since we are only moving a segment of 3 monomers, this means
                // that, for j = 2, we add in interactions with atom 7 in the
                // original configuration 
                for (int m = 0; m < j - 1; ++m)
                {
                    double r = (r_curr - coords.row(length - segment_length + m).transpose()).norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Get the Boltzmann weight and increment the Rosenbluth weight 
                double boltzmann = exp(-residual_ijk / kT);
                rosenbluth_j += boltzmann; 
            }

            // Multiply the Rosenbluth weights to get the Rosenbluth factor 
            log_reverse_rosenbluth_ += log(rosenbluth_j); 
        }
        REQUIRE_THAT(
            log_reverse_rosenbluth, 
            Catch::Matchers::WithinAbs(log_reverse_rosenbluth_, tol)
        );
    }

    // --------------------------------------------------------------- //
    // Terminal segment moves on 10-mer with a cosine angle potential and no 
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

    // Try generating a collection of 3-mer terminal segment moves at the head
    for (int i = 0; i < n_moves; ++i)
    {
        // First apply the terminal segment move to generate a new configuration 
        auto result = sampler_cosine.generateForwardTerminalSegmentMove(
            segment_length, TerminalSegmentEnd::HEAD, n_candidates
        );
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);
        PolymerConfiguration<double> config_moved(config); 
        config_moved.replaceSegment(r_new.colwise().reverse(), 0);
        Matrix<double, Dynamic, 3> coords_moved = config_moved.getSegment(0, length);

        // Check the coordinates of the moved configuration 
        for (int j = 0; j < segment_length; ++j)
        {
            REQUIRE_THAT(
                (coords_moved.row(j) - r_new.row(segment_length - 1 - j)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        }
        for (int j = segment_length; j < length; ++j)
        {
            REQUIRE_THAT(
                (coords_moved.row(j) - coords.row(j)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            );  
        }

        // Now calculate the backward Rosenbluth weights
        auto result2 = sampler_cosine.getBackwardTerminalSegmentMoveRosenbluthWeight(
            segment_length, TerminalSegmentEnd::HEAD, n_candidates, coords_moved
        );
        Matrix<double, Dynamic, Dynamic> candidates = result2.first; 
        double log_reverse_rosenbluth = result2.second;
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * segment_length); 

        // Check that the 0-th candidate for each position is reversion to 
        // the original configuration
        //
        // This involves comparing candidate atom 0 at each position (0, 1, 2)
        // with atoms 2, 1, 0 in the original configuration
        //
        // Note that the indices are mirrored because we are moving the 
        // head terminal segment 
        for (int j = 0; j < segment_length; ++j)
        { 
            REQUIRE_THAT(
                (candidates(0, Eigen::seqN(3 * j, 3)) - coords.row(segment_length - 1 - j)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < segment_length; ++j)
        {
            Matrix<double, 3, 1> predecessor;

            // If j == 0, then the predecessor is the 3rd atom in the original
            // configuration 
            //
            // If j > 0, then the predecessor is the 2nd or 1st atom in the 
            // original configuration 
            predecessor = coords.row(segment_length - j); 
            for (int k = 0; k < n_candidates; ++k)
            {
                double r = (predecessor.transpose() - candidates(k, Eigen::seqN(3 * j, 3))).norm();
                REQUIRE(r < fene_params["R0"]); 
            } 
        }

        // Check the backward Rosenbluth factor ... 
        //
        // For each proposed atom at each position, get the total non-bonded 
        // interaction energy between that atom and:
        // 1) every atom in the original configuration that survives the move
        // 2) the previous atoms along the segment
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_reverse_rosenbluth_ = 0; 
        for (int j = 0; j < segment_length; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are moving the head terminal segment, run through
                // all atoms 3, ... 9 in the moved configuration, which are
                // the atoms that survive the reverse move 
                //
                // Omit atom 3 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = segment_length + 1; m < length; ++m)
                {
                    double r = (r_curr - coords_moved.row(m).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    double r = (r_curr - coords_moved.row(segment_length).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the reversion segment (except for the immediately preceding
                // atom)
                //
                // Since we are only moving a segment of 3 monomers, this means
                // that, for j = 2, we add in interactions with atom 2 in the
                // original configuration 
                for (int m = 0; m < j - 1; ++m)
                {
                    double r = (r_curr - coords.row(segment_length - 1 - m).transpose()).norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Get the Boltzmann weight and increment the Rosenbluth weight 
                double boltzmann = exp(-residual_ijk / kT);
                rosenbluth_j += boltzmann; 
            }

            // Multiply the Rosenbluth weights to get the Rosenbluth factor 
            log_reverse_rosenbluth_ += log(rosenbluth_j); 
        }
        REQUIRE_THAT(
            log_reverse_rosenbluth, 
            Catch::Matchers::WithinAbs(log_reverse_rosenbluth_, tol)
        );
    }

    // Try generating a collection of 3-mer terminal segment moves at the tail 
    for (int i = 0; i < n_moves; ++i)
    {
        // First apply the terminal segment move to generate a new configuration 
        auto result = sampler_cosine.generateForwardTerminalSegmentMove(
            segment_length, TerminalSegmentEnd::TAIL, n_candidates
        );
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);
        PolymerConfiguration<double> config_moved(config); 
        config_moved.replaceSegment(r_new, length - segment_length); 
        Matrix<double, Dynamic, 3> coords_moved = config_moved.getSegment(0, length);

        // Check the coordinates of the moved configuration 
        for (int j = 0; j < length - segment_length; ++j)
        {
            REQUIRE_THAT(
                (coords_moved.row(j) - coords.row(j)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        }
        for (int j = length - segment_length; j < length; ++j)
        {
            REQUIRE_THAT(
                (coords_moved.row(j) - r_new.row(j - (length - segment_length))).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            );  
        }

        // Now calculate the backward Rosenbluth weights
        auto result2 = sampler_cosine.getBackwardTerminalSegmentMoveRosenbluthWeight(
            segment_length, TerminalSegmentEnd::TAIL, n_candidates, coords_moved
        );
        Matrix<double, Dynamic, Dynamic> candidates = result2.first; 
        double log_reverse_rosenbluth = result2.second;
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * segment_length); 

        // Check that the 0-th candidate for each position is reversion to 
        // the original configuration
        //
        // This involves comparing candidate atom 0 at each position (0, 1, 2)
        // with atoms 7, 8, 9 in the original configuration
        for (int j = 0; j < segment_length; ++j)
        { 
            REQUIRE_THAT(
                (candidates(0, Eigen::seqN(3 * j, 3)) - coords.row(length - segment_length + j)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < segment_length; ++j)
        {
            Matrix<double, 3, 1> predecessor;

            // If j == 0, then the predecessor is the 6th atom in the original
            // configuration
            //
            // If j > 0, then the predecessor is the 7th or 8th atom in the 
            // original configuration 
            predecessor = coords.row(length - segment_length - 1 + j); 
            for (int k = 0; k < n_candidates; ++k)
            {
                double r = (predecessor.transpose() - candidates(k, Eigen::seqN(3 * j, 3))).norm();
                REQUIRE(r < fene_params["R0"]); 
            } 
        }

        // Check the backward Rosenbluth factor ... 
        //
        // For each proposed atom at each position, get the total non-bonded 
        // interaction energy between that atom and:
        // 1) every atom in the original configuration that survives the move
        // 2) the previous atoms along the segment
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_reverse_rosenbluth_ = 0; 
        for (int j = 0; j < segment_length; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are moving the tail terminal segment, run through
                // all atoms 0, ..., 6 in the moved configuration, which are
                // the atoms that survive the reverse move
                //
                // Omit atom 6 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = 0; m < length - segment_length - 1; ++m)
                {
                    double r = (r_curr - coords_moved.row(m).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    double r = (r_curr - coords_moved.row(length - segment_length - 1).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the reversion segment (except for the immediately preceding
                // atom)
                //
                // Since we are only moving a segment of 3 monomers, this means
                // that, for j = 2, we add in interactions with atom 7 in the
                // original configuration 
                for (int m = 0; m < j - 1; ++m)
                {
                    double r = (r_curr - coords.row(length - segment_length + m).transpose()).norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Get the Boltzmann weight and increment the Rosenbluth weight 
                double boltzmann = exp(-residual_ijk / kT);
                rosenbluth_j += boltzmann; 
            }

            // Multiply the Rosenbluth weights to get the Rosenbluth factor 
            log_reverse_rosenbluth_ += log(rosenbluth_j); 
        }
        REQUIRE_THAT(
            log_reverse_rosenbluth, 
            Catch::Matchers::WithinAbs(log_reverse_rosenbluth_, tol)
        );
    }

    // --------------------------------------------------------------- //
    // Terminal segment moves on 10-mer with a cosine angle potential and a
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
        cosine_params, nodihedral_params, rng
    ); 

    // Try generating a collection of 3-mer terminal segment moves at the head
    for (int i = 0; i < n_moves; ++i)
    {
        // First apply the terminal segment move to generate a new configuration 
        auto result = sampler_cosine_dihedral.generateForwardTerminalSegmentMove(
            segment_length, TerminalSegmentEnd::HEAD, n_candidates
        );
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);
        PolymerConfiguration<double> config_moved(config); 
        config_moved.replaceSegment(r_new.colwise().reverse(), 0);
        Matrix<double, Dynamic, 3> coords_moved = config_moved.getSegment(0, length);

        // Check the coordinates of the moved configuration 
        for (int j = 0; j < segment_length; ++j)
        {
            REQUIRE_THAT(
                (coords_moved.row(j) - r_new.row(segment_length - 1 - j)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        }
        for (int j = segment_length; j < length; ++j)
        {
            REQUIRE_THAT(
                (coords_moved.row(j) - coords.row(j)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            );  
        }

        // Now calculate the backward Rosenbluth weights
        auto result2 = sampler_cosine_dihedral.getBackwardTerminalSegmentMoveRosenbluthWeight(
            segment_length, TerminalSegmentEnd::HEAD, n_candidates, coords_moved
        );
        Matrix<double, Dynamic, Dynamic> candidates = result2.first; 
        double log_reverse_rosenbluth = result2.second;
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * segment_length); 

        // Check that the 0-th candidate for each position is reversion to 
        // the original configuration
        //
        // This involves comparing candidate atom 0 at each position (0, 1, 2)
        // with atoms 2, 1, 0 in the original configuration
        //
        // Note that the indices are mirrored because we are moving the 
        // head terminal segment 
        for (int j = 0; j < segment_length; ++j)
        { 
            REQUIRE_THAT(
                (candidates(0, Eigen::seqN(3 * j, 3)) - coords.row(segment_length - 1 - j)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < segment_length; ++j)
        {
            Matrix<double, 3, 1> predecessor;

            // If j == 0, then the predecessor is the 3rd atom in the original
            // configuration 
            //
            // If j > 0, then the predecessor is the 2nd or 1st atom in the 
            // original configuration 
            predecessor = coords.row(segment_length - j); 
            for (int k = 0; k < n_candidates; ++k)
            {
                double r = (predecessor.transpose() - candidates(k, Eigen::seqN(3 * j, 3))).norm();
                REQUIRE(r < fene_params["R0"]); 
            } 
        }

        // Check the backward Rosenbluth factor ... 
        //
        // For each proposed atom at each position, get the total non-bonded 
        // interaction energy between that atom and:
        // 1) every atom in the original configuration that survives the move
        // 2) the previous atoms along the segment
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_reverse_rosenbluth_ = 0; 
        for (int j = 0; j < segment_length; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are moving the head terminal segment, run through
                // all atoms 3, ... 9 in the moved configuration, which are
                // the atoms that survive the reverse move 
                //
                // Omit atom 3 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = segment_length + 1; m < length; ++m)
                {
                    double r = (r_curr - coords_moved.row(m).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    double r = (r_curr - coords_moved.row(segment_length).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the reversion segment (except for the immediately preceding
                // atom)
                //
                // Since we are only moving a segment of 3 monomers, this means
                // that, for j = 2, we add in interactions with atom 2 in the
                // original configuration 
                for (int m = 0; m < j - 1; ++m)
                {
                    double r = (r_curr - coords.row(segment_length - 1 - m).transpose()).norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Get the Boltzmann weight and increment the Rosenbluth weight 
                double boltzmann = exp(-residual_ijk / kT);
                rosenbluth_j += boltzmann; 
            }

            // Multiply the Rosenbluth weights to get the Rosenbluth factor 
            log_reverse_rosenbluth_ += log(rosenbluth_j); 
        }
        REQUIRE_THAT(
            log_reverse_rosenbluth, 
            Catch::Matchers::WithinAbs(log_reverse_rosenbluth_, tol)
        );
    }

    // Try generating a collection of 3-mer terminal segment moves at the tail 
    for (int i = 0; i < n_moves; ++i)
    {
        // First apply the terminal segment move to generate a new configuration 
        auto result = sampler_cosine_dihedral.generateForwardTerminalSegmentMove(
            segment_length, TerminalSegmentEnd::TAIL, n_candidates
        );
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);
        PolymerConfiguration<double> config_moved(config); 
        config_moved.replaceSegment(r_new, length - segment_length); 
        Matrix<double, Dynamic, 3> coords_moved = config_moved.getSegment(0, length);

        // Check the coordinates of the moved configuration
        for (int j = 0; j < length - segment_length; ++j)
        {
            REQUIRE_THAT(
                (coords_moved.row(j) - coords.row(j)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        }
        for (int j = length - segment_length; j < length; ++j)
        {
            REQUIRE_THAT(
                (coords_moved.row(j) - r_new.row(j - (length - segment_length))).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            );  
        }

        // Now calculate the backward Rosenbluth weights
        auto result2 = sampler_cosine_dihedral.getBackwardTerminalSegmentMoveRosenbluthWeight(
            segment_length, TerminalSegmentEnd::TAIL, n_candidates, coords_moved
        );
        Matrix<double, Dynamic, Dynamic> candidates = result2.first; 
        double log_reverse_rosenbluth = result2.second;
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * segment_length); 

        // Check that the 0-th candidate for each position is reversion to 
        // the original configuration
        //
        // This involves comparing candidate atom 0 at each position (0, 1, 2)
        // with atoms 7, 8, 9 in the original configuration
        for (int j = 0; j < segment_length; ++j)
        { 
            REQUIRE_THAT(
                (candidates(0, Eigen::seqN(3 * j, 3)) - coords.row(length - segment_length + j)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < segment_length; ++j)
        {
            Matrix<double, 3, 1> predecessor;

            // If j == 0, then the predecessor is the 6th atom in the original
            // configuration
            //
            // If j > 0, then the predecessor is the 7th or 8th atom in the 
            // original configuration 
            predecessor = coords.row(length - segment_length - 1 + j); 
            for (int k = 0; k < n_candidates; ++k)
            {
                double r = (predecessor.transpose() - candidates(k, Eigen::seqN(3 * j, 3))).norm();
                REQUIRE(r < fene_params["R0"]); 
            } 
        }

        // Check the backward Rosenbluth factor ... 
        //
        // For each proposed atom at each position, get the total non-bonded 
        // interaction energy between that atom and:
        // 1) every atom in the original configuration that survives the move
        // 2) the previous atoms along the segment
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_reverse_rosenbluth_ = 0; 
        for (int j = 0; j < segment_length; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are moving the tail terminal segment, run through
                // all atoms 0, ..., 6 in the moved configuration, which are
                // the atoms that survive the reverse move
                //
                // Omit atom 6 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = 0; m < length - segment_length - 1; ++m)
                {
                    double r = (r_curr - coords_moved.row(m).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    double r = (r_curr - coords_moved.row(length - segment_length - 1).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the reversion segment (except for the immediately preceding
                // atom)
                //
                // Since we are only moving a segment of 3 monomers, this means
                // that, for j = 2, we add in interactions with atom 7 in the
                // original configuration 
                for (int m = 0; m < j - 1; ++m)
                {
                    double r = (r_curr - coords.row(length - segment_length + m).transpose()).norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Get the Boltzmann weight and increment the Rosenbluth weight 
                double boltzmann = exp(-residual_ijk / kT);
                rosenbluth_j += boltzmann; 
            }

            // Multiply the Rosenbluth weights to get the Rosenbluth factor 
            log_reverse_rosenbluth_ += log(rosenbluth_j); 
        }
        REQUIRE_THAT(
            log_reverse_rosenbluth, 
            Catch::Matchers::WithinAbs(log_reverse_rosenbluth_, tol)
        );
    }

    // --------------------------------------------------------------- //
    // Terminal segment moves on 10-mer with a Gaussian angle potential and a
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
        config, lj_params, neighbor_threshold, fene_params, AngleMode::COSINE,
        cosine_params, nodihedral_params, rng
    ); 

    // Try generating a collection of 3-mer terminal segment moves at the head
    for (int i = 0; i < n_moves; ++i)
    {
        // First apply the terminal segment move to generate a new configuration 
        auto result = sampler_gaussian.generateForwardTerminalSegmentMove(
            segment_length, TerminalSegmentEnd::HEAD, n_candidates
        );
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);
        PolymerConfiguration<double> config_moved(config); 
        config_moved.replaceSegment(r_new.colwise().reverse(), 0);
        Matrix<double, Dynamic, 3> coords_moved = config_moved.getSegment(0, length);

        // Check the coordinates of the moved configuration 
        for (int j = 0; j < segment_length; ++j)
        {
            REQUIRE_THAT(
                (coords_moved.row(j) - r_new.row(segment_length - 1 - j)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        }
        for (int j = segment_length; j < length; ++j)
        {
            REQUIRE_THAT(
                (coords_moved.row(j) - coords.row(j)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            );  
        }

        // Now calculate the backward Rosenbluth weights
        auto result2 = sampler_gaussian.getBackwardTerminalSegmentMoveRosenbluthWeight(
            segment_length, TerminalSegmentEnd::HEAD, n_candidates, coords_moved
        );
        Matrix<double, Dynamic, Dynamic> candidates = result2.first; 
        double log_reverse_rosenbluth = result2.second;
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * segment_length); 

        // Check that the 0-th candidate for each position is reversion to 
        // the original configuration
        //
        // This involves comparing candidate atom 0 at each position (0, 1, 2)
        // with atoms 2, 1, 0 in the original configuration
        //
        // Note that the indices are mirrored because we are moving the 
        // head terminal segment 
        for (int j = 0; j < segment_length; ++j)
        { 
            REQUIRE_THAT(
                (candidates(0, Eigen::seqN(3 * j, 3)) - coords.row(segment_length - 1 - j)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < segment_length; ++j)
        {
            Matrix<double, 3, 1> predecessor;

            // If j == 0, then the predecessor is the 3rd atom in the original
            // configuration 
            //
            // If j > 0, then the predecessor is the 2nd or 1st atom in the 
            // original configuration 
            predecessor = coords.row(segment_length - j); 
            for (int k = 0; k < n_candidates; ++k)
            {
                double r = (predecessor.transpose() - candidates(k, Eigen::seqN(3 * j, 3))).norm();
                REQUIRE(r < fene_params["R0"]); 
            } 
        }

        // Check the backward Rosenbluth factor ... 
        //
        // For each proposed atom at each position, get the total non-bonded 
        // interaction energy between that atom and:
        // 1) every atom in the original configuration that survives the move
        // 2) the previous atoms along the segment
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_reverse_rosenbluth_ = 0; 
        for (int j = 0; j < segment_length; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are moving the head terminal segment, run through
                // all atoms 3, ... 9 in the moved configuration, which are
                // the atoms that survive the reverse move 
                //
                // Omit atom 3 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = segment_length + 1; m < length; ++m)
                {
                    double r = (r_curr - coords_moved.row(m).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    double r = (r_curr - coords_moved.row(segment_length).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the reversion segment (except for the immediately preceding
                // atom)
                //
                // Since we are only moving a segment of 3 monomers, this means
                // that, for j = 2, we add in interactions with atom 2 in the
                // original configuration 
                for (int m = 0; m < j - 1; ++m)
                {
                    double r = (r_curr - coords.row(segment_length - 1 - m).transpose()).norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Get the Boltzmann weight and increment the Rosenbluth weight 
                double boltzmann = exp(-residual_ijk / kT);
                rosenbluth_j += boltzmann; 
            }

            // Multiply the Rosenbluth weights to get the Rosenbluth factor 
            log_reverse_rosenbluth_ += log(rosenbluth_j); 
        }
        REQUIRE_THAT(
            log_reverse_rosenbluth, 
            Catch::Matchers::WithinAbs(log_reverse_rosenbluth_, tol)
        );
    }

    // Try generating a collection of 3-mer terminal segment moves at the tail 
    for (int i = 0; i < n_moves; ++i)
    {
        // First apply the terminal segment move to generate a new configuration 
        auto result = sampler_gaussian.generateForwardTerminalSegmentMove(
            segment_length, TerminalSegmentEnd::TAIL, n_candidates
        );
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);
        PolymerConfiguration<double> config_moved(config); 
        config_moved.replaceSegment(r_new, length - segment_length); 
        Matrix<double, Dynamic, 3> coords_moved = config_moved.getSegment(0, length);

        // Check the coordinates of the moved configuration 
        for (int j = 0; j < length - segment_length; ++j)
        {
            REQUIRE_THAT(
                (coords_moved.row(j) - coords.row(j)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        }
        for (int j = length - segment_length; j < length; ++j)
        {
            REQUIRE_THAT(
                (coords_moved.row(j) - r_new.row(j - (length - segment_length))).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            );  
        }

        // Now calculate the backward Rosenbluth weights
        auto result2 = sampler_gaussian.getBackwardTerminalSegmentMoveRosenbluthWeight(
            segment_length, TerminalSegmentEnd::TAIL, n_candidates, coords_moved
        );
        Matrix<double, Dynamic, Dynamic> candidates = result2.first; 
        double log_reverse_rosenbluth = result2.second;
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * segment_length); 

        // Check that the 0-th candidate for each position is reversion to 
        // the original configuration
        //
        // This involves comparing candidate atom 0 at each position (0, 1, 2)
        // with atoms 7, 8, 9 in the original configuration
        for (int j = 0; j < segment_length; ++j)
        { 
            REQUIRE_THAT(
                (candidates(0, Eigen::seqN(3 * j, 3)) - coords.row(length - segment_length + j)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < segment_length; ++j)
        {
            Matrix<double, 3, 1> predecessor;

            // If j == 0, then the predecessor is the 6th atom in the original
            // configuration
            //
            // If j > 0, then the predecessor is the 7th or 8th atom in the 
            // original configuration 
            predecessor = coords.row(length - segment_length - 1 + j); 
            for (int k = 0; k < n_candidates; ++k)
            {
                double r = (predecessor.transpose() - candidates(k, Eigen::seqN(3 * j, 3))).norm();
                REQUIRE(r < fene_params["R0"]); 
            } 
        }

        // Check the backward Rosenbluth factor ... 
        //
        // For each proposed atom at each position, get the total non-bonded 
        // interaction energy between that atom and:
        // 1) every atom in the original configuration that survives the move
        // 2) the previous atoms along the segment
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_reverse_rosenbluth_ = 0; 
        for (int j = 0; j < segment_length; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are moving the tail terminal segment, run through
                // all atoms 0, ..., 6 in the moved configuration, which are
                // the atoms that survive the reverse move
                //
                // Omit atom 6 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = 0; m < length - segment_length - 1; ++m)
                {
                    double r = (r_curr - coords_moved.row(m).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    double r = (r_curr - coords_moved.row(length - segment_length - 1).transpose()).norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the reversion segment (except for the immediately preceding
                // atom)
                //
                // Since we are only moving a segment of 3 monomers, this means
                // that, for j = 2, we add in interactions with atom 7 in the
                // original configuration 
                for (int m = 0; m < j - 1; ++m)
                {
                    double r = (r_curr - coords.row(length - segment_length + m).transpose()).norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Get the Boltzmann weight and increment the Rosenbluth weight 
                double boltzmann = exp(-residual_ijk / kT);
                rosenbluth_j += boltzmann; 
            }

            // Multiply the Rosenbluth weights to get the Rosenbluth factor 
            log_reverse_rosenbluth_ += log(rosenbluth_j); 
        }
        REQUIRE_THAT(
            log_reverse_rosenbluth, 
            Catch::Matchers::WithinAbs(log_reverse_rosenbluth_, tol)
        );
    }
}

/**
 * Tests for terminal segment moves in the PolymerCBMCSampler class via
 * moveOnce().
 *
 * Note that, because the key thermodynamic information regarding the move 
 * generation (Boltzmann weights, Rosenbluth weights/factors, etc.) are lost 
 * in calling the terminal segment move generation methods, the correctness
 * of these quantities is not tested here, but rather in the above two test 
 * modules. 
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

    // Make cosine potential soft to allow for terminal segment move candidates
    // that are close to the original configuration  
    cosine_params["K"] = 0.5 * kT;
    cosine_params["theta0"] = 160 * boost::math::constants::pi<double>() / 180;

    // Similarly make Gaussian potential soft to allow for terminal segment 
    // move candidates that are close to the original configuration 
    gaussian_params["A1"] = 0.9; 
    gaussian_params["A2"] = 0.1;
    gaussian_params["w1"] = 2.0;     // Standard deviations of 1 for each component
    gaussian_params["w2"] = 2.0;
    gaussian_params["theta1"] = 160 * boost::math::constants::pi<double>() / 180; 
    gaussian_params["theta2"] = 90 * boost::math::constants::pi<double>() / 180;

    // Define dihedral potential parameters 
    nodihedral_params["K"] = 0;  
    dihedral_params["K"] = 0.5 * kT;

    // Define additional parameters for initialization and sampling
    const double collision_threshold = 0.9;    // Slightly less than 2^(1/6) * sigma ~ 1.01 
    const int max_tries_per_atom = 50;
    const int max_n_backtracks = 50;  
    Matrix<double, Dynamic, 2> bond_length_cdf = getFeneCDF<double>(
        lj_params["eps"], lj_params["sigma"], fene_params["K"], fene_params["R0"],
        kT, 10000
    );

    // --------------------------------------------------------------- //
    // Terminal segment moves on 10-mer with no angle or dihedral potentials 
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

    // Try generating 50 terminal segment moves ... 
    const int n_moves = 50; 
    for (int i = 0; i < n_moves; ++i)
    { 
        // Try moving the configuration by choosing from 10000 3-mer candidate
        // terminal segment moves ...
        //
        // First generate the move 
        const int n_candidates = 10000;
        int segment_length = 3;  
        auto result = sampler_random.moveOnce(
            n_candidates, CBMCMoveType::TERMINAL_SEGMENT, segment_length
        ); 
        Matrix<double, Dynamic, Dynamic> forward_moves = std::get<0>(result); 
        Matrix<double, Dynamic, Dynamic> reverse_moves = std::get<1>(result);   // Ill-defined 
        int move_idx = std::get<2>(result); 
        double prob_accept = std::get<3>(result); 
        CBMCMoveResult accepted_move = std::get<4>(result);
        TerminalSegmentEnd terminal_end = static_cast<TerminalSegmentEnd>(
            std::get<5>(result).at("terminal_end")
        );

        // Check that the output is correctly specified 
        REQUIRE(forward_moves.rows() == segment_length); 
        REQUIRE(forward_moves.cols() == 3); 
        REQUIRE(move_idx == 0); 
        REQUIRE((prob_accept >= 0 && prob_accept <= 1));

        // Generate the moved configuration 
        //
        // Note that rows must be reversed if moving the head terminal segment 
        PolymerConfiguration<double> config_moved(config); 
        if (terminal_end == TerminalSegmentEnd::HEAD)
            config_moved.replaceSegment(forward_moves.colwise().reverse(), 0);
        else 
            config_moved.replaceSegment(forward_moves, length - segment_length);
        Matrix<double, Dynamic, 3> coords_moved = config_moved.getSegment(0, 10); 

        // Check that the moved configuration only contains valid bond lengths 
        for (int j = 0; j < length - 1; ++j)
        {
            Matrix<double, 3, 1> r1 = coords_moved.row(j); 
            Matrix<double, 3, 1> r2 = coords_moved.row(j + 1); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }
        
        // Check that, if the acceptance probability is 1, the chosen move was taken 
        if (prob_accept == 1)
            REQUIRE(accepted_move == CBMCMoveResult::ACCEPT); 

        // Check that, if the chosen move was taken, the resulting configuration is
        // as expected
        Matrix<double, Dynamic, 3> coords_result = sampler_random.getCoords();  
        if (accepted_move == CBMCMoveResult::ACCEPT)
        {
            // Move was taken 
            for (int j = 0; j < length; ++j)
            {
                REQUIRE_THAT(
                    (coords_result.row(j) - coords_moved.row(j)).norm(),
                    Catch::Matchers::WithinAbs(0, tol)
                ); 
            } 
        }
        else 
        {
            // Move was not taken 
            for (int j = 0; j < length; ++j)
            {
                REQUIRE_THAT(
                    (coords_result.row(j) - coords.row(j)).norm(),
                    Catch::Matchers::WithinAbs(0, tol)
                ); 
            } 
        }

        // Reset the coordinates in the sampler 
        sampler_random.setCoords(coords); 
    }

    // --------------------------------------------------------------- //
    // Terminal segment moves on 10-mer with a cosine angle potential and no 
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

    // Try generating 50 terminal segment moves ...
    for (int i = 0; i < n_moves; ++i)
    { 
        // Try moving the configuration by choosing from 10000 3-mer candidate
        // terminal segment moves ...
        //
        // First generate the move 
        const int n_candidates = 10000;
        int segment_length = 3;  
        auto result = sampler_cosine.moveOnce(
            n_candidates, CBMCMoveType::TERMINAL_SEGMENT, segment_length
        ); 
        Matrix<double, Dynamic, Dynamic> forward_moves = std::get<0>(result); 
        Matrix<double, Dynamic, Dynamic> reverse_moves = std::get<1>(result);   // Ill-defined 
        int move_idx = std::get<2>(result); 
        double prob_accept = std::get<3>(result); 
        CBMCMoveResult accepted_move = std::get<4>(result);
        TerminalSegmentEnd terminal_end = static_cast<TerminalSegmentEnd>(
            std::get<5>(result).at("terminal_end")
        );

        // Check that the output is correctly specified 
        REQUIRE(forward_moves.rows() == segment_length); 
        REQUIRE(forward_moves.cols() == 3); 
        REQUIRE(move_idx == 0); 
        REQUIRE((prob_accept >= 0 && prob_accept <= 1));

        // Generate the moved configuration 
        //
        // Note that rows must be reversed if moving the head terminal segment 
        PolymerConfiguration<double> config_moved(config); 
        if (terminal_end == TerminalSegmentEnd::HEAD)
            config_moved.replaceSegment(forward_moves.colwise().reverse(), 0);
        else 
            config_moved.replaceSegment(forward_moves, length - segment_length);
        Matrix<double, Dynamic, 3> coords_moved = config_moved.getSegment(0, 10); 

        // Check that the moved configuration only contains valid bond lengths 
        for (int j = 0; j < length - 1; ++j)
        {
            Matrix<double, 3, 1> r1 = coords_moved.row(j); 
            Matrix<double, 3, 1> r2 = coords_moved.row(j + 1); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }
        
        // Check that, if the acceptance probability is 1, the chosen move was taken 
        if (prob_accept == 1)
            REQUIRE(accepted_move == CBMCMoveResult::ACCEPT); 

        // Check that, if the chosen move was taken, the resulting configuration is
        // as expected
        Matrix<double, Dynamic, 3> coords_result = sampler_cosine.getCoords();  
        if (accepted_move == CBMCMoveResult::ACCEPT)
        {
            // Move was taken 
            for (int j = 0; j < length; ++j)
            {
                REQUIRE_THAT(
                    (coords_result.row(j) - coords_moved.row(j)).norm(),
                    Catch::Matchers::WithinAbs(0, tol)
                ); 
            } 
        }
        else 
        {
            // Move was not taken 
            for (int j = 0; j < length; ++j)
            {
                REQUIRE_THAT(
                    (coords_result.row(j) - coords.row(j)).norm(),
                    Catch::Matchers::WithinAbs(0, tol)
                ); 
            } 
        }

        // Reset the coordinates in the sampler 
        sampler_cosine.setCoords(coords); 
    }

    // --------------------------------------------------------------- //
    // Terminal segment moves on 10-mer with a cosine angle potential and a
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

    // Try generating 50 terminal segment moves ...
    for (int i = 0; i < n_moves; ++i)
    { 
        // Try moving the configuration by choosing from 10000 3-mer candidate
        // terminal segment moves ...
        //
        // First generate the move 
        const int n_candidates = 10000;
        int segment_length = 3;  
        auto result = sampler_cosine_dihedral.moveOnce(
            n_candidates, CBMCMoveType::TERMINAL_SEGMENT, segment_length
        ); 
        Matrix<double, Dynamic, Dynamic> forward_moves = std::get<0>(result); 
        Matrix<double, Dynamic, Dynamic> reverse_moves = std::get<1>(result);   // Ill-defined 
        int move_idx = std::get<2>(result); 
        double prob_accept = std::get<3>(result); 
        CBMCMoveResult accepted_move = std::get<4>(result);
        TerminalSegmentEnd terminal_end = static_cast<TerminalSegmentEnd>(
            std::get<5>(result).at("terminal_end")
        );

        // Check that the output is correctly specified 
        REQUIRE(forward_moves.rows() == segment_length); 
        REQUIRE(forward_moves.cols() == 3); 
        REQUIRE(move_idx == 0); 
        REQUIRE((prob_accept >= 0 && prob_accept <= 1));

        // Generate the moved configuration 
        //
        // Note that rows must be reversed if moving the head terminal segment 
        PolymerConfiguration<double> config_moved(config); 
        if (terminal_end == TerminalSegmentEnd::HEAD)
            config_moved.replaceSegment(forward_moves.colwise().reverse(), 0);
        else 
            config_moved.replaceSegment(forward_moves, length - segment_length);
        Matrix<double, Dynamic, 3> coords_moved = config_moved.getSegment(0, 10); 

        // Check that the moved configuration only contains valid bond lengths 
        for (int j = 0; j < length - 1; ++j)
        {
            Matrix<double, 3, 1> r1 = coords_moved.row(j); 
            Matrix<double, 3, 1> r2 = coords_moved.row(j + 1); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }
        
        // Check that, if the acceptance probability is 1, the chosen move was taken 
        if (prob_accept == 1)
            REQUIRE(accepted_move == CBMCMoveResult::ACCEPT); 

        // Check that, if the chosen move was taken, the resulting configuration is
        // as expected
        Matrix<double, Dynamic, 3> coords_result = sampler_cosine_dihedral.getCoords();  
        if (accepted_move == CBMCMoveResult::ACCEPT)
        {
            // Move was taken 
            for (int j = 0; j < length; ++j)
            {
                REQUIRE_THAT(
                    (coords_result.row(j) - coords_moved.row(j)).norm(),
                    Catch::Matchers::WithinAbs(0, tol)
                ); 
            } 
        }
        else 
        {
            // Move was not taken 
            for (int j = 0; j < length; ++j)
            {
                REQUIRE_THAT(
                    (coords_result.row(j) - coords.row(j)).norm(),
                    Catch::Matchers::WithinAbs(0, tol)
                ); 
            } 
        }

        // Reset the coordinates in the sampler 
        sampler_cosine_dihedral.setCoords(coords); 
    }

    // --------------------------------------------------------------- //
    // Terminal segment moves on 10-mer with a Gaussian angle potential and a
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

    // Try generating 50 terminal segment moves ...
    for (int i = 0; i < n_moves; ++i)
    { 
        // Try moving the configuration by choosing from 10000 3-mer candidate
        // terminal segment moves ...
        //
        // First generate the move 
        const int n_candidates = 10000;
        int segment_length = 3;  
        auto result = sampler_gaussian.moveOnce(
            n_candidates, CBMCMoveType::TERMINAL_SEGMENT, segment_length
        ); 
        Matrix<double, Dynamic, Dynamic> forward_moves = std::get<0>(result); 
        Matrix<double, Dynamic, Dynamic> reverse_moves = std::get<1>(result);   // Ill-defined 
        int move_idx = std::get<2>(result); 
        double prob_accept = std::get<3>(result); 
        CBMCMoveResult accepted_move = std::get<4>(result);
        TerminalSegmentEnd terminal_end = static_cast<TerminalSegmentEnd>(
            std::get<5>(result).at("terminal_end")
        );

        // Check that the output is correctly specified 
        REQUIRE(forward_moves.rows() == segment_length); 
        REQUIRE(forward_moves.cols() == 3); 
        REQUIRE(move_idx == 0); 
        REQUIRE((prob_accept >= 0 && prob_accept <= 1));

        // Generate the moved configuration 
        //
        // Note that rows must be reversed if moving the head terminal segment 
        PolymerConfiguration<double> config_moved(config); 
        if (terminal_end == TerminalSegmentEnd::HEAD)
            config_moved.replaceSegment(forward_moves.colwise().reverse(), 0);
        else 
            config_moved.replaceSegment(forward_moves, length - segment_length);
        Matrix<double, Dynamic, 3> coords_moved = config_moved.getSegment(0, 10); 

        // Check that the moved configuration only contains valid bond lengths 
        for (int j = 0; j < length - 1; ++j)
        {
            Matrix<double, 3, 1> r1 = coords_moved.row(j); 
            Matrix<double, 3, 1> r2 = coords_moved.row(j + 1); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }
        
        // Check that, if the acceptance probability is 1, the chosen move was taken 
        if (prob_accept == 1)
            REQUIRE(accepted_move == CBMCMoveResult::ACCEPT); 

        // Check that, if the chosen move was taken, the resulting configuration is
        // as expected
        Matrix<double, Dynamic, 3> coords_result = sampler_gaussian.getCoords();  
        if (accepted_move == CBMCMoveResult::ACCEPT)
        {
            // Move was taken 
            for (int j = 0; j < length; ++j)
            {
                REQUIRE_THAT(
                    (coords_result.row(j) - coords_moved.row(j)).norm(),
                    Catch::Matchers::WithinAbs(0, tol)
                ); 
            } 
        }
        else 
        {
            // Move was not taken 
            for (int j = 0; j < length; ++j)
            {
                REQUIRE_THAT(
                    (coords_result.row(j) - coords.row(j)).norm(),
                    Catch::Matchers::WithinAbs(0, tol)
                ); 
            } 
        }

        // Reset the coordinates in the sampler 
        sampler_gaussian.setCoords(coords); 
    }
}

