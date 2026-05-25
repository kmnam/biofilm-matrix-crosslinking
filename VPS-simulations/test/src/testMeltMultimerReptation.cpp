/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     5/25/2026
 */

#include <iostream>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/random.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/utils.hpp"
#include "../../include/polymerConfiguration.hpp"
#include "../../include/polymerMelt.hpp"
#include "../../include/cbmcMelt.hpp"

using namespace Eigen;

/**
 * Tests for PolymerMeltCBMCSampler::generateForwardMultimerReptationMove(). 
 */
TEST_CASE(
    "Tests for multimer reptation move generation",
    "[generateForwardMultimerReptationMove()]"
)
{
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;
    const double tol = 1e-8;  

    // Set up potential and sampling parameters
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

    // Define additional parameters for initialization and sampling
    const double intra_collision_threshold = 0.9;    // Slightly less than 2^(1/6) * sigma ~ 1.01
    const double inter_collision_threshold = 0.9; 
    const int max_tries_per_atom = 50;
    const int max_tries_per_kmer = 50; 
    const int max_tries_per_seed = 50; 
    const int max_n_backtracks = 50; 
    const int max_n_restarts = 50; 
    const double xmax = 50.0; 
    const double ymax = 50.0;
    const double zmax = 50.0;
    const double xlen = 2 * xmax; 
    const double ylen = 2 * ymax; 
    const double zlen = 2 * zmax;  
    Matrix<double, Dynamic, 2> bond_length_cdf = getFeneCDF<double>(
        lj_params["eps"], lj_params["sigma"], fene_params["K"], fene_params["R0"],
        kT, 10000
    );

    // --------------------------------------------------------------- //
    // Reptation moves on 10-mers with no angle or dihedral potentials 
    // --------------------------------------------------------------- //
    const int length = 10;
    const int n_chains = 5;  
    PolymerMeltConfiguration<double> melt_config = generateKMerMelt<double>(
        length, n_chains, lj_params, fene_params, AngleMode::COSINE,
        random_params, nodihedral_params, intra_collision_threshold,
        inter_collision_threshold, max_tries_per_atom, max_tries_per_kmer,
        max_tries_per_seed, max_n_backtracks, max_n_restarts, rng,
        uniform_dist, xmax, ymax, zmax, bond_length_cdf
    );
    std::vector<Matrix<double, Dynamic, 3> > coords; 
    for (int i = 0; i < 5; ++i)
    {
        Matrix<double, Dynamic, 3> coords_i = melt_config.getSegment(i, 0, length);  
        REQUIRE(melt_config.getLength(i) == length);
        REQUIRE(coords_i.rows() == length);
        coords.push_back(coords_i); 
    }

    // Initialize sampler instance
    double neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"]; 
    PolymerMeltCBMCSampler<double> sampler_random(
        melt_config, lj_params, neighbor_threshold, fene_params,
        AngleMode::COSINE, random_params, nodihedral_params, rng, -xmax, 
        xmax, -ymax, ymax, -zmax, zmax, bond_length_cdf
    );  

    // Try generating a collection of 3-mer reptation moves at the head of
    // chain 2
    const int n_moves = 50; 
    const int n_candidates = 50;
    int n_reptate = 3;
    int chain_idx = 2;  
    for (int i = 0; i < n_moves; ++i)
    {
        auto result = sampler_random.generateForwardMultimerReptationMove(
            chain_idx, ReptationDirection::HEAD, n_reptate, n_candidates
        );
        Matrix<double, Dynamic, Dynamic> candidates = std::get<0>(result); 
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);  
        double log_forward_rosenbluth = std::get<2>(result);
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * n_reptate);  
        REQUIRE(r_new.rows() == n_reptate);

        // Check that the chosen 3-mer reptation move indeed features among
        // the proposed atom positions
        for (int j = 0; j < n_reptate; ++j)
        {
            bool found_r_new_j = false;
            Matrix<double, 3, 1> r_new_j = r_new.row(j);  
            for (int k = 0; k < n_candidates; ++k)
            {
                // We are looking for a perfect match, so an unwrapped norm 
                // is fine here 
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
        // The first atom should be bonded to the 0-th atom in the original
        // configuration
        REQUIRE((r_new.row(0) - coords[chain_idx].row(0)).norm() < fene_params["R0"]); 

        // Each subsequent pair of atoms should also be separated by valid 
        // bond lengths 
        for (int j = 1; j < n_reptate; ++j)
        {
            Matrix<double, 3, 1> r1 = r_new.row(j - 1); 
            Matrix<double, 3, 1> r2 = r_new.row(j); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < n_reptate; ++j)
        {
            Matrix<double, 3, 1> predecessor; 
            if (j == 0)
                predecessor = coords[chain_idx].row(0); 
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
        // 1) every atom in the original configuration that survives the
        //    reptation move
        // 2) the previous atoms along the segment
        // 3) all atoms along all other chains
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_forward_rosenbluth_ = 0;  
        for (int j = 0; j < n_reptate; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are reptating towards the head, run through all
                // atoms 0, ... 6 in the original configuration, which are the
                // atoms that survive the reptation move
                //
                // Omit atom 0 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = 1; m < length - n_reptate; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords[chain_idx].row(m), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords[chain_idx].row(0), xlen, ylen, zlen
                    );
                    double r = dvec.norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the segment (except for the immediately preceding atom)
                for (int m = 0; m < j - 1; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, r_new.row(m), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all atoms along all 
                // other chains 
                for (int m = 0; m < n_chains; ++m)
                {
                    if (m != chain_idx)
                    {
                        for (int p = 0; p < length; ++p)
                        {
                            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                                r_curr, coords[m].row(p), xlen, ylen, zlen
                            );
                            double r = dvec.norm(); 
                            residual_ijk += lj<double>(
                                r, lj_params["eps"], lj_params["sigma"], true
                            );  
                        }
                    }
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

    // Try generating a collection of 3-mer reptation moves at the tail of
    // chain 3
    chain_idx = 3; 
    for (int i = 0; i < n_moves; ++i)
    {
        auto result = sampler_random.generateForwardMultimerReptationMove(
            chain_idx, ReptationDirection::TAIL, n_reptate, n_candidates
        );
        Matrix<double, Dynamic, Dynamic> candidates = std::get<0>(result); 
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);  
        double log_forward_rosenbluth = std::get<2>(result);
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * n_reptate);  
        REQUIRE(r_new.rows() == n_reptate);

        // Check that the chosen 3-mer reptation move indeed features among
        // the proposed atom positions
        for (int j = 0; j < n_reptate; ++j)
        {
            bool found_r_new_j = false;
            Matrix<double, 3, 1> r_new_j = r_new.row(j);  
            for (int k = 0; k < n_candidates; ++k)
            {
                // We are looking for a perfect match, so an unwrapped norm 
                // is fine here 
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
        // The first atom should be bonded to the 9-th atom in the original
        // configuration
        int tail_idx = length - 1;  
        REQUIRE((r_new.row(0) - coords[chain_idx].row(tail_idx)).norm() < fene_params["R0"]); 

        // Each subsequent pair of atoms should also be separated by valid 
        // bond lengths 
        for (int j = 1; j < n_reptate; ++j)
        {
            Matrix<double, 3, 1> r1 = r_new.row(j - 1); 
            Matrix<double, 3, 1> r2 = r_new.row(j); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < n_reptate; ++j)
        {
            Matrix<double, 3, 1> predecessor; 
            if (j == 0)
                predecessor = coords[chain_idx].row(tail_idx); 
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
        // 1) every atom in the original configuration that survives the
        //    reptation move
        // 2) the previous atoms along the segment
        // 3) all atoms along all other chains
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_forward_rosenbluth_ = 0;  
        for (int j = 0; j < n_reptate; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are reptating towards the tail, run through all
                // atoms 3, ..., 9 in the original configuration, which are
                // the atoms that survive the reptation move
                //
                // Omit atom 9 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = n_reptate; m < length - 1; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords[chain_idx].row(m), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords[chain_idx].row(tail_idx), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the segment (except for the immediately preceding atom)
                for (int m = 0; m < j - 1; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, r_new.row(m), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all atoms along all 
                // other chains 
                for (int m = 0; m < n_chains; ++m)
                {
                    if (m != chain_idx)
                    {
                        for (int p = 0; p < length; ++p)
                        {
                            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                                r_curr, coords[m].row(p), xlen, ylen, zlen
                            );
                            double r = dvec.norm(); 
                            residual_ijk += lj<double>(
                                r, lj_params["eps"], lj_params["sigma"], true
                            );  
                        }
                    }
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
    // Reptation moves on 10-mers with a cosine angle potential and no 
    // dihedral potential 
    // --------------------------------------------------------------- //
    melt_config = generateKMerMelt<double>(
        length, n_chains, lj_params, fene_params, AngleMode::COSINE,
        cosine_params, nodihedral_params, intra_collision_threshold,
        inter_collision_threshold, max_tries_per_atom, max_tries_per_kmer,
        max_tries_per_seed, max_n_backtracks, max_n_restarts, rng,
        uniform_dist, xmax, ymax, zmax, bond_length_cdf
    );
    coords.clear(); 
    for (int i = 0; i < 5; ++i)
    {
        Matrix<double, Dynamic, 3> coords_i = melt_config.getSegment(i, 0, length);  
        REQUIRE(melt_config.getLength(i) == length);
        REQUIRE(coords_i.rows() == length);
        coords.push_back(coords_i); 
    }

    // Initialize sampler instance
    PolymerMeltCBMCSampler<double> sampler_cosine(
        melt_config, lj_params, neighbor_threshold, fene_params,
        AngleMode::COSINE, cosine_params, nodihedral_params, rng, -xmax, 
        xmax, -ymax, ymax, -zmax, zmax, bond_length_cdf
    );  

    // Try generating a collection of 3-mer reptation moves at the head of
    // chain 2
    n_reptate = 3;
    chain_idx = 2;  
    for (int i = 0; i < n_moves; ++i)
    {
        auto result = sampler_cosine.generateForwardMultimerReptationMove(
            chain_idx, ReptationDirection::HEAD, n_reptate, n_candidates
        );
        Matrix<double, Dynamic, Dynamic> candidates = std::get<0>(result); 
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);  
        double log_forward_rosenbluth = std::get<2>(result);
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * n_reptate);  
        REQUIRE(r_new.rows() == n_reptate);

        // Check that the chosen 3-mer reptation move indeed features among
        // the proposed atom positions
        for (int j = 0; j < n_reptate; ++j)
        {
            bool found_r_new_j = false;
            Matrix<double, 3, 1> r_new_j = r_new.row(j);  
            for (int k = 0; k < n_candidates; ++k)
            {
                // We are looking for a perfect match, so an unwrapped norm 
                // is fine here 
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
        // The first atom should be bonded to the 0-th atom in the original
        // configuration
        REQUIRE((r_new.row(0) - coords[chain_idx].row(0)).norm() < fene_params["R0"]); 

        // Each subsequent pair of atoms should also be separated by valid 
        // bond lengths 
        for (int j = 1; j < n_reptate; ++j)
        {
            Matrix<double, 3, 1> r1 = r_new.row(j - 1); 
            Matrix<double, 3, 1> r2 = r_new.row(j); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < n_reptate; ++j)
        {
            Matrix<double, 3, 1> predecessor; 
            if (j == 0)
                predecessor = coords[chain_idx].row(0); 
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
        // 1) every atom in the original configuration that survives the
        //    reptation move
        // 2) the previous atoms along the segment
        // 3) all atoms along all other chains
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_forward_rosenbluth_ = 0;  
        for (int j = 0; j < n_reptate; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are reptating towards the head, run through all
                // atoms 0, ... 6 in the original configuration, which are the
                // atoms that survive the reptation move
                //
                // Omit atom 0 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = 1; m < length - n_reptate; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords[chain_idx].row(m), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords[chain_idx].row(0), xlen, ylen, zlen
                    );
                    double r = dvec.norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the segment (except for the immediately preceding atom)
                for (int m = 0; m < j - 1; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, r_new.row(m), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all atoms along all 
                // other chains 
                for (int m = 0; m < n_chains; ++m)
                {
                    if (m != chain_idx)
                    {
                        for (int p = 0; p < length; ++p)
                        {
                            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                                r_curr, coords[m].row(p), xlen, ylen, zlen
                            );
                            double r = dvec.norm(); 
                            residual_ijk += lj<double>(
                                r, lj_params["eps"], lj_params["sigma"], true
                            );  
                        }
                    }
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

    // Try generating a collection of 3-mer reptation moves at the tail of
    // chain 3
    chain_idx = 3; 
    for (int i = 0; i < n_moves; ++i)
    {
        auto result = sampler_cosine.generateForwardMultimerReptationMove(
            chain_idx, ReptationDirection::TAIL, n_reptate, n_candidates
        );
        Matrix<double, Dynamic, Dynamic> candidates = std::get<0>(result); 
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);  
        double log_forward_rosenbluth = std::get<2>(result);
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * n_reptate);  
        REQUIRE(r_new.rows() == n_reptate);

        // Check that the chosen 3-mer reptation move indeed features among
        // the proposed atom positions
        for (int j = 0; j < n_reptate; ++j)
        {
            bool found_r_new_j = false;
            Matrix<double, 3, 1> r_new_j = r_new.row(j);  
            for (int k = 0; k < n_candidates; ++k)
            {
                // We are looking for a perfect match, so an unwrapped norm 
                // is fine here 
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
        // The first atom should be bonded to the 9-th atom in the original
        // configuration
        int tail_idx = length - 1;  
        REQUIRE((r_new.row(0) - coords[chain_idx].row(tail_idx)).norm() < fene_params["R0"]); 

        // Each subsequent pair of atoms should also be separated by valid 
        // bond lengths 
        for (int j = 1; j < n_reptate; ++j)
        {
            Matrix<double, 3, 1> r1 = r_new.row(j - 1); 
            Matrix<double, 3, 1> r2 = r_new.row(j); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < n_reptate; ++j)
        {
            Matrix<double, 3, 1> predecessor; 
            if (j == 0)
                predecessor = coords[chain_idx].row(tail_idx); 
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
        // 1) every atom in the original configuration that survives the
        //    reptation move
        // 2) the previous atoms along the segment
        // 3) all atoms along all other chains
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_forward_rosenbluth_ = 0;  
        for (int j = 0; j < n_reptate; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are reptating towards the tail, run through all
                // atoms 3, ..., 9 in the original configuration, which are
                // the atoms that survive the reptation move
                //
                // Omit atom 9 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = n_reptate; m < length - 1; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords[chain_idx].row(m), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords[chain_idx].row(tail_idx), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the segment (except for the immediately preceding atom)
                for (int m = 0; m < j - 1; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, r_new.row(m), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all atoms along all 
                // other chains 
                for (int m = 0; m < n_chains; ++m)
                {
                    if (m != chain_idx)
                    {
                        for (int p = 0; p < length; ++p)
                        {
                            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                                r_curr, coords[m].row(p), xlen, ylen, zlen
                            );
                            double r = dvec.norm(); 
                            residual_ijk += lj<double>(
                                r, lj_params["eps"], lj_params["sigma"], true
                            );  
                        }
                    }
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
    // Reptation moves on 10-mers with a cosine angle potential and a
    // harmonic dihedral potential 
    // --------------------------------------------------------------- //
    melt_config = generateKMerMelt<double>(
        length, n_chains, lj_params, fene_params, AngleMode::COSINE,
        cosine_params, dihedral_params, intra_collision_threshold,
        inter_collision_threshold, max_tries_per_atom, max_tries_per_kmer,
        max_tries_per_seed, max_n_backtracks, max_n_restarts, rng,
        uniform_dist, xmax, ymax, zmax, bond_length_cdf
    );
    coords.clear(); 
    for (int i = 0; i < 5; ++i)
    {
        Matrix<double, Dynamic, 3> coords_i = melt_config.getSegment(i, 0, length);  
        REQUIRE(melt_config.getLength(i) == length);
        REQUIRE(coords_i.rows() == length);
        coords.push_back(coords_i); 
    }

    // Initialize sampler instance
    PolymerMeltCBMCSampler<double> sampler_cosine_dihedral(
        melt_config, lj_params, neighbor_threshold, fene_params,
        AngleMode::COSINE, cosine_params, dihedral_params, rng, -xmax, 
        xmax, -ymax, ymax, -zmax, zmax, bond_length_cdf
    );  

    // Try generating a collection of 3-mer reptation moves at the head of
    // chain 2
    n_reptate = 3;
    chain_idx = 2;  
    for (int i = 0; i < n_moves; ++i)
    {
        auto result = sampler_cosine_dihedral.generateForwardMultimerReptationMove(
            chain_idx, ReptationDirection::HEAD, n_reptate, n_candidates
        );
        Matrix<double, Dynamic, Dynamic> candidates = std::get<0>(result); 
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);  
        double log_forward_rosenbluth = std::get<2>(result);
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * n_reptate);  
        REQUIRE(r_new.rows() == n_reptate);

        // Check that the chosen 3-mer reptation move indeed features among
        // the proposed atom positions
        for (int j = 0; j < n_reptate; ++j)
        {
            bool found_r_new_j = false;
            Matrix<double, 3, 1> r_new_j = r_new.row(j);  
            for (int k = 0; k < n_candidates; ++k)
            {
                // We are looking for a perfect match, so an unwrapped norm 
                // is fine here 
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
        // The first atom should be bonded to the 0-th atom in the original
        // configuration
        REQUIRE((r_new.row(0) - coords[chain_idx].row(0)).norm() < fene_params["R0"]); 

        // Each subsequent pair of atoms should also be separated by valid 
        // bond lengths 
        for (int j = 1; j < n_reptate; ++j)
        {
            Matrix<double, 3, 1> r1 = r_new.row(j - 1); 
            Matrix<double, 3, 1> r2 = r_new.row(j); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < n_reptate; ++j)
        {
            Matrix<double, 3, 1> predecessor; 
            if (j == 0)
                predecessor = coords[chain_idx].row(0); 
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
        // 1) every atom in the original configuration that survives the
        //    reptation move
        // 2) the previous atoms along the segment
        // 3) all atoms along all other chains
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_forward_rosenbluth_ = 0;  
        for (int j = 0; j < n_reptate; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are reptating towards the head, run through all
                // atoms 0, ... 6 in the original configuration, which are the
                // atoms that survive the reptation move
                //
                // Omit atom 0 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = 1; m < length - n_reptate; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords[chain_idx].row(m), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords[chain_idx].row(0), xlen, ylen, zlen
                    );
                    double r = dvec.norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the segment (except for the immediately preceding atom)
                for (int m = 0; m < j - 1; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, r_new.row(m), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all atoms along all 
                // other chains 
                for (int m = 0; m < n_chains; ++m)
                {
                    if (m != chain_idx)
                    {
                        for (int p = 0; p < length; ++p)
                        {
                            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                                r_curr, coords[m].row(p), xlen, ylen, zlen
                            );
                            double r = dvec.norm(); 
                            residual_ijk += lj<double>(
                                r, lj_params["eps"], lj_params["sigma"], true
                            );  
                        }
                    }
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

    // Try generating a collection of 3-mer reptation moves at the tail of
    // chain 3
    chain_idx = 3; 
    for (int i = 0; i < n_moves; ++i)
    {
        auto result = sampler_cosine_dihedral.generateForwardMultimerReptationMove(
            chain_idx, ReptationDirection::TAIL, n_reptate, n_candidates
        );
        Matrix<double, Dynamic, Dynamic> candidates = std::get<0>(result); 
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);  
        double log_forward_rosenbluth = std::get<2>(result);
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * n_reptate);  
        REQUIRE(r_new.rows() == n_reptate);

        // Check that the chosen 3-mer reptation move indeed features among
        // the proposed atom positions
        for (int j = 0; j < n_reptate; ++j)
        {
            bool found_r_new_j = false;
            Matrix<double, 3, 1> r_new_j = r_new.row(j);  
            for (int k = 0; k < n_candidates; ++k)
            {
                // We are looking for a perfect match, so an unwrapped norm 
                // is fine here 
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
        // The first atom should be bonded to the 9-th atom in the original
        // configuration
        int tail_idx = length - 1;  
        REQUIRE((r_new.row(0) - coords[chain_idx].row(tail_idx)).norm() < fene_params["R0"]); 

        // Each subsequent pair of atoms should also be separated by valid 
        // bond lengths 
        for (int j = 1; j < n_reptate; ++j)
        {
            Matrix<double, 3, 1> r1 = r_new.row(j - 1); 
            Matrix<double, 3, 1> r2 = r_new.row(j); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < n_reptate; ++j)
        {
            Matrix<double, 3, 1> predecessor; 
            if (j == 0)
                predecessor = coords[chain_idx].row(tail_idx); 
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
        // 1) every atom in the original configuration that survives the
        //    reptation move
        // 2) the previous atoms along the segment
        // 3) all atoms along all other chains
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_forward_rosenbluth_ = 0;  
        for (int j = 0; j < n_reptate; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are reptating towards the tail, run through all
                // atoms 3, ..., 9 in the original configuration, which are
                // the atoms that survive the reptation move
                //
                // Omit atom 9 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = n_reptate; m < length - 1; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords[chain_idx].row(m), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords[chain_idx].row(tail_idx), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the segment (except for the immediately preceding atom)
                for (int m = 0; m < j - 1; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, r_new.row(m), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all atoms along all 
                // other chains 
                for (int m = 0; m < n_chains; ++m)
                {
                    if (m != chain_idx)
                    {
                        for (int p = 0; p < length; ++p)
                        {
                            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                                r_curr, coords[m].row(p), xlen, ylen, zlen
                            );
                            double r = dvec.norm(); 
                            residual_ijk += lj<double>(
                                r, lj_params["eps"], lj_params["sigma"], true
                            );  
                        }
                    }
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
    // Reptation moves on 10-mers with a Gaussian angle potential and a
    // harmonic dihedral potential 
    // --------------------------------------------------------------- //
    melt_config = generateKMerMelt<double>(
        length, n_chains, lj_params, fene_params, AngleMode::GAUSSIAN,
        gaussian_params, dihedral_params, intra_collision_threshold,
        inter_collision_threshold, max_tries_per_atom, max_tries_per_kmer,
        max_tries_per_seed, max_n_backtracks, max_n_restarts, rng,
        uniform_dist, xmax, ymax, zmax, bond_length_cdf
    );
    coords.clear(); 
    for (int i = 0; i < 5; ++i)
    {
        Matrix<double, Dynamic, 3> coords_i = melt_config.getSegment(i, 0, length);  
        REQUIRE(melt_config.getLength(i) == length);
        REQUIRE(coords_i.rows() == length);
        coords.push_back(coords_i); 
    }

    // Initialize sampler instance
    PolymerMeltCBMCSampler<double> sampler_gaussian(
        melt_config, lj_params, neighbor_threshold, fene_params,
        AngleMode::GAUSSIAN, gaussian_params, dihedral_params, rng, -xmax, 
        xmax, -ymax, ymax, -zmax, zmax, bond_length_cdf
    );  

    // Try generating a collection of 3-mer reptation moves at the head of
    // chain 2
    n_reptate = 3;
    chain_idx = 2;  
    for (int i = 0; i < n_moves; ++i)
    {
        auto result = sampler_gaussian.generateForwardMultimerReptationMove(
            chain_idx, ReptationDirection::HEAD, n_reptate, n_candidates
        );
        Matrix<double, Dynamic, Dynamic> candidates = std::get<0>(result); 
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);  
        double log_forward_rosenbluth = std::get<2>(result);
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * n_reptate);  
        REQUIRE(r_new.rows() == n_reptate);

        // Check that the chosen 3-mer reptation move indeed features among
        // the proposed atom positions
        for (int j = 0; j < n_reptate; ++j)
        {
            bool found_r_new_j = false;
            Matrix<double, 3, 1> r_new_j = r_new.row(j);  
            for (int k = 0; k < n_candidates; ++k)
            {
                // We are looking for a perfect match, so an unwrapped norm 
                // is fine here 
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
        // The first atom should be bonded to the 0-th atom in the original
        // configuration
        REQUIRE((r_new.row(0) - coords[chain_idx].row(0)).norm() < fene_params["R0"]); 

        // Each subsequent pair of atoms should also be separated by valid 
        // bond lengths 
        for (int j = 1; j < n_reptate; ++j)
        {
            Matrix<double, 3, 1> r1 = r_new.row(j - 1); 
            Matrix<double, 3, 1> r2 = r_new.row(j); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < n_reptate; ++j)
        {
            Matrix<double, 3, 1> predecessor; 
            if (j == 0)
                predecessor = coords[chain_idx].row(0); 
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
        // 1) every atom in the original configuration that survives the
        //    reptation move
        // 2) the previous atoms along the segment
        // 3) all atoms along all other chains
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_forward_rosenbluth_ = 0;  
        for (int j = 0; j < n_reptate; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are reptating towards the head, run through all
                // atoms 0, ... 6 in the original configuration, which are the
                // atoms that survive the reptation move
                //
                // Omit atom 0 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = 1; m < length - n_reptate; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords[chain_idx].row(m), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords[chain_idx].row(0), xlen, ylen, zlen
                    );
                    double r = dvec.norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the segment (except for the immediately preceding atom)
                for (int m = 0; m < j - 1; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, r_new.row(m), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all atoms along all 
                // other chains 
                for (int m = 0; m < n_chains; ++m)
                {
                    if (m != chain_idx)
                    {
                        for (int p = 0; p < length; ++p)
                        {
                            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                                r_curr, coords[m].row(p), xlen, ylen, zlen
                            );
                            double r = dvec.norm(); 
                            residual_ijk += lj<double>(
                                r, lj_params["eps"], lj_params["sigma"], true
                            );  
                        }
                    }
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

    // Try generating a collection of 3-mer reptation moves at the tail of
    // chain 3
    chain_idx = 3; 
    for (int i = 0; i < n_moves; ++i)
    {
        auto result = sampler_gaussian.generateForwardMultimerReptationMove(
            chain_idx, ReptationDirection::TAIL, n_reptate, n_candidates
        );
        Matrix<double, Dynamic, Dynamic> candidates = std::get<0>(result); 
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);  
        double log_forward_rosenbluth = std::get<2>(result);
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * n_reptate);  
        REQUIRE(r_new.rows() == n_reptate);

        // Check that the chosen 3-mer reptation move indeed features among
        // the proposed atom positions
        for (int j = 0; j < n_reptate; ++j)
        {
            bool found_r_new_j = false;
            Matrix<double, 3, 1> r_new_j = r_new.row(j);  
            for (int k = 0; k < n_candidates; ++k)
            {
                // We are looking for a perfect match, so an unwrapped norm 
                // is fine here 
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
        // The first atom should be bonded to the 9-th atom in the original
        // configuration
        int tail_idx = length - 1;  
        REQUIRE((r_new.row(0) - coords[chain_idx].row(tail_idx)).norm() < fene_params["R0"]); 

        // Each subsequent pair of atoms should also be separated by valid 
        // bond lengths 
        for (int j = 1; j < n_reptate; ++j)
        {
            Matrix<double, 3, 1> r1 = r_new.row(j - 1); 
            Matrix<double, 3, 1> r2 = r_new.row(j); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < n_reptate; ++j)
        {
            Matrix<double, 3, 1> predecessor; 
            if (j == 0)
                predecessor = coords[chain_idx].row(tail_idx); 
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
        // 1) every atom in the original configuration that survives the
        //    reptation move
        // 2) the previous atoms along the segment
        // 3) all atoms along all other chains
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_forward_rosenbluth_ = 0;  
        for (int j = 0; j < n_reptate; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are reptating towards the tail, run through all
                // atoms 3, ..., 9 in the original configuration, which are
                // the atoms that survive the reptation move
                //
                // Omit atom 9 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = n_reptate; m < length - 1; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords[chain_idx].row(m), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords[chain_idx].row(tail_idx), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the segment (except for the immediately preceding atom)
                for (int m = 0; m < j - 1; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, r_new.row(m), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all atoms along all 
                // other chains 
                for (int m = 0; m < n_chains; ++m)
                {
                    if (m != chain_idx)
                    {
                        for (int p = 0; p < length; ++p)
                        {
                            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                                r_curr, coords[m].row(p), xlen, ylen, zlen
                            );
                            double r = dvec.norm(); 
                            residual_ijk += lj<double>(
                                r, lj_params["eps"], lj_params["sigma"], true
                            );  
                        }
                    }
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
 * Tests for PolymerMeltCBMCSampler::getBackwardMultimerReptationRosenbluthWeight(). 
 */
TEST_CASE(
    "Tests for backward Rosenbluth factor calculation for multimer reptation", 
    "[getBackwardMultimerReptationRosenbluthWeight()]"
)
{
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;
    const double tol = 1e-8;  

    // Set up potential and sampling parameters
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

    // Define additional parameters for initialization and sampling
    const double intra_collision_threshold = 0.9;    // Slightly less than 2^(1/6) * sigma ~ 1.01
    const double inter_collision_threshold = 0.9; 
    const int max_tries_per_atom = 50;
    const int max_tries_per_kmer = 50; 
    const int max_tries_per_seed = 50; 
    const int max_n_backtracks = 50; 
    const int max_n_restarts = 50; 
    const double xmax = 50.0; 
    const double ymax = 50.0;
    const double zmax = 50.0;
    const double xlen = 2 * xmax; 
    const double ylen = 2 * ymax; 
    const double zlen = 2 * zmax;  
    Matrix<double, Dynamic, 2> bond_length_cdf = getFeneCDF<double>(
        lj_params["eps"], lj_params["sigma"], fene_params["K"], fene_params["R0"],
        kT, 10000
    );

    // --------------------------------------------------------------- //
    // Reptation moves on 10-mers with no angle or dihedral potentials 
    // --------------------------------------------------------------- //
    const int length = 10; 
    const int n_chains = 5;  
    PolymerMeltConfiguration<double> melt_config = generateKMerMelt<double>(
        length, n_chains, lj_params, fene_params, AngleMode::COSINE,
        random_params, nodihedral_params, intra_collision_threshold,
        inter_collision_threshold, max_tries_per_atom, max_tries_per_kmer,
        max_tries_per_seed, max_n_backtracks, max_n_restarts, rng,
        uniform_dist, xmax, ymax, zmax, bond_length_cdf
    );
    std::vector<Matrix<double, Dynamic, 3> > coords; 
    for (int i = 0; i < 5; ++i)
    {
        Matrix<double, Dynamic, 3> coords_i = melt_config.getSegment(i, 0, length);  
        REQUIRE(melt_config.getLength(i) == length);
        REQUIRE(coords_i.rows() == length);
        coords.push_back(coords_i); 
    }

    // Initialize sampler instance
    double neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"]; 
    PolymerMeltCBMCSampler<double> sampler_random(
        melt_config, lj_params, neighbor_threshold, fene_params,
        AngleMode::COSINE, random_params, nodihedral_params, rng, -xmax, 
        xmax, -ymax, ymax, -zmax, zmax, bond_length_cdf
    );  

    // Try generating a collection of 3-mer reptation moves at the head
    // of chain 0
    const int n_moves = 50; 
    const int n_candidates = 50;
    int n_reptate = 3;
    int chain_idx = 0; 
    for (int i = 0; i < n_moves; ++i)
    {
        // First apply the reptation move to generate a new configuration 
        auto result = sampler_random.generateForwardMultimerReptationMove(
            chain_idx, ReptationDirection::HEAD, n_reptate, n_candidates
        );
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);
        PolymerMeltConfiguration<double> melt_config_reptated(melt_config); 
        melt_config_reptated.reptateTowardsHeadMultimer(
            chain_idx, r_new.colwise().reverse()
        );
        Matrix<double, Dynamic, 3> coords_reptated = melt_config_reptated.getSegment(chain_idx, 0, length);

        // Check the coordinates of the reptated configuration 
        for (int j = 0; j < n_reptate; ++j)
        {
            // We are looking for a perfect match, so an unwrapped norm 
            // is fine here 
            REQUIRE_THAT(
                (coords_reptated.row(j) - r_new.row(n_reptate - 1 - j)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        }
        for (int j = n_reptate; j < length; ++j)
        {
            // We are looking for a perfect match, so an unwrapped norm 
            // is fine here 
            REQUIRE_THAT(
                (coords_reptated.row(j) - coords[chain_idx].row(j - n_reptate)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            );  
        }

        // Now calculate the backward Rosenbluth weights
        auto result2 = sampler_random.getBackwardMultimerReptationRosenbluthWeight(
            chain_idx, ReptationDirection::TAIL, n_reptate, n_candidates,
            coords_reptated
        );
        Matrix<double, Dynamic, Dynamic> candidates = result2.first; 
        double log_reverse_rosenbluth = result2.second;
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * n_reptate);

        // Check that the 0-th candidate for each position is reversion to 
        // the original configuration
        //
        // This involves comparing candidate atom 0 at each position (0, 1, 2)
        // with atoms 7, 8, 9 in the original configuration 
        for (int j = 0; j < n_reptate; ++j)
        { 
            // We are looking for a perfect match, so an unwrapped norm 
            // is fine here 
            REQUIRE_THAT(
                (candidates(0, Eigen::seqN(3 * j, 3)) - coords[chain_idx].row(length - n_reptate + j)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < n_reptate; ++j)
        {
            Matrix<double, 3, 1> predecessor;

            // If j == 0, then the predecessor is the last atom in the 
            // reptated configuration, which is the 6th atom in the original
            // configuration 
            //
            // If j > 0, then the predecessor is the 7th or 8th atom in the
            // original configuration 
            predecessor = coords[chain_idx].row(length - 1 - n_reptate + j);
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
        // 1) every atom in the original configuration that survives the
        //    reptation move
        // 2) the previous atoms along the segment
        // 3) all atoms along all other chains
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_reverse_rosenbluth_ = 0; 
        for (int j = 0; j < n_reptate; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are reptating towards the tail (in reverse), run
                // through all atoms 3, ... 9 in the reptated configuration, 
                // which are the atoms that survive the reverse reptation move
                //
                // Omit atom 9 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = n_reptate; m < length - 1; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords_reptated.row(m), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm();
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords_reptated.row(length - 1),
                        xlen, ylen, zlen
                    );
                    double r = dvec.norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the reversion segment (except for the immediately preceding
                // atom)
                //
                // Since we are only reptating by 3 monomers, this means that,
                // for j = 2, we add in interactions with atom 7 in the original
                // configuration 
                for (int m = 0; m < j - 1; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords[chain_idx].row(length - n_reptate + m),
                        xlen, ylen, zlen
                    );
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all atoms along all 
                // other chains 
                for (int m = 0; m < n_chains; ++m)
                {
                    if (m != chain_idx)
                    {
                        for (int p = 0; p < length; ++p)
                        {
                            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                                r_curr, coords[m].row(p), xlen, ylen, zlen
                            );
                            double r = dvec.norm(); 
                            residual_ijk += lj<double>(
                                r, lj_params["eps"], lj_params["sigma"], true
                            ); 
                        }
                    }
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

    // Try generating a collection of 3-mer reptation moves at the tail 
    for (int i = 0; i < n_moves; ++i)
    {
        // First apply the reptation move to generate a new configuration 
        auto result = sampler_random.generateForwardMultimerReptationMove(
            chain_idx, ReptationDirection::TAIL, n_reptate, n_candidates
        );
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);
        PolymerMeltConfiguration<double> melt_config_reptated(melt_config); 
        melt_config_reptated.reptateTowardsTailMultimer(chain_idx, r_new); 
        Matrix<double, Dynamic, 3> coords_reptated = melt_config_reptated.getSegment(chain_idx, 0, length);

        // Check the coordinates of the reptated configuration 
        for (int j = 0; j < length - n_reptate; ++j)
        {
            // We are looking for a perfect match, so an unwrapped norm 
            // is fine here 
            REQUIRE_THAT(
                (coords_reptated.row(j) - coords[chain_idx].row(j + n_reptate)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        }
        for (int j = length - n_reptate; j < length; ++j)
        {
            // We are looking for a perfect match, so an unwrapped norm 
            // is fine here 
            REQUIRE_THAT(
                (coords_reptated.row(j) - r_new.row(j - (length - n_reptate))).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            );  
        }

        // Now calculate the backward Rosenbluth weights
        auto result2 = sampler_random.getBackwardMultimerReptationRosenbluthWeight(
            chain_idx, ReptationDirection::HEAD, n_reptate, n_candidates,
            coords_reptated
        );
        Matrix<double, Dynamic, Dynamic> candidates = result2.first; 
        double log_reverse_rosenbluth = result2.second;
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * n_reptate);

        // Check that the 0-th candidate for each position is reversion to 
        // the original configuration
        //
        // This involves comparing candidate atom 0 at each position (0, 1, 2)
        // with atoms 2, 1, 0 in the original configuration
        //
        // Note that the indices are mirrored because reversion requires 
        // reptation towards the head  
        for (int j = 0; j < n_reptate; ++j)
        { 
            REQUIRE_THAT(
                (candidates(0, Eigen::seqN(3 * j, 3)) - coords[chain_idx].row(n_reptate - 1 - j)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < n_reptate; ++j)
        {
            Matrix<double, 3, 1> predecessor;

            // If j == 0, then the predecessor is the 0th atom in the 
            // reptated configuration, which is the 3rd atom in the original
            // configuration 
            //
            // If j > 0, then the predecessor is the 2nd or 1st atom in the 
            // original configuration 
            predecessor = coords[chain_idx].row(n_reptate - j); 
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
        // 1) every atom in the original configuration that survives the
        //    reptation move
        // 2) the previous atoms along the segment
        // 3) all atoms along all other chains
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_reverse_rosenbluth_ = 0; 
        for (int j = 0; j < n_reptate; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are reptating towards the head (in reverse), run
                // through all atoms 0, ..., 6 in the reptated configuration, 
                // which are the atoms that survive the reverse reptation move
                //
                // Omit atom 0 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = 1; m < length - n_reptate; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords_reptated.row(m), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords_reptated.row(0), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the reversion segment (except for the immediately preceding
                // atom)
                //
                // Since we are only reptating by 3 monomers, this means that,
                // for j = 2, we add in interactions with atom 2 in the original
                // configuration 
                for (int m = 0; m < j - 1; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords[chain_idx].row(n_reptate - 1 - m),
                        xlen, ylen, zlen
                    );
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all atoms along all 
                // other chains 
                for (int m = 0; m < n_chains; ++m)
                {
                    if (m != chain_idx)
                    {
                        for (int p = 0; p < length; ++p)
                        {
                            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                                r_curr, coords[m].row(p), xlen, ylen, zlen
                            );
                            double r = dvec.norm(); 
                            residual_ijk += lj<double>(
                                r, lj_params["eps"], lj_params["sigma"], true
                            ); 
                        }
                    }
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
    // Reptation moves on 10-mers with a cosine angle potential and no 
    // dihedral potential 
    // --------------------------------------------------------------- //
    melt_config = generateKMerMelt<double>(
        length, n_chains, lj_params, fene_params, AngleMode::COSINE,
        cosine_params, nodihedral_params, intra_collision_threshold,
        inter_collision_threshold, max_tries_per_atom, max_tries_per_kmer,
        max_tries_per_seed, max_n_backtracks, max_n_restarts, rng,
        uniform_dist, xmax, ymax, zmax, bond_length_cdf
    );
    coords.clear(); 
    for (int i = 0; i < 5; ++i)
    {
        Matrix<double, Dynamic, 3> coords_i = melt_config.getSegment(i, 0, length);  
        REQUIRE(melt_config.getLength(i) == length);
        REQUIRE(coords_i.rows() == length);
        coords.push_back(coords_i); 
    }

    // Initialize sampler instance
    PolymerMeltCBMCSampler<double> sampler_cosine(
        melt_config, lj_params, neighbor_threshold, fene_params,
        AngleMode::COSINE, cosine_params, nodihedral_params, rng, -xmax, 
        xmax, -ymax, ymax, -zmax, zmax, bond_length_cdf
    );  

    // Try generating a collection of 3-mer reptation moves at the head
    // of chain 0
    n_reptate = 3;
    chain_idx = 0; 
    for (int i = 0; i < n_moves; ++i)
    {
        // First apply the reptation move to generate a new configuration 
        auto result = sampler_cosine.generateForwardMultimerReptationMove(
            chain_idx, ReptationDirection::HEAD, n_reptate, n_candidates
        );
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);
        PolymerMeltConfiguration<double> melt_config_reptated(melt_config); 
        melt_config_reptated.reptateTowardsHeadMultimer(
            chain_idx, r_new.colwise().reverse()
        );
        Matrix<double, Dynamic, 3> coords_reptated = melt_config_reptated.getSegment(chain_idx, 0, length);

        // Check the coordinates of the reptated configuration 
        for (int j = 0; j < n_reptate; ++j)
        {
            // We are looking for a perfect match, so an unwrapped norm 
            // is fine here 
            REQUIRE_THAT(
                (coords_reptated.row(j) - r_new.row(n_reptate - 1 - j)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        }
        for (int j = n_reptate; j < length; ++j)
        {
            // We are looking for a perfect match, so an unwrapped norm 
            // is fine here 
            REQUIRE_THAT(
                (coords_reptated.row(j) - coords[chain_idx].row(j - n_reptate)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            );  
        }

        // Now calculate the backward Rosenbluth weights
        auto result2 = sampler_cosine.getBackwardMultimerReptationRosenbluthWeight(
            chain_idx, ReptationDirection::TAIL, n_reptate, n_candidates,
            coords_reptated
        );
        Matrix<double, Dynamic, Dynamic> candidates = result2.first; 
        double log_reverse_rosenbluth = result2.second;
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * n_reptate);

        // Check that the 0-th candidate for each position is reversion to 
        // the original configuration
        //
        // This involves comparing candidate atom 0 at each position (0, 1, 2)
        // with atoms 7, 8, 9 in the original configuration 
        for (int j = 0; j < n_reptate; ++j)
        { 
            // We are looking for a perfect match, so an unwrapped norm 
            // is fine here 
            REQUIRE_THAT(
                (candidates(0, Eigen::seqN(3 * j, 3)) - coords[chain_idx].row(length - n_reptate + j)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < n_reptate; ++j)
        {
            Matrix<double, 3, 1> predecessor;

            // If j == 0, then the predecessor is the last atom in the 
            // reptated configuration, which is the 6th atom in the original
            // configuration 
            //
            // If j > 0, then the predecessor is the 7th or 8th atom in the
            // original configuration 
            predecessor = coords[chain_idx].row(length - 1 - n_reptate + j);
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
        // 1) every atom in the original configuration that survives the
        //    reptation move
        // 2) the previous atoms along the segment
        // 3) all atoms along all other chains
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_reverse_rosenbluth_ = 0; 
        for (int j = 0; j < n_reptate; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are reptating towards the tail (in reverse), run
                // through all atoms 3, ... 9 in the reptated configuration, 
                // which are the atoms that survive the reverse reptation move
                //
                // Omit atom 9 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = n_reptate; m < length - 1; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords_reptated.row(m), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm();
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords_reptated.row(length - 1),
                        xlen, ylen, zlen
                    );
                    double r = dvec.norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the reversion segment (except for the immediately preceding
                // atom)
                //
                // Since we are only reptating by 3 monomers, this means that,
                // for j = 2, we add in interactions with atom 7 in the original
                // configuration 
                for (int m = 0; m < j - 1; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords[chain_idx].row(length - n_reptate + m),
                        xlen, ylen, zlen
                    );
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all atoms along all 
                // other chains 
                for (int m = 0; m < n_chains; ++m)
                {
                    if (m != chain_idx)
                    {
                        for (int p = 0; p < length; ++p)
                        {
                            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                                r_curr, coords[m].row(p), xlen, ylen, zlen
                            );
                            double r = dvec.norm(); 
                            residual_ijk += lj<double>(
                                r, lj_params["eps"], lj_params["sigma"], true
                            ); 
                        }
                    }
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

    // Try generating a collection of 3-mer reptation moves at the tail 
    for (int i = 0; i < n_moves; ++i)
    {
        // First apply the reptation move to generate a new configuration 
        auto result = sampler_cosine.generateForwardMultimerReptationMove(
            chain_idx, ReptationDirection::TAIL, n_reptate, n_candidates
        );
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);
        PolymerMeltConfiguration<double> melt_config_reptated(melt_config); 
        melt_config_reptated.reptateTowardsTailMultimer(chain_idx, r_new); 
        Matrix<double, Dynamic, 3> coords_reptated = melt_config_reptated.getSegment(chain_idx, 0, length);

        // Check the coordinates of the reptated configuration 
        for (int j = 0; j < length - n_reptate; ++j)
        {
            // We are looking for a perfect match, so an unwrapped norm 
            // is fine here 
            REQUIRE_THAT(
                (coords_reptated.row(j) - coords[chain_idx].row(j + n_reptate)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        }
        for (int j = length - n_reptate; j < length; ++j)
        {
            // We are looking for a perfect match, so an unwrapped norm 
            // is fine here 
            REQUIRE_THAT(
                (coords_reptated.row(j) - r_new.row(j - (length - n_reptate))).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            );  
        }

        // Now calculate the backward Rosenbluth weights
        auto result2 = sampler_cosine.getBackwardMultimerReptationRosenbluthWeight(
            chain_idx, ReptationDirection::HEAD, n_reptate, n_candidates,
            coords_reptated
        );
        Matrix<double, Dynamic, Dynamic> candidates = result2.first; 
        double log_reverse_rosenbluth = result2.second;
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * n_reptate);

        // Check that the 0-th candidate for each position is reversion to 
        // the original configuration
        //
        // This involves comparing candidate atom 0 at each position (0, 1, 2)
        // with atoms 2, 1, 0 in the original configuration
        //
        // Note that the indices are mirrored because reversion requires 
        // reptation towards the head  
        for (int j = 0; j < n_reptate; ++j)
        { 
            REQUIRE_THAT(
                (candidates(0, Eigen::seqN(3 * j, 3)) - coords[chain_idx].row(n_reptate - 1 - j)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < n_reptate; ++j)
        {
            Matrix<double, 3, 1> predecessor;

            // If j == 0, then the predecessor is the 0th atom in the 
            // reptated configuration, which is the 3rd atom in the original
            // configuration 
            //
            // If j > 0, then the predecessor is the 2nd or 1st atom in the 
            // original configuration 
            predecessor = coords[chain_idx].row(n_reptate - j); 
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
        // 1) every atom in the original configuration that survives the
        //    reptation move
        // 2) the previous atoms along the segment
        // 3) all atoms along all other chains
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_reverse_rosenbluth_ = 0; 
        for (int j = 0; j < n_reptate; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are reptating towards the head (in reverse), run
                // through all atoms 0, ..., 6 in the reptated configuration, 
                // which are the atoms that survive the reverse reptation move
                //
                // Omit atom 0 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = 1; m < length - n_reptate; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords_reptated.row(m), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords_reptated.row(0), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the reversion segment (except for the immediately preceding
                // atom)
                //
                // Since we are only reptating by 3 monomers, this means that,
                // for j = 2, we add in interactions with atom 2 in the original
                // configuration 
                for (int m = 0; m < j - 1; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords[chain_idx].row(n_reptate - 1 - m),
                        xlen, ylen, zlen
                    );
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all atoms along all 
                // other chains 
                for (int m = 0; m < n_chains; ++m)
                {
                    if (m != chain_idx)
                    {
                        for (int p = 0; p < length; ++p)
                        {
                            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                                r_curr, coords[m].row(p), xlen, ylen, zlen
                            );
                            double r = dvec.norm(); 
                            residual_ijk += lj<double>(
                                r, lj_params["eps"], lj_params["sigma"], true
                            ); 
                        }
                    }
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
    // Reptation moves on 10-mers with a cosine angle potential and a
    // harmonic dihedral potential 
    // --------------------------------------------------------------- //
    melt_config = generateKMerMelt<double>(
        length, n_chains, lj_params, fene_params, AngleMode::COSINE,
        cosine_params, dihedral_params, intra_collision_threshold,
        inter_collision_threshold, max_tries_per_atom, max_tries_per_kmer,
        max_tries_per_seed, max_n_backtracks, max_n_restarts, rng,
        uniform_dist, xmax, ymax, zmax, bond_length_cdf
    );
    coords.clear(); 
    for (int i = 0; i < 5; ++i)
    {
        Matrix<double, Dynamic, 3> coords_i = melt_config.getSegment(i, 0, length);  
        REQUIRE(melt_config.getLength(i) == length);
        REQUIRE(coords_i.rows() == length);
        coords.push_back(coords_i); 
    }

    // Initialize sampler instance
    PolymerMeltCBMCSampler<double> sampler_cosine_dihedral(
        melt_config, lj_params, neighbor_threshold, fene_params,
        AngleMode::COSINE, cosine_params, dihedral_params, rng, -xmax, 
        xmax, -ymax, ymax, -zmax, zmax, bond_length_cdf
    );  

    // Try generating a collection of 3-mer reptation moves at the head
    // of chain 0
    n_reptate = 3;
    chain_idx = 0; 
    for (int i = 0; i < n_moves; ++i)
    {
        // First apply the reptation move to generate a new configuration 
        auto result = sampler_cosine_dihedral.generateForwardMultimerReptationMove(
            chain_idx, ReptationDirection::HEAD, n_reptate, n_candidates
        );
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);
        PolymerMeltConfiguration<double> melt_config_reptated(melt_config); 
        melt_config_reptated.reptateTowardsHeadMultimer(
            chain_idx, r_new.colwise().reverse()
        );
        Matrix<double, Dynamic, 3> coords_reptated = melt_config_reptated.getSegment(chain_idx, 0, length);

        // Check the coordinates of the reptated configuration 
        for (int j = 0; j < n_reptate; ++j)
        {
            // We are looking for a perfect match, so an unwrapped norm 
            // is fine here 
            REQUIRE_THAT(
                (coords_reptated.row(j) - r_new.row(n_reptate - 1 - j)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        }
        for (int j = n_reptate; j < length; ++j)
        {
            // We are looking for a perfect match, so an unwrapped norm 
            // is fine here 
            REQUIRE_THAT(
                (coords_reptated.row(j) - coords[chain_idx].row(j - n_reptate)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            );  
        }

        // Now calculate the backward Rosenbluth weights
        auto result2 = sampler_cosine_dihedral.getBackwardMultimerReptationRosenbluthWeight(
            chain_idx, ReptationDirection::TAIL, n_reptate, n_candidates,
            coords_reptated
        );
        Matrix<double, Dynamic, Dynamic> candidates = result2.first; 
        double log_reverse_rosenbluth = result2.second;
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * n_reptate);

        // Check that the 0-th candidate for each position is reversion to 
        // the original configuration
        //
        // This involves comparing candidate atom 0 at each position (0, 1, 2)
        // with atoms 7, 8, 9 in the original configuration 
        for (int j = 0; j < n_reptate; ++j)
        { 
            // We are looking for a perfect match, so an unwrapped norm 
            // is fine here 
            REQUIRE_THAT(
                (candidates(0, Eigen::seqN(3 * j, 3)) - coords[chain_idx].row(length - n_reptate + j)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < n_reptate; ++j)
        {
            Matrix<double, 3, 1> predecessor;

            // If j == 0, then the predecessor is the last atom in the 
            // reptated configuration, which is the 6th atom in the original
            // configuration 
            //
            // If j > 0, then the predecessor is the 7th or 8th atom in the
            // original configuration 
            predecessor = coords[chain_idx].row(length - 1 - n_reptate + j);
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
        // 1) every atom in the original configuration that survives the
        //    reptation move
        // 2) the previous atoms along the segment
        // 3) all atoms along all other chains
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_reverse_rosenbluth_ = 0; 
        for (int j = 0; j < n_reptate; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are reptating towards the tail (in reverse), run
                // through all atoms 3, ... 9 in the reptated configuration, 
                // which are the atoms that survive the reverse reptation move
                //
                // Omit atom 9 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = n_reptate; m < length - 1; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords_reptated.row(m), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm();
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords_reptated.row(length - 1),
                        xlen, ylen, zlen
                    );
                    double r = dvec.norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the reversion segment (except for the immediately preceding
                // atom)
                //
                // Since we are only reptating by 3 monomers, this means that,
                // for j = 2, we add in interactions with atom 7 in the original
                // configuration 
                for (int m = 0; m < j - 1; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords[chain_idx].row(length - n_reptate + m),
                        xlen, ylen, zlen
                    );
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all atoms along all 
                // other chains 
                for (int m = 0; m < n_chains; ++m)
                {
                    if (m != chain_idx)
                    {
                        for (int p = 0; p < length; ++p)
                        {
                            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                                r_curr, coords[m].row(p), xlen, ylen, zlen
                            );
                            double r = dvec.norm(); 
                            residual_ijk += lj<double>(
                                r, lj_params["eps"], lj_params["sigma"], true
                            ); 
                        }
                    }
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

    // Try generating a collection of 3-mer reptation moves at the tail 
    for (int i = 0; i < n_moves; ++i)
    {
        // First apply the reptation move to generate a new configuration 
        auto result = sampler_cosine_dihedral.generateForwardMultimerReptationMove(
            chain_idx, ReptationDirection::TAIL, n_reptate, n_candidates
        );
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);
        PolymerMeltConfiguration<double> melt_config_reptated(melt_config); 
        melt_config_reptated.reptateTowardsTailMultimer(chain_idx, r_new); 
        Matrix<double, Dynamic, 3> coords_reptated = melt_config_reptated.getSegment(chain_idx, 0, length);

        // Check the coordinates of the reptated configuration 
        for (int j = 0; j < length - n_reptate; ++j)
        {
            // We are looking for a perfect match, so an unwrapped norm 
            // is fine here 
            REQUIRE_THAT(
                (coords_reptated.row(j) - coords[chain_idx].row(j + n_reptate)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        }
        for (int j = length - n_reptate; j < length; ++j)
        {
            // We are looking for a perfect match, so an unwrapped norm 
            // is fine here 
            REQUIRE_THAT(
                (coords_reptated.row(j) - r_new.row(j - (length - n_reptate))).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            );  
        }

        // Now calculate the backward Rosenbluth weights
        auto result2 = sampler_cosine_dihedral.getBackwardMultimerReptationRosenbluthWeight(
            chain_idx, ReptationDirection::HEAD, n_reptate, n_candidates,
            coords_reptated
        );
        Matrix<double, Dynamic, Dynamic> candidates = result2.first; 
        double log_reverse_rosenbluth = result2.second;
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * n_reptate);

        // Check that the 0-th candidate for each position is reversion to 
        // the original configuration
        //
        // This involves comparing candidate atom 0 at each position (0, 1, 2)
        // with atoms 2, 1, 0 in the original configuration
        //
        // Note that the indices are mirrored because reversion requires 
        // reptation towards the head  
        for (int j = 0; j < n_reptate; ++j)
        { 
            REQUIRE_THAT(
                (candidates(0, Eigen::seqN(3 * j, 3)) - coords[chain_idx].row(n_reptate - 1 - j)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < n_reptate; ++j)
        {
            Matrix<double, 3, 1> predecessor;

            // If j == 0, then the predecessor is the 0th atom in the 
            // reptated configuration, which is the 3rd atom in the original
            // configuration 
            //
            // If j > 0, then the predecessor is the 2nd or 1st atom in the 
            // original configuration 
            predecessor = coords[chain_idx].row(n_reptate - j); 
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
        // 1) every atom in the original configuration that survives the
        //    reptation move
        // 2) the previous atoms along the segment
        // 3) all atoms along all other chains
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_reverse_rosenbluth_ = 0; 
        for (int j = 0; j < n_reptate; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are reptating towards the head (in reverse), run
                // through all atoms 0, ..., 6 in the reptated configuration, 
                // which are the atoms that survive the reverse reptation move
                //
                // Omit atom 0 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = 1; m < length - n_reptate; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords_reptated.row(m), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords_reptated.row(0), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the reversion segment (except for the immediately preceding
                // atom)
                //
                // Since we are only reptating by 3 monomers, this means that,
                // for j = 2, we add in interactions with atom 2 in the original
                // configuration 
                for (int m = 0; m < j - 1; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords[chain_idx].row(n_reptate - 1 - m),
                        xlen, ylen, zlen
                    );
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all atoms along all 
                // other chains 
                for (int m = 0; m < n_chains; ++m)
                {
                    if (m != chain_idx)
                    {
                        for (int p = 0; p < length; ++p)
                        {
                            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                                r_curr, coords[m].row(p), xlen, ylen, zlen
                            );
                            double r = dvec.norm(); 
                            residual_ijk += lj<double>(
                                r, lj_params["eps"], lj_params["sigma"], true
                            ); 
                        }
                    }
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
    // Reptation moves on 10-mers with a Gaussian angle potential and a
    // harmonic dihedral potential 
    // --------------------------------------------------------------- //
    melt_config = generateKMerMelt<double>(
        length, n_chains, lj_params, fene_params, AngleMode::GAUSSIAN,
        gaussian_params, dihedral_params, intra_collision_threshold,
        inter_collision_threshold, max_tries_per_atom, max_tries_per_kmer,
        max_tries_per_seed, max_n_backtracks, max_n_restarts, rng,
        uniform_dist, xmax, ymax, zmax, bond_length_cdf
    );
    coords.clear(); 
    for (int i = 0; i < 5; ++i)
    {
        Matrix<double, Dynamic, 3> coords_i = melt_config.getSegment(i, 0, length);  
        REQUIRE(melt_config.getLength(i) == length);
        REQUIRE(coords_i.rows() == length);
        coords.push_back(coords_i); 
    }

    // Initialize sampler instance
    PolymerMeltCBMCSampler<double> sampler_gaussian(
        melt_config, lj_params, neighbor_threshold, fene_params,
        AngleMode::GAUSSIAN, gaussian_params, dihedral_params, rng, -xmax, 
        xmax, -ymax, ymax, -zmax, zmax, bond_length_cdf
    );  

    // Try generating a collection of 3-mer reptation moves at the head
    // of chain 0
    n_reptate = 3;
    chain_idx = 0; 
    for (int i = 0; i < n_moves; ++i)
    {
        // First apply the reptation move to generate a new configuration 
        auto result = sampler_gaussian.generateForwardMultimerReptationMove(
            chain_idx, ReptationDirection::HEAD, n_reptate, n_candidates
        );
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);
        PolymerMeltConfiguration<double> melt_config_reptated(melt_config); 
        melt_config_reptated.reptateTowardsHeadMultimer(
            chain_idx, r_new.colwise().reverse()
        );
        Matrix<double, Dynamic, 3> coords_reptated = melt_config_reptated.getSegment(chain_idx, 0, length);

        // Check the coordinates of the reptated configuration 
        for (int j = 0; j < n_reptate; ++j)
        {
            // We are looking for a perfect match, so an unwrapped norm 
            // is fine here 
            REQUIRE_THAT(
                (coords_reptated.row(j) - r_new.row(n_reptate - 1 - j)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        }
        for (int j = n_reptate; j < length; ++j)
        {
            // We are looking for a perfect match, so an unwrapped norm 
            // is fine here 
            REQUIRE_THAT(
                (coords_reptated.row(j) - coords[chain_idx].row(j - n_reptate)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            );  
        }

        // Now calculate the backward Rosenbluth weights
        auto result2 = sampler_gaussian.getBackwardMultimerReptationRosenbluthWeight(
            chain_idx, ReptationDirection::TAIL, n_reptate, n_candidates,
            coords_reptated
        );
        Matrix<double, Dynamic, Dynamic> candidates = result2.first; 
        double log_reverse_rosenbluth = result2.second;
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * n_reptate);

        // Check that the 0-th candidate for each position is reversion to 
        // the original configuration
        //
        // This involves comparing candidate atom 0 at each position (0, 1, 2)
        // with atoms 7, 8, 9 in the original configuration 
        for (int j = 0; j < n_reptate; ++j)
        { 
            // We are looking for a perfect match, so an unwrapped norm 
            // is fine here 
            REQUIRE_THAT(
                (candidates(0, Eigen::seqN(3 * j, 3)) - coords[chain_idx].row(length - n_reptate + j)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < n_reptate; ++j)
        {
            Matrix<double, 3, 1> predecessor;

            // If j == 0, then the predecessor is the last atom in the 
            // reptated configuration, which is the 6th atom in the original
            // configuration 
            //
            // If j > 0, then the predecessor is the 7th or 8th atom in the
            // original configuration 
            predecessor = coords[chain_idx].row(length - 1 - n_reptate + j);
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
        // 1) every atom in the original configuration that survives the
        //    reptation move
        // 2) the previous atoms along the segment
        // 3) all atoms along all other chains
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_reverse_rosenbluth_ = 0; 
        for (int j = 0; j < n_reptate; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are reptating towards the tail (in reverse), run
                // through all atoms 3, ... 9 in the reptated configuration, 
                // which are the atoms that survive the reverse reptation move
                //
                // Omit atom 9 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = n_reptate; m < length - 1; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords_reptated.row(m), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm();
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords_reptated.row(length - 1),
                        xlen, ylen, zlen
                    );
                    double r = dvec.norm();  
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the reversion segment (except for the immediately preceding
                // atom)
                //
                // Since we are only reptating by 3 monomers, this means that,
                // for j = 2, we add in interactions with atom 7 in the original
                // configuration 
                for (int m = 0; m < j - 1; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords[chain_idx].row(length - n_reptate + m),
                        xlen, ylen, zlen
                    );
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all atoms along all 
                // other chains 
                for (int m = 0; m < n_chains; ++m)
                {
                    if (m != chain_idx)
                    {
                        for (int p = 0; p < length; ++p)
                        {
                            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                                r_curr, coords[m].row(p), xlen, ylen, zlen
                            );
                            double r = dvec.norm(); 
                            residual_ijk += lj<double>(
                                r, lj_params["eps"], lj_params["sigma"], true
                            ); 
                        }
                    }
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

    // Try generating a collection of 3-mer reptation moves at the tail 
    for (int i = 0; i < n_moves; ++i)
    {
        // First apply the reptation move to generate a new configuration 
        auto result = sampler_gaussian.generateForwardMultimerReptationMove(
            chain_idx, ReptationDirection::TAIL, n_reptate, n_candidates
        );
        Matrix<double, Dynamic, 3> r_new = std::get<1>(result);
        PolymerMeltConfiguration<double> melt_config_reptated(melt_config); 
        melt_config_reptated.reptateTowardsTailMultimer(chain_idx, r_new); 
        Matrix<double, Dynamic, 3> coords_reptated = melt_config_reptated.getSegment(chain_idx, 0, length);

        // Check the coordinates of the reptated configuration 
        for (int j = 0; j < length - n_reptate; ++j)
        {
            // We are looking for a perfect match, so an unwrapped norm 
            // is fine here 
            REQUIRE_THAT(
                (coords_reptated.row(j) - coords[chain_idx].row(j + n_reptate)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        }
        for (int j = length - n_reptate; j < length; ++j)
        {
            // We are looking for a perfect match, so an unwrapped norm 
            // is fine here 
            REQUIRE_THAT(
                (coords_reptated.row(j) - r_new.row(j - (length - n_reptate))).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            );  
        }

        // Now calculate the backward Rosenbluth weights
        auto result2 = sampler_gaussian.getBackwardMultimerReptationRosenbluthWeight(
            chain_idx, ReptationDirection::HEAD, n_reptate, n_candidates,
            coords_reptated
        );
        Matrix<double, Dynamic, Dynamic> candidates = result2.first; 
        double log_reverse_rosenbluth = result2.second;
        REQUIRE(candidates.rows() == n_candidates); 
        REQUIRE(candidates.cols() == 3 * n_reptate);

        // Check that the 0-th candidate for each position is reversion to 
        // the original configuration
        //
        // This involves comparing candidate atom 0 at each position (0, 1, 2)
        // with atoms 2, 1, 0 in the original configuration
        //
        // Note that the indices are mirrored because reversion requires 
        // reptation towards the head  
        for (int j = 0; j < n_reptate; ++j)
        { 
            REQUIRE_THAT(
                (candidates(0, Eigen::seqN(3 * j, 3)) - coords[chain_idx].row(n_reptate - 1 - j)).norm(), 
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 

        // Check that, for each position j along the segment, the atoms
        // proposed for position j + 1 have valid distances to the j-th atom
        // in the segment
        for (int j = 0; j < n_reptate; ++j)
        {
            Matrix<double, 3, 1> predecessor;

            // If j == 0, then the predecessor is the 0th atom in the 
            // reptated configuration, which is the 3rd atom in the original
            // configuration 
            //
            // If j > 0, then the predecessor is the 2nd or 1st atom in the 
            // original configuration 
            predecessor = coords[chain_idx].row(n_reptate - j); 
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
        // 1) every atom in the original configuration that survives the
        //    reptation move
        // 2) the previous atoms along the segment
        // 3) all atoms along all other chains
        //
        // Then sum up the corresponding Boltzmann weights to get the 
        // Rosenbluth weight for that position; multiply the Rosenbluth
        // weights to get the total forward Rosenbluth factor
        double log_reverse_rosenbluth_ = 0; 
        for (int j = 0; j < n_reptate; ++j)
        {
            // Get the Rosenbluth weight for the j-th position
            double rosenbluth_j = 0;

            // For each candidate atom for the j-th position ... 
            for (int k = 0; k < n_candidates; ++k)
            {
                Matrix<double, 3, 1> r_curr = candidates(k, Eigen::seqN(3 * j, 3)); 

                // Since we are reptating towards the head (in reverse), run
                // through all atoms 0, ..., 6 in the reptated configuration, 
                // which are the atoms that survive the reverse reptation move
                //
                // Omit atom 0 if we are looking at the first atom in the new
                // segment 
                double residual_ijk = 0; 
                for (int m = 1; m < length - n_reptate; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords_reptated.row(m), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
                if (j > 0)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords_reptated.row(0), xlen, ylen, zlen
                    ); 
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all preceding atoms in
                // the reversion segment (except for the immediately preceding
                // atom)
                //
                // Since we are only reptating by 3 monomers, this means that,
                // for j = 2, we add in interactions with atom 2 in the original
                // configuration 
                for (int m = 0; m < j - 1; ++m)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_curr, coords[chain_idx].row(n_reptate - 1 - m),
                        xlen, ylen, zlen
                    );
                    double r = dvec.norm(); 
                    residual_ijk += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }

                // Add in non-bonded interactions with all atoms along all 
                // other chains 
                for (int m = 0; m < n_chains; ++m)
                {
                    if (m != chain_idx)
                    {
                        for (int p = 0; p < length; ++p)
                        {
                            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                                r_curr, coords[m].row(p), xlen, ylen, zlen
                            );
                            double r = dvec.norm(); 
                            residual_ijk += lj<double>(
                                r, lj_params["eps"], lj_params["sigma"], true
                            ); 
                        }
                    }
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
 * Tests for multimer reptation in the PolymerCBMCSampler class via moveOnce().
 *
 * Note that, because the key thermodynamic information regarding the move 
 * generation (Boltzmann weights, Rosenbluth weights/factors, etc.) are lost 
 * in calling the multimer reptation move generation methods, the correctness
 * of these quantities is not tested here, but rather in the above two test 
 * modules. 
 */
TEST_CASE("Tests for multimer reptation", "[moveOnce()]")
{
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;
    const double tol = 1e-8;  

    // Set up potential and sampling parameters
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

    // Define additional parameters for initialization and sampling
    const double intra_collision_threshold = 0.9;    // Slightly less than 2^(1/6) * sigma ~ 1.01
    const double inter_collision_threshold = 0.9; 
    const int max_tries_per_atom = 50;
    const int max_tries_per_kmer = 50; 
    const int max_tries_per_seed = 50; 
    const int max_n_backtracks = 50; 
    const int max_n_restarts = 50; 
    const double xmax = 50.0; 
    const double ymax = 50.0;
    const double zmax = 50.0;
    const double xlen = 2 * xmax; 
    const double ylen = 2 * ymax; 
    const double zlen = 2 * zmax;  
    Matrix<double, Dynamic, 2> bond_length_cdf = getFeneCDF<double>(
        lj_params["eps"], lj_params["sigma"], fene_params["K"], fene_params["R0"],
        kT, 10000
    );

    // --------------------------------------------------------------- //
    // Reptation moves on 10-mers with no angle or dihedral potentials 
    // --------------------------------------------------------------- //
    const int length = 10; 
    const int n_chains = 5;
    PolymerMeltConfiguration<double> melt_config = generateKMerMelt<double>(
        length, n_chains, lj_params, fene_params, AngleMode::COSINE,
        random_params, nodihedral_params, intra_collision_threshold,
        inter_collision_threshold, max_tries_per_atom, max_tries_per_kmer,
        max_tries_per_seed, max_n_backtracks, max_n_restarts, rng,
        uniform_dist, xmax, ymax, zmax, bond_length_cdf
    );
    std::vector<Matrix<double, Dynamic, 3> > coords; 
    for (int i = 0; i < 5; ++i)
    {
        Matrix<double, Dynamic, 3> coords_i = melt_config.getSegment(i, 0, length);  
        REQUIRE(melt_config.getLength(i) == length);
        REQUIRE(coords_i.rows() == length);
        coords.push_back(coords_i); 
    }

    // Initialize sampler instance
    double neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"]; 
    PolymerMeltCBMCSampler<double> sampler_random(
        melt_config, lj_params, neighbor_threshold, fene_params,
        AngleMode::COSINE, random_params, nodihedral_params, rng, -xmax, 
        xmax, -ymax, ymax, -zmax, zmax, bond_length_cdf
    );  

    // Try generating 50 reptation moves ...
    const int n_moves = 50; 
    for (int i = 0; i < n_moves; ++i)
    { 
        // Try reptating by choosing from 10000 3-mer reptation candidate moves
        // on chain i % 5 ...
        //
        // First generate the move 
        const int n_candidates = 10000;
        int n_reptate = 3; 
        int chain_idx = i % 5; 
        auto result = sampler_random.moveOnce(
            n_candidates, chain_idx, CBMCMoveType::MULTIMER_REPTATION, n_reptate
        ); 
        Matrix<double, Dynamic, Dynamic> forward_moves = std::get<0>(result); 
        Matrix<double, Dynamic, Dynamic> reverse_moves = std::get<1>(result);   // Ill-defined 
        int move_idx = std::get<2>(result); 
        double prob_accept = std::get<3>(result); 
        CBMCMoveResult accepted_move = std::get<4>(result);
        ReptationDirection direction = static_cast<ReptationDirection>(
            std::get<5>(result).at("direction")
        );

        // Check that the output is correctly specified 
        REQUIRE(forward_moves.rows() == n_reptate); 
        REQUIRE(forward_moves.cols() == 3); 
        REQUIRE(move_idx == 0); 
        REQUIRE((prob_accept >= 0 && prob_accept <= 1));

        // Generate the reptated configuration
        //
        // Note that rows must be reversed if reptating towards the head 
        PolymerMeltConfiguration<double> melt_config_reptated(melt_config);  
        if (direction == ReptationDirection::HEAD)
            melt_config_reptated.reptateTowardsHeadMultimer(
                chain_idx, forward_moves.colwise().reverse()
            );
        else 
            melt_config_reptated.reptateTowardsTailMultimer(
                chain_idx, forward_moves
            );
        Matrix<double, Dynamic, 3> coords_reptated = melt_config_reptated.getSegment(chain_idx, 0, 10); 

        // Check that the reptated configuration only contains valid bond lengths 
        for (int j = 0; j < length - 1; ++j)
        {
            Matrix<double, 3, 1> r1 = coords_reptated.row(j); 
            Matrix<double, 3, 1> r2 = coords_reptated.row(j + 1); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }
        
        // Check that, if the acceptance probability is 1, the chosen move was taken 
        if (prob_accept == 1)
            REQUIRE(accepted_move == CBMCMoveResult::ACCEPT); 

        // Check that, if the chosen move was taken, the resulting configuration is
        // as expected
        Matrix<double, Dynamic, 3> coords_result = sampler_random.getCoords(chain_idx);  
        if (accepted_move == CBMCMoveResult::ACCEPT)
        {
            // Move was taken 
            for (int j = 0; j < length; ++j)
            {
                REQUIRE_THAT(
                    (coords_result.row(j) - coords_reptated.row(j)).norm(),
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
                    (coords_result.row(j) - coords[chain_idx].row(j)).norm(),
                    Catch::Matchers::WithinAbs(0, tol)
                ); 
            } 
        }

        // Reset the coordinates in the sampler 
        sampler_random.setCoords(chain_idx, coords[chain_idx]); 
    }

    // --------------------------------------------------------------- //
    // Reptation moves on 10-mers with a cosine angle potential and no 
    // dihedral potential 
    // --------------------------------------------------------------- //
    melt_config = generateKMerMelt<double>(
        length, n_chains, lj_params, fene_params, AngleMode::COSINE,
        cosine_params, nodihedral_params, intra_collision_threshold,
        inter_collision_threshold, max_tries_per_atom, max_tries_per_kmer,
        max_tries_per_seed, max_n_backtracks, max_n_restarts, rng,
        uniform_dist, xmax, ymax, zmax, bond_length_cdf
    );
    coords.clear(); 
    for (int i = 0; i < 5; ++i)
    {
        Matrix<double, Dynamic, 3> coords_i = melt_config.getSegment(i, 0, length);  
        REQUIRE(melt_config.getLength(i) == length);
        REQUIRE(coords_i.rows() == length);
        coords.push_back(coords_i); 
    }

    // Initialize sampler instance
    PolymerMeltCBMCSampler<double> sampler_cosine(
        melt_config, lj_params, neighbor_threshold, fene_params,
        AngleMode::COSINE, cosine_params, nodihedral_params, rng, -xmax, 
        xmax, -ymax, ymax, -zmax, zmax, bond_length_cdf
    );  

    // Try generating 50 reptation moves ...
    for (int i = 0; i < n_moves; ++i)
    { 
        // Try reptating by choosing from 10000 3-mer reptation candidate moves
        // on chain i % 5 ...
        //
        // First generate the move 
        const int n_candidates = 10000;
        int n_reptate = 3; 
        int chain_idx = i % 5; 
        auto result = sampler_cosine.moveOnce(
            n_candidates, chain_idx, CBMCMoveType::MULTIMER_REPTATION, n_reptate
        ); 
        Matrix<double, Dynamic, Dynamic> forward_moves = std::get<0>(result); 
        Matrix<double, Dynamic, Dynamic> reverse_moves = std::get<1>(result);   // Ill-defined 
        int move_idx = std::get<2>(result); 
        double prob_accept = std::get<3>(result); 
        CBMCMoveResult accepted_move = std::get<4>(result);
        ReptationDirection direction = static_cast<ReptationDirection>(
            std::get<5>(result).at("direction")
        );

        // Check that the output is correctly specified 
        REQUIRE(forward_moves.rows() == n_reptate); 
        REQUIRE(forward_moves.cols() == 3); 
        REQUIRE(move_idx == 0); 
        REQUIRE((prob_accept >= 0 && prob_accept <= 1));

        // Generate the reptated configuration
        //
        // Note that rows must be reversed if reptating towards the head 
        PolymerMeltConfiguration<double> melt_config_reptated(melt_config);  
        if (direction == ReptationDirection::HEAD)
            melt_config_reptated.reptateTowardsHeadMultimer(
                chain_idx, forward_moves.colwise().reverse()
            );
        else 
            melt_config_reptated.reptateTowardsTailMultimer(
                chain_idx, forward_moves
            );
        Matrix<double, Dynamic, 3> coords_reptated = melt_config_reptated.getSegment(chain_idx, 0, 10); 

        // Check that the reptated configuration only contains valid bond lengths 
        for (int j = 0; j < length - 1; ++j)
        {
            Matrix<double, 3, 1> r1 = coords_reptated.row(j); 
            Matrix<double, 3, 1> r2 = coords_reptated.row(j + 1); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }
        
        // Check that, if the acceptance probability is 1, the chosen move was taken 
        if (prob_accept == 1)
            REQUIRE(accepted_move == CBMCMoveResult::ACCEPT); 

        // Check that, if the chosen move was taken, the resulting configuration is
        // as expected
        Matrix<double, Dynamic, 3> coords_result = sampler_cosine.getCoords(chain_idx);  
        if (accepted_move == CBMCMoveResult::ACCEPT)
        {
            // Move was taken 
            for (int j = 0; j < length; ++j)
            {
                REQUIRE_THAT(
                    (coords_result.row(j) - coords_reptated.row(j)).norm(),
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
                    (coords_result.row(j) - coords[chain_idx].row(j)).norm(),
                    Catch::Matchers::WithinAbs(0, tol)
                ); 
            } 
        }

        // Reset the coordinates in the sampler 
        sampler_cosine.setCoords(chain_idx, coords[chain_idx]); 
    }

    // --------------------------------------------------------------- //
    // Reptation moves on 10-mers with a cosine angle potential and a
    // harmonic dihedral potential 
    // --------------------------------------------------------------- //
    melt_config = generateKMerMelt<double>(
        length, n_chains, lj_params, fene_params, AngleMode::COSINE,
        cosine_params, dihedral_params, intra_collision_threshold,
        inter_collision_threshold, max_tries_per_atom, max_tries_per_kmer,
        max_tries_per_seed, max_n_backtracks, max_n_restarts, rng,
        uniform_dist, xmax, ymax, zmax, bond_length_cdf
    );
    coords.clear(); 
    for (int i = 0; i < 5; ++i)
    {
        Matrix<double, Dynamic, 3> coords_i = melt_config.getSegment(i, 0, length);  
        REQUIRE(melt_config.getLength(i) == length);
        REQUIRE(coords_i.rows() == length);
        coords.push_back(coords_i); 
    }

    // Initialize sampler instance
    PolymerMeltCBMCSampler<double> sampler_cosine_dihedral(
        melt_config, lj_params, neighbor_threshold, fene_params,
        AngleMode::COSINE, cosine_params, dihedral_params, rng, -xmax, 
        xmax, -ymax, ymax, -zmax, zmax, bond_length_cdf
    );  

    // Try generating 50 reptation moves ...
    for (int i = 0; i < n_moves; ++i)
    { 
        // Try reptating by choosing from 10000 3-mer reptation candidate moves
        // on chain i % 5 ...
        //
        // First generate the move 
        const int n_candidates = 10000;
        int n_reptate = 3; 
        int chain_idx = i % 5; 
        auto result = sampler_cosine_dihedral.moveOnce(
            n_candidates, chain_idx, CBMCMoveType::MULTIMER_REPTATION, n_reptate
        ); 
        Matrix<double, Dynamic, Dynamic> forward_moves = std::get<0>(result); 
        Matrix<double, Dynamic, Dynamic> reverse_moves = std::get<1>(result);   // Ill-defined 
        int move_idx = std::get<2>(result); 
        double prob_accept = std::get<3>(result); 
        CBMCMoveResult accepted_move = std::get<4>(result);
        ReptationDirection direction = static_cast<ReptationDirection>(
            std::get<5>(result).at("direction")
        );

        // Check that the output is correctly specified 
        REQUIRE(forward_moves.rows() == n_reptate); 
        REQUIRE(forward_moves.cols() == 3); 
        REQUIRE(move_idx == 0); 
        REQUIRE((prob_accept >= 0 && prob_accept <= 1));

        // Generate the reptated configuration
        //
        // Note that rows must be reversed if reptating towards the head 
        PolymerMeltConfiguration<double> melt_config_reptated(melt_config);  
        if (direction == ReptationDirection::HEAD)
            melt_config_reptated.reptateTowardsHeadMultimer(
                chain_idx, forward_moves.colwise().reverse()
            );
        else 
            melt_config_reptated.reptateTowardsTailMultimer(
                chain_idx, forward_moves
            );
        Matrix<double, Dynamic, 3> coords_reptated = melt_config_reptated.getSegment(chain_idx, 0, 10); 

        // Check that the reptated configuration only contains valid bond lengths 
        for (int j = 0; j < length - 1; ++j)
        {
            Matrix<double, 3, 1> r1 = coords_reptated.row(j); 
            Matrix<double, 3, 1> r2 = coords_reptated.row(j + 1); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }
        
        // Check that, if the acceptance probability is 1, the chosen move was taken 
        if (prob_accept == 1)
            REQUIRE(accepted_move == CBMCMoveResult::ACCEPT); 

        // Check that, if the chosen move was taken, the resulting configuration is
        // as expected
        Matrix<double, Dynamic, 3> coords_result = sampler_cosine_dihedral.getCoords(chain_idx);  
        if (accepted_move == CBMCMoveResult::ACCEPT)
        {
            // Move was taken 
            for (int j = 0; j < length; ++j)
            {
                REQUIRE_THAT(
                    (coords_result.row(j) - coords_reptated.row(j)).norm(),
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
                    (coords_result.row(j) - coords[chain_idx].row(j)).norm(),
                    Catch::Matchers::WithinAbs(0, tol)
                ); 
            } 
        }

        // Reset the coordinates in the sampler 
        sampler_cosine_dihedral.setCoords(chain_idx, coords[chain_idx]); 
    }

    // --------------------------------------------------------------- //
    // Reptation moves on 10-mer with a Gaussian angle potential and a
    // harmonic dihedral potential 
    // --------------------------------------------------------------- //
    melt_config = generateKMerMelt<double>(
        length, n_chains, lj_params, fene_params, AngleMode::GAUSSIAN,
        gaussian_params, dihedral_params, intra_collision_threshold,
        inter_collision_threshold, max_tries_per_atom, max_tries_per_kmer,
        max_tries_per_seed, max_n_backtracks, max_n_restarts, rng,
        uniform_dist, xmax, ymax, zmax, bond_length_cdf
    );
    coords.clear(); 
    for (int i = 0; i < 5; ++i)
    {
        Matrix<double, Dynamic, 3> coords_i = melt_config.getSegment(i, 0, length);  
        REQUIRE(melt_config.getLength(i) == length);
        REQUIRE(coords_i.rows() == length);
        coords.push_back(coords_i); 
    }

    // Initialize sampler instance
    PolymerMeltCBMCSampler<double> sampler_gaussian(
        melt_config, lj_params, neighbor_threshold, fene_params,
        AngleMode::GAUSSIAN, gaussian_params, dihedral_params, rng, -xmax, 
        xmax, -ymax, ymax, -zmax, zmax, bond_length_cdf
    );  

    // Try generating 50 reptation moves ...
    for (int i = 0; i < n_moves; ++i)
    { 
        // Try reptating by choosing from 10000 3-mer reptation candidate moves
        // on chain i % 5 ...
        //
        // First generate the move 
        const int n_candidates = 10000;
        int n_reptate = 3; 
        int chain_idx = i % 5; 
        auto result = sampler_gaussian.moveOnce(
            n_candidates, chain_idx, CBMCMoveType::MULTIMER_REPTATION, n_reptate
        ); 
        Matrix<double, Dynamic, Dynamic> forward_moves = std::get<0>(result); 
        Matrix<double, Dynamic, Dynamic> reverse_moves = std::get<1>(result);   // Ill-defined 
        int move_idx = std::get<2>(result); 
        double prob_accept = std::get<3>(result); 
        CBMCMoveResult accepted_move = std::get<4>(result);
        ReptationDirection direction = static_cast<ReptationDirection>(
            std::get<5>(result).at("direction")
        );

        // Check that the output is correctly specified 
        REQUIRE(forward_moves.rows() == n_reptate); 
        REQUIRE(forward_moves.cols() == 3); 
        REQUIRE(move_idx == 0); 
        REQUIRE((prob_accept >= 0 && prob_accept <= 1));

        // Generate the reptated configuration
        //
        // Note that rows must be reversed if reptating towards the head 
        PolymerMeltConfiguration<double> melt_config_reptated(melt_config);  
        if (direction == ReptationDirection::HEAD)
            melt_config_reptated.reptateTowardsHeadMultimer(
                chain_idx, forward_moves.colwise().reverse()
            );
        else 
            melt_config_reptated.reptateTowardsTailMultimer(
                chain_idx, forward_moves
            );
        Matrix<double, Dynamic, 3> coords_reptated = melt_config_reptated.getSegment(chain_idx, 0, 10); 

        // Check that the reptated configuration only contains valid bond lengths 
        for (int j = 0; j < length - 1; ++j)
        {
            Matrix<double, 3, 1> r1 = coords_reptated.row(j); 
            Matrix<double, 3, 1> r2 = coords_reptated.row(j + 1); 
            REQUIRE((r1 - r2).norm() < fene_params["R0"]); 
        }
        
        // Check that, if the acceptance probability is 1, the chosen move was taken 
        if (prob_accept == 1)
            REQUIRE(accepted_move == CBMCMoveResult::ACCEPT); 

        // Check that, if the chosen move was taken, the resulting configuration is
        // as expected
        Matrix<double, Dynamic, 3> coords_result = sampler_gaussian.getCoords(chain_idx);  
        if (accepted_move == CBMCMoveResult::ACCEPT)
        {
            // Move was taken 
            for (int j = 0; j < length; ++j)
            {
                REQUIRE_THAT(
                    (coords_result.row(j) - coords_reptated.row(j)).norm(),
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
                    (coords_result.row(j) - coords[chain_idx].row(j)).norm(),
                    Catch::Matchers::WithinAbs(0, tol)
                ); 
            } 
        }

        // Reset the coordinates in the sampler 
        sampler_gaussian.setCoords(chain_idx, coords[chain_idx]); 
    }
}

