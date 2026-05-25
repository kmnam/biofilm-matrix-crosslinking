/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     5/24/2026
 */

#include <iostream>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/random.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/utils.hpp"
#include "../../include/polymerMelt.hpp"
#include "../../include/cbmcMelt.hpp"

using namespace Eigen;

/**
 * Tests for PolymerMeltCBMCSampler::generateReptationMoves().
 *
 * This module includes tests for the *forward* version of this method.
 */
TEST_CASE("Tests for forward reptation move generation", "[generateReptationMoves()]")
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
    // Reptation moves on 10-mers with no angle and dihedral potentials 
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

    // Try generating 50 reptation moves at the head of chain 2
    int chain_idx = 2;
    const int n_candidates = 50;  
    auto result = sampler_random.generateReptationMoves(
        chain_idx, ReptationDirection::HEAD, n_candidates
    );
    Matrix<double, Dynamic, Dynamic> r_new = result.first; 
    Matrix<double, Dynamic, 1> residuals = result.second; 
    REQUIRE(r_new.rows() == n_candidates);
    REQUIRE(r_new.cols() == 3);  
    REQUIRE(residuals.size() == n_candidates); 

    // Check that each new atom has a valid distance to the 0-th atom
    for (int i = 0; i < n_candidates; ++i) 
        REQUIRE((r_new.row(i) - coords[chain_idx].row(0)).norm() < fene_params["R0"]); 

    // Check the residual energy of each proposed move 
    //
    // This residual energy is the total non-bonded interaction energy between
    // the new atom and every atom that survives the reptation move
    for (int i = 0; i < n_candidates; ++i)
    {
        PolymerConfiguration<double> config_reptated(
            coords[chain_idx], Units::NANO, 300.0
        ); 
        config_reptated.reptateTowardsHead(r_new.row(i)); 

        // Re-calculate this residual energy 
        //
        // If we are reptating towards the head, this consists of:
        // - the non-bonded interaction energy between the new atom and atoms
        //   2, ..., 9 in the reptated configuration (the new atom is atom 0)
        // - the non-bonded interaction energy between the new atom and every
        //   atom on every other chain 
        //
        // Start with the intra-chain interaction energy 
        Matrix<double, Dynamic, 3> coords_reptated = config_reptated.getSegment(0, length); 
        double residual = 0;  
        for (int j = 2; j < length; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                r_new.row(i), coords_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            residual += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }

        // Then calculate the inter-chain interaction energy 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_new.row(i), coords[j].row(k), xlen, ylen, zlen 
                    );
                    double r = dvec.norm(); 
                    residual += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
            }
        }
        REQUIRE_THAT(residuals(i), Catch::Matchers::WithinRel(residual, tol)); 
    }

    // Try generating 50 reptation moves at the tail of chain 3
    chain_idx = 3;  
    result = sampler_random.generateReptationMoves(
        chain_idx, ReptationDirection::TAIL, n_candidates
    );
    r_new = result.first; 
    residuals = result.second;
    REQUIRE(r_new.rows() == n_candidates);
    REQUIRE(r_new.cols() == 3);  
    REQUIRE(residuals.size() == n_candidates); 

    // Check that each new atom has a valid distance to the 0-th atom
    int tail_idx = length - 1; 
    for (int i = 0; i < n_candidates; ++i)
        REQUIRE((r_new.row(i) - coords[chain_idx].row(tail_idx)).norm() < fene_params["R0"]); 

    // Check the residual energy of each proposed move 
    //
    // This residual energy is the total non-bonded interaction energy between
    // the new atom and every atom that survives the reptation move
    for (int i = 0; i < n_candidates; ++i)
    {
        PolymerConfiguration<double> config_reptated(
            coords[chain_idx], Units::NANO, 300.0
        ); 
        config_reptated.reptateTowardsTail(r_new.row(i)); 

        // Re-calculate this residual energy 
        //
        // If we are reptating towards the tail, this consists of:
        // - the non-bonded interaction energy between the new atom and atoms
        //   0, ..., 7 in the reptated configuration (the new atom is atom 9)
        // - the non-bonded interaction energy between the new atom and every
        //   atom on every other chain 
        //
        // Start with the intra-chain interaction energy 
        Matrix<double, Dynamic, 3> coords_reptated = config_reptated.getSegment(0, length); 
        double residual = 0;  
        for (int j = 0; j < length - 2; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                r_new.row(i), coords_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            residual += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }

        // Then calculate the inter-chain interaction energy 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_new.row(i), coords[j].row(k), xlen, ylen, zlen 
                    );
                    double r = dvec.norm(); 
                    residual += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
            }
        }
        REQUIRE_THAT(residuals(i), Catch::Matchers::WithinRel(residual, tol));  
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

    // Try generating 50 reptation moves at the head of chain 2
    chain_idx = 2;
    result = sampler_cosine.generateReptationMoves(
        chain_idx, ReptationDirection::HEAD, n_candidates
    );
    r_new = result.first; 
    residuals = result.second; 
    REQUIRE(r_new.rows() == n_candidates);
    REQUIRE(r_new.cols() == 3);  
    REQUIRE(residuals.size() == n_candidates); 

    // Check that each new atom has a valid distance to the 0-th atom
    for (int i = 0; i < n_candidates; ++i) 
        REQUIRE((r_new.row(i) - coords[chain_idx].row(0)).norm() < fene_params["R0"]); 

    // Check the residual energy of each proposed move 
    //
    // This residual energy is the total non-bonded interaction energy between
    // the new atom and every atom that survives the reptation move
    for (int i = 0; i < n_candidates; ++i)
    {
        PolymerConfiguration<double> config_reptated(
            coords[chain_idx], Units::NANO, 300.0
        ); 
        config_reptated.reptateTowardsHead(r_new.row(i)); 

        // Re-calculate this residual energy 
        //
        // If we are reptating towards the head, this consists of:
        // - the non-bonded interaction energy between the new atom and atoms
        //   2, ..., 9 in the reptated configuration (the new atom is atom 0)
        // - the non-bonded interaction energy between the new atom and every
        //   atom on every other chain 
        //
        // Start with the intra-chain interaction energy 
        Matrix<double, Dynamic, 3> coords_reptated = config_reptated.getSegment(0, length); 
        double residual = 0;  
        for (int j = 2; j < length; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                r_new.row(i), coords_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            residual += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }

        // Then calculate the inter-chain interaction energy 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_new.row(i), coords[j].row(k), xlen, ylen, zlen 
                    );
                    double r = dvec.norm(); 
                    residual += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
            }
        }
        REQUIRE_THAT(residuals(i), Catch::Matchers::WithinRel(residual, tol)); 
    }

    // Try generating 50 reptation moves at the tail of chain 3
    chain_idx = 3;  
    result = sampler_cosine.generateReptationMoves(
        chain_idx, ReptationDirection::TAIL, n_candidates
    );
    r_new = result.first; 
    residuals = result.second;
    REQUIRE(r_new.rows() == n_candidates);
    REQUIRE(r_new.cols() == 3);  
    REQUIRE(residuals.size() == n_candidates); 

    // Check that each new atom has a valid distance to the 0-th atom
    tail_idx = length - 1; 
    for (int i = 0; i < n_candidates; ++i)
        REQUIRE((r_new.row(i) - coords[chain_idx].row(tail_idx)).norm() < fene_params["R0"]); 

    // Check the residual energy of each proposed move 
    //
    // This residual energy is the total non-bonded interaction energy between
    // the new atom and every atom that survives the reptation move
    for (int i = 0; i < n_candidates; ++i)
    {
        PolymerConfiguration<double> config_reptated(
            coords[chain_idx], Units::NANO, 300.0
        ); 
        config_reptated.reptateTowardsTail(r_new.row(i)); 

        // Re-calculate this residual energy 
        //
        // If we are reptating towards the tail, this consists of:
        // - the non-bonded interaction energy between the new atom and atoms
        //   0, ..., 7 in the reptated configuration (the new atom is atom 9)
        // - the non-bonded interaction energy between the new atom and every
        //   atom on every other chain 
        //
        // Start with the intra-chain interaction energy 
        Matrix<double, Dynamic, 3> coords_reptated = config_reptated.getSegment(0, length); 
        double residual = 0;  
        for (int j = 0; j < length - 2; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                r_new.row(i), coords_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            residual += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }

        // Then calculate the inter-chain interaction energy 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_new.row(i), coords[j].row(k), xlen, ylen, zlen 
                    );
                    double r = dvec.norm(); 
                    residual += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
            }
        }
        REQUIRE_THAT(residuals(i), Catch::Matchers::WithinRel(residual, tol));  
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

    // Try generating 50 reptation moves at the head of chain 2
    chain_idx = 2;
    result = sampler_cosine_dihedral.generateReptationMoves(
        chain_idx, ReptationDirection::HEAD, n_candidates
    );
    r_new = result.first; 
    residuals = result.second; 
    REQUIRE(r_new.rows() == n_candidates);
    REQUIRE(r_new.cols() == 3);  
    REQUIRE(residuals.size() == n_candidates); 

    // Check that each new atom has a valid distance to the 0-th atom
    for (int i = 0; i < n_candidates; ++i) 
        REQUIRE((r_new.row(i) - coords[chain_idx].row(0)).norm() < fene_params["R0"]); 

    // Check the residual energy of each proposed move 
    //
    // This residual energy is the total non-bonded interaction energy between
    // the new atom and every atom that survives the reptation move
    for (int i = 0; i < n_candidates; ++i)
    {
        PolymerConfiguration<double> config_reptated(
            coords[chain_idx], Units::NANO, 300.0
        ); 
        config_reptated.reptateTowardsHead(r_new.row(i)); 

        // Re-calculate this residual energy 
        //
        // If we are reptating towards the head, this consists of:
        // - the non-bonded interaction energy between the new atom and atoms
        //   2, ..., 9 in the reptated configuration (the new atom is atom 0)
        // - the non-bonded interaction energy between the new atom and every
        //   atom on every other chain 
        //
        // Start with the intra-chain interaction energy 
        Matrix<double, Dynamic, 3> coords_reptated = config_reptated.getSegment(0, length); 
        double residual = 0;  
        for (int j = 2; j < length; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                r_new.row(i), coords_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            residual += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }

        // Then calculate the inter-chain interaction energy 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_new.row(i), coords[j].row(k), xlen, ylen, zlen 
                    );
                    double r = dvec.norm(); 
                    residual += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
            }
        }
        REQUIRE_THAT(residuals(i), Catch::Matchers::WithinRel(residual, tol)); 
    }

    // Try generating 50 reptation moves at the tail of chain 3
    chain_idx = 3;  
    result = sampler_cosine_dihedral.generateReptationMoves(
        chain_idx, ReptationDirection::TAIL, n_candidates
    );
    r_new = result.first; 
    residuals = result.second;
    REQUIRE(r_new.rows() == n_candidates);
    REQUIRE(r_new.cols() == 3);  
    REQUIRE(residuals.size() == n_candidates); 

    // Check that each new atom has a valid distance to the 0-th atom
    tail_idx = length - 1; 
    for (int i = 0; i < n_candidates; ++i)
        REQUIRE((r_new.row(i) - coords[chain_idx].row(tail_idx)).norm() < fene_params["R0"]); 

    // Check the residual energy of each proposed move 
    //
    // This residual energy is the total non-bonded interaction energy between
    // the new atom and every atom that survives the reptation move
    for (int i = 0; i < n_candidates; ++i)
    {
        PolymerConfiguration<double> config_reptated(
            coords[chain_idx], Units::NANO, 300.0
        ); 
        config_reptated.reptateTowardsTail(r_new.row(i)); 

        // Re-calculate this residual energy 
        //
        // If we are reptating towards the tail, this consists of:
        // - the non-bonded interaction energy between the new atom and atoms
        //   0, ..., 7 in the reptated configuration (the new atom is atom 9)
        // - the non-bonded interaction energy between the new atom and every
        //   atom on every other chain 
        //
        // Start with the intra-chain interaction energy 
        Matrix<double, Dynamic, 3> coords_reptated = config_reptated.getSegment(0, length); 
        double residual = 0;  
        for (int j = 0; j < length - 2; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                r_new.row(i), coords_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            residual += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }

        // Then calculate the inter-chain interaction energy 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_new.row(i), coords[j].row(k), xlen, ylen, zlen 
                    );
                    double r = dvec.norm(); 
                    residual += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
            }
        }
        REQUIRE_THAT(residuals(i), Catch::Matchers::WithinRel(residual, tol));  
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

    // Try generating 50 reptation moves at the head of chain 2
    chain_idx = 2;
    result = sampler_gaussian.generateReptationMoves(
        chain_idx, ReptationDirection::HEAD, n_candidates
    );
    r_new = result.first; 
    residuals = result.second; 
    REQUIRE(r_new.rows() == n_candidates);
    REQUIRE(r_new.cols() == 3);  
    REQUIRE(residuals.size() == n_candidates); 

    // Check that each new atom has a valid distance to the 0-th atom
    for (int i = 0; i < n_candidates; ++i) 
        REQUIRE((r_new.row(i) - coords[chain_idx].row(0)).norm() < fene_params["R0"]); 

    // Check the residual energy of each proposed move 
    //
    // This residual energy is the total non-bonded interaction energy between
    // the new atom and every atom that survives the reptation move
    for (int i = 0; i < n_candidates; ++i)
    {
        PolymerConfiguration<double> config_reptated(
            coords[chain_idx], Units::NANO, 300.0
        ); 
        config_reptated.reptateTowardsHead(r_new.row(i)); 

        // Re-calculate this residual energy 
        //
        // If we are reptating towards the head, this consists of:
        // - the non-bonded interaction energy between the new atom and atoms
        //   2, ..., 9 in the reptated configuration (the new atom is atom 0)
        // - the non-bonded interaction energy between the new atom and every
        //   atom on every other chain 
        //
        // Start with the intra-chain interaction energy 
        Matrix<double, Dynamic, 3> coords_reptated = config_reptated.getSegment(0, length); 
        double residual = 0;  
        for (int j = 2; j < length; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                r_new.row(i), coords_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            residual += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }

        // Then calculate the inter-chain interaction energy 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_new.row(i), coords[j].row(k), xlen, ylen, zlen 
                    );
                    double r = dvec.norm(); 
                    residual += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
            }
        }
        REQUIRE_THAT(residuals(i), Catch::Matchers::WithinRel(residual, tol)); 
    }

    // Try generating 50 reptation moves at the tail of chain 3
    chain_idx = 3;  
    result = sampler_gaussian.generateReptationMoves(
        chain_idx, ReptationDirection::TAIL, n_candidates
    );
    r_new = result.first; 
    residuals = result.second;
    REQUIRE(r_new.rows() == n_candidates);
    REQUIRE(r_new.cols() == 3);  
    REQUIRE(residuals.size() == n_candidates); 

    // Check that each new atom has a valid distance to the 0-th atom
    tail_idx = length - 1; 
    for (int i = 0; i < n_candidates; ++i)
        REQUIRE((r_new.row(i) - coords[chain_idx].row(tail_idx)).norm() < fene_params["R0"]); 

    // Check the residual energy of each proposed move 
    //
    // This residual energy is the total non-bonded interaction energy between
    // the new atom and every atom that survives the reptation move
    for (int i = 0; i < n_candidates; ++i)
    {
        PolymerConfiguration<double> config_reptated(
            coords[chain_idx], Units::NANO, 300.0
        ); 
        config_reptated.reptateTowardsTail(r_new.row(i)); 

        // Re-calculate this residual energy 
        //
        // If we are reptating towards the tail, this consists of:
        // - the non-bonded interaction energy between the new atom and atoms
        //   0, ..., 7 in the reptated configuration (the new atom is atom 9)
        // - the non-bonded interaction energy between the new atom and every
        //   atom on every other chain 
        //
        // Start with the intra-chain interaction energy 
        Matrix<double, Dynamic, 3> coords_reptated = config_reptated.getSegment(0, length); 
        double residual = 0;  
        for (int j = 0; j < length - 2; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                r_new.row(i), coords_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            residual += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }

        // Then calculate the inter-chain interaction energy 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_new.row(i), coords[j].row(k), xlen, ylen, zlen 
                    );
                    double r = dvec.norm(); 
                    residual += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
            }
        }
        REQUIRE_THAT(residuals(i), Catch::Matchers::WithinRel(residual, tol));  
    }
}

/**
 * Tests for PolymerCBMCSampler::generateReptationMoves().
 *
 * This module includes tests for the *backward* version of this method.
 */
TEST_CASE("Tests for reverse reptation move generation", "[generateReptationMoves()]")
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
    // Reptation moves on 10-mers with no angle and dihedral potentials 
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

    // Try generating 50 *forward* reptation moves at the head of chain 1
    int chain_idx = 1;
    const int n_candidates = 50;  
    auto forward_result = sampler_random.generateReptationMoves(
        chain_idx, ReptationDirection::HEAD, n_candidates
    );
    Matrix<double, Dynamic, Dynamic> r_forward = forward_result.first; 
    Matrix<double, Dynamic, 1> residuals_forward = forward_result.second;

    // Choose a reptation move and generate 50 corresponding *backward* 
    // reptation moves 
    int move_idx = 40;
    PolymerConfiguration<double> config_reptated(
        coords[chain_idx], Units::NANO, 300.0
    ); 
    config_reptated.reptateTowardsHead(r_forward.row(move_idx));
    Matrix<double, Dynamic, 3> coords_reptated = config_reptated.getSegment(0, length); 
    auto reverse_result = sampler_random.generateReptationMoves(
        chain_idx, ReptationDirection::TAIL, n_candidates, coords_reptated
    ); 
    Matrix<double, Dynamic, Dynamic> r_reverse = reverse_result.first; 
    Matrix<double, Dynamic, 1> residuals_reverse = reverse_result.second;

    // Check that the 0-th atom is reversion to the original configuration
    //
    // Since we were originally reptating towards the head (reverse = tail), 
    // we are looking for the 9-th atom in the original configuration 
    int tail_idx = length - 1; 
    REQUIRE_THAT(
        (r_reverse.row(0) - coords[chain_idx].row(tail_idx)).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );

    // Check that each new atom has a valid distance to the 8-th atom in the
    // original configuration
    for (int i = 0; i < n_candidates; ++i)
        REQUIRE((r_reverse.row(i) - coords[chain_idx].row(tail_idx - 1)).norm() < fene_params["R0"]);

    // Check the residual energy of each proposed move 
    //
    // This residual energy is the total non-bonded interaction energy between
    // the new atom and every atom that survives the (reverse) reptation move
    for (int i = 0; i < n_candidates; ++i)
    {
        PolymerConfiguration<double> config_reverse_reptated(config_reptated);
        config_reverse_reptated.reptateTowardsTail(r_reverse.row(i));
        Matrix<double, Dynamic, 3> coords_reverse_reptated = config_reverse_reptated.getSegment(0, length); 

        // Check that, if i == 0, reversion was successful
        if (i == 0)
        {
            for (int j = 0; j < length; ++j)
            {
                REQUIRE_THAT(
                    (coords_reverse_reptated.row(j) - coords[chain_idx].row(j)).norm(), 
                    Catch::Matchers::WithinAbs(0, tol)
                ); 
            }
        }

        // Re-calculate the residual energy 
        //
        // If we are reptating towards the tail (in reverse), this consists of:
        // - the non-bonded interaction energy between the new atom and atoms
        //   0, ..., 7 in the reverse-reptated configuration (the new atom is
        //   atom 9)
        // - the non-bonded interaction energy between the new atom and every
        //   atom on every other chain 
        //
        // Start with the intra-chain interaction energy 
        double residual = 0;  
        for (int j = 0; j < length - 2; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                r_reverse.row(i), coords_reverse_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            residual += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }

        // Then calculate the inter-chain interaction energy 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_reverse.row(i), coords[j].row(k), xlen, ylen, zlen 
                    );
                    double r = dvec.norm(); 
                    residual += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
            }
        }
        REQUIRE_THAT(residuals_reverse(i), Catch::Matchers::WithinRel(residual, tol)); 

        // In other words, the intra-chain interaction energy is also the
        // non-bonded interaction energy between the new atom and atoms 1, ..., 8
        // in the once-reptated configuration
        residual = 0; 
        for (int j = 1; j < length - 1; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                r_reverse.row(i), coords_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            residual += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_reverse.row(i), coords[j].row(k), xlen, ylen, zlen 
                    );
                    double r = dvec.norm(); 
                    residual += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
            }
        }
        REQUIRE_THAT(residuals_reverse(i), Catch::Matchers::WithinRel(residual, tol));
    } 
  
    // Try generating 50 *forward* reptation moves at the tail of chain 0
    chain_idx = 0;
    forward_result = sampler_random.generateReptationMoves(
        chain_idx, ReptationDirection::TAIL, n_candidates
    );
    r_forward = forward_result.first; 
    residuals_forward = forward_result.second;

    // Choose a reptation move and generate 50 corresponding *backward* 
    // reptation moves 
    move_idx = 31;
    config_reptated = PolymerConfiguration<double>(
        coords[chain_idx], Units::NANO, 300.0
    ); 
    config_reptated.reptateTowardsTail(r_forward.row(move_idx)); 
    coords_reptated = config_reptated.getSegment(0, length); 
    reverse_result = sampler_random.generateReptationMoves(
        chain_idx, ReptationDirection::HEAD, n_candidates, coords_reptated
    ); 
    r_reverse = reverse_result.first; 
    residuals_reverse = reverse_result.second; 

    // Check that the 0-th atom is reversion to the original configuration 
    // 
    // Since we were originally reptating towards the tail (reverse = head), 
    // we are looking for the 0-th atom in the original configuration 
    REQUIRE_THAT(
        (r_reverse.row(0) - coords[chain_idx].row(0)).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );  

    // Check that each new atom has a valid distance to the 1st atom in the
    // original configuration
    for (int i = 0; i < n_candidates; ++i)
        REQUIRE((r_reverse.row(i) - coords[chain_idx].row(1)).norm() < fene_params["R0"]); 

    // Check the residual energy of each proposed move 
    //
    // This residual energy is the total non-bonded interaction energy between
    // the new atom and every atom that survives the (reverse) reptation move
    for (int i = 0; i < n_candidates; ++i)
    {
        PolymerConfiguration<double> config_reverse_reptated(config_reptated);
        config_reverse_reptated.reptateTowardsHead(r_reverse.row(i));
        Matrix<double, Dynamic, 3> coords_reverse_reptated = config_reverse_reptated.getSegment(0, length);

        // Check that, if i == 0, reversion was successful 
        if (i == 0)
        {
            for (int j = 0; j < length; ++j)
            {
                REQUIRE_THAT(
                    (coords_reverse_reptated.row(j) - coords[chain_idx].row(j)).norm(), 
                    Catch::Matchers::WithinAbs(0, tol)
                ); 
            }
        }  

        // Re-calculate this residual energy 
        //
        // If we are reptating towards the head (in reverse), this consists of:
        // - the non-bonded interaction energy between the new atom and atoms
        //   2, ..., 9 in the reverse-reptated configuration (the new atom is 
        //   atom 0)
        // - the non-bonded interaction energy between the new atom and every
        //   atom on every other chain 
        //
        // Start with the intra-chain interaction energy
        double residual = 0;  
        for (int j = 2; j < length; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                r_reverse.row(i), coords_reverse_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            residual += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }

        // Then calculate the inter-chain interaction energy 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_reverse.row(i), coords[j].row(k), xlen, ylen, zlen 
                    );
                    double r = dvec.norm(); 
                    residual += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
            }
        }
        REQUIRE_THAT(residuals_reverse(i), Catch::Matchers::WithinRel(residual, tol)); 

        // In other words, the intra-chain interaction energy is also the
        // non-bonded interaction energy between the new atom and atoms 1, ..., 8
        // in the once-reptated configuration 
        residual = 0;
        for (int j = 1; j < length - 1; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                r_reverse.row(i), coords_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            residual += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_reverse.row(i), coords[j].row(k), xlen, ylen, zlen 
                    );
                    double r = dvec.norm(); 
                    residual += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
            }
        }
        REQUIRE_THAT(residuals_reverse(i), Catch::Matchers::WithinRel(residual, tol));
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

    // Try generating 50 *forward* reptation moves at the head of chain 1
    chain_idx = 1;
    forward_result = sampler_cosine.generateReptationMoves(
        chain_idx, ReptationDirection::HEAD, n_candidates
    );
    r_forward = forward_result.first; 
    residuals_forward = forward_result.second;

    // Choose a reptation move and generate 50 corresponding *backward* 
    // reptation moves 
    move_idx = 40;
    config_reptated = PolymerConfiguration<double>(
        coords[chain_idx], Units::NANO, 300.0
    ); 
    config_reptated.reptateTowardsHead(r_forward.row(move_idx));
    coords_reptated = config_reptated.getSegment(0, length); 
    reverse_result = sampler_cosine.generateReptationMoves(
        chain_idx, ReptationDirection::TAIL, n_candidates, coords_reptated
    ); 
    r_reverse = reverse_result.first; 
    residuals_reverse = reverse_result.second;

    // Check that the 0-th atom is reversion to the original configuration
    //
    // Since we were originally reptating towards the head (reverse = tail), 
    // we are looking for the 9-th atom in the original configuration 
    tail_idx = length - 1; 
    REQUIRE_THAT(
        (r_reverse.row(0) - coords[chain_idx].row(tail_idx)).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );

    // Check that each new atom has a valid distance to the 8-th atom in the
    // original configuration
    for (int i = 0; i < n_candidates; ++i)
        REQUIRE((r_reverse.row(i) - coords[chain_idx].row(tail_idx - 1)).norm() < fene_params["R0"]);

    // Check the residual energy of each proposed move 
    //
    // This residual energy is the total non-bonded interaction energy between
    // the new atom and every atom that survives the (reverse) reptation move
    for (int i = 0; i < n_candidates; ++i)
    {
        PolymerConfiguration<double> config_reverse_reptated(config_reptated);
        config_reverse_reptated.reptateTowardsTail(r_reverse.row(i));
        Matrix<double, Dynamic, 3> coords_reverse_reptated = config_reverse_reptated.getSegment(0, length); 

        // Check that, if i == 0, reversion was successful
        if (i == 0)
        {
            for (int j = 0; j < length; ++j)
            {
                REQUIRE_THAT(
                    (coords_reverse_reptated.row(j) - coords[chain_idx].row(j)).norm(), 
                    Catch::Matchers::WithinAbs(0, tol)
                ); 
            }
        }

        // Re-calculate the residual energy 
        //
        // If we are reptating towards the tail (in reverse), this consists of:
        // - the non-bonded interaction energy between the new atom and atoms
        //   0, ..., 7 in the reverse-reptated configuration (the new atom is
        //   atom 9)
        // - the non-bonded interaction energy between the new atom and every
        //   atom on every other chain 
        //
        // Start with the intra-chain interaction energy 
        double residual = 0;  
        for (int j = 0; j < length - 2; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                r_reverse.row(i), coords_reverse_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            residual += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }

        // Then calculate the inter-chain interaction energy 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_reverse.row(i), coords[j].row(k), xlen, ylen, zlen 
                    );
                    double r = dvec.norm(); 
                    residual += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
            }
        }
        REQUIRE_THAT(residuals_reverse(i), Catch::Matchers::WithinRel(residual, tol)); 

        // In other words, the intra-chain interaction energy is also the
        // non-bonded interaction energy between the new atom and atoms 1, ..., 8
        // in the once-reptated configuration
        residual = 0; 
        for (int j = 1; j < length - 1; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                r_reverse.row(i), coords_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            residual += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_reverse.row(i), coords[j].row(k), xlen, ylen, zlen 
                    );
                    double r = dvec.norm(); 
                    residual += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
            }
        }
        REQUIRE_THAT(residuals_reverse(i), Catch::Matchers::WithinRel(residual, tol));
    } 
  
    // Try generating 50 *forward* reptation moves at the tail of chain 0
    chain_idx = 0;
    forward_result = sampler_cosine.generateReptationMoves(
        chain_idx, ReptationDirection::TAIL, n_candidates
    );
    r_forward = forward_result.first; 
    residuals_forward = forward_result.second;

    // Choose a reptation move and generate 50 corresponding *backward* 
    // reptation moves 
    move_idx = 31;
    config_reptated = PolymerConfiguration<double>(
        coords[chain_idx], Units::NANO, 300.0
    ); 
    config_reptated.reptateTowardsTail(r_forward.row(move_idx)); 
    coords_reptated = config_reptated.getSegment(0, length); 
    reverse_result = sampler_cosine.generateReptationMoves(
        chain_idx, ReptationDirection::HEAD, n_candidates, coords_reptated
    ); 
    r_reverse = reverse_result.first; 
    residuals_reverse = reverse_result.second; 

    // Check that the 0-th atom is reversion to the original configuration 
    // 
    // Since we were originally reptating towards the tail (reverse = head), 
    // we are looking for the 0-th atom in the original configuration 
    REQUIRE_THAT(
        (r_reverse.row(0) - coords[chain_idx].row(0)).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );  

    // Check that each new atom has a valid distance to the 1st atom in the
    // original configuration
    for (int i = 0; i < n_candidates; ++i)
        REQUIRE((r_reverse.row(i) - coords[chain_idx].row(1)).norm() < fene_params["R0"]); 

    // Check the residual energy of each proposed move 
    //
    // This residual energy is the total non-bonded interaction energy between
    // the new atom and every atom that survives the (reverse) reptation move
    for (int i = 0; i < n_candidates; ++i)
    {
        PolymerConfiguration<double> config_reverse_reptated(config_reptated);
        config_reverse_reptated.reptateTowardsHead(r_reverse.row(i));
        Matrix<double, Dynamic, 3> coords_reverse_reptated = config_reverse_reptated.getSegment(0, length);

        // Check that, if i == 0, reversion was successful 
        if (i == 0)
        {
            for (int j = 0; j < length; ++j)
            {
                REQUIRE_THAT(
                    (coords_reverse_reptated.row(j) - coords[chain_idx].row(j)).norm(), 
                    Catch::Matchers::WithinAbs(0, tol)
                ); 
            }
        }  

        // Re-calculate this residual energy 
        //
        // If we are reptating towards the head (in reverse), this consists of:
        // - the non-bonded interaction energy between the new atom and atoms
        //   2, ..., 9 in the reverse-reptated configuration (the new atom is 
        //   atom 0)
        // - the non-bonded interaction energy between the new atom and every
        //   atom on every other chain 
        //
        // Start with the intra-chain interaction energy
        double residual = 0;  
        for (int j = 2; j < length; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                r_reverse.row(i), coords_reverse_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            residual += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }

        // Then calculate the inter-chain interaction energy 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_reverse.row(i), coords[j].row(k), xlen, ylen, zlen 
                    );
                    double r = dvec.norm(); 
                    residual += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
            }
        }
        REQUIRE_THAT(residuals_reverse(i), Catch::Matchers::WithinRel(residual, tol)); 

        // In other words, the intra-chain interaction energy is also the
        // non-bonded interaction energy between the new atom and atoms 1, ..., 8
        // in the once-reptated configuration 
        residual = 0;
        for (int j = 1; j < length - 1; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                r_reverse.row(i), coords_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            residual += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_reverse.row(i), coords[j].row(k), xlen, ylen, zlen 
                    );
                    double r = dvec.norm(); 
                    residual += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
            }
        }
        REQUIRE_THAT(residuals_reverse(i), Catch::Matchers::WithinRel(residual, tol));
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

    // Try generating 50 *forward* reptation moves at the head of chain 1
    chain_idx = 1;
    forward_result = sampler_cosine_dihedral.generateReptationMoves(
        chain_idx, ReptationDirection::HEAD, n_candidates
    );
    r_forward = forward_result.first; 
    residuals_forward = forward_result.second;

    // Choose a reptation move and generate 50 corresponding *backward* 
    // reptation moves 
    move_idx = 40;
    config_reptated = PolymerConfiguration<double>(
        coords[chain_idx], Units::NANO, 300.0
    ); 
    config_reptated.reptateTowardsHead(r_forward.row(move_idx));
    coords_reptated = config_reptated.getSegment(0, length); 
    reverse_result = sampler_cosine_dihedral.generateReptationMoves(
        chain_idx, ReptationDirection::TAIL, n_candidates, coords_reptated
    ); 
    r_reverse = reverse_result.first; 
    residuals_reverse = reverse_result.second;

    // Check that the 0-th atom is reversion to the original configuration
    //
    // Since we were originally reptating towards the head (reverse = tail), 
    // we are looking for the 9-th atom in the original configuration 
    tail_idx = length - 1; 
    REQUIRE_THAT(
        (r_reverse.row(0) - coords[chain_idx].row(tail_idx)).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );

    // Check that each new atom has a valid distance to the 8-th atom in the
    // original configuration
    for (int i = 0; i < n_candidates; ++i)
        REQUIRE((r_reverse.row(i) - coords[chain_idx].row(tail_idx - 1)).norm() < fene_params["R0"]);

    // Check the residual energy of each proposed move 
    //
    // This residual energy is the total non-bonded interaction energy between
    // the new atom and every atom that survives the (reverse) reptation move
    for (int i = 0; i < n_candidates; ++i)
    {
        PolymerConfiguration<double> config_reverse_reptated(config_reptated);
        config_reverse_reptated.reptateTowardsTail(r_reverse.row(i));
        Matrix<double, Dynamic, 3> coords_reverse_reptated = config_reverse_reptated.getSegment(0, length); 

        // Check that, if i == 0, reversion was successful
        if (i == 0)
        {
            for (int j = 0; j < length; ++j)
            {
                REQUIRE_THAT(
                    (coords_reverse_reptated.row(j) - coords[chain_idx].row(j)).norm(), 
                    Catch::Matchers::WithinAbs(0, tol)
                ); 
            }
        }

        // Re-calculate the residual energy 
        //
        // If we are reptating towards the tail (in reverse), this consists of:
        // - the non-bonded interaction energy between the new atom and atoms
        //   0, ..., 7 in the reverse-reptated configuration (the new atom is
        //   atom 9)
        // - the non-bonded interaction energy between the new atom and every
        //   atom on every other chain 
        //
        // Start with the intra-chain interaction energy 
        double residual = 0;  
        for (int j = 0; j < length - 2; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                r_reverse.row(i), coords_reverse_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            residual += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }

        // Then calculate the inter-chain interaction energy 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_reverse.row(i), coords[j].row(k), xlen, ylen, zlen 
                    );
                    double r = dvec.norm(); 
                    residual += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
            }
        }
        REQUIRE_THAT(residuals_reverse(i), Catch::Matchers::WithinRel(residual, tol)); 

        // In other words, the intra-chain interaction energy is also the
        // non-bonded interaction energy between the new atom and atoms 1, ..., 8
        // in the once-reptated configuration
        residual = 0; 
        for (int j = 1; j < length - 1; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                r_reverse.row(i), coords_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            residual += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_reverse.row(i), coords[j].row(k), xlen, ylen, zlen 
                    );
                    double r = dvec.norm(); 
                    residual += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
            }
        }
        REQUIRE_THAT(residuals_reverse(i), Catch::Matchers::WithinRel(residual, tol));
    } 
  
    // Try generating 50 *forward* reptation moves at the tail of chain 0
    chain_idx = 0;
    forward_result = sampler_cosine_dihedral.generateReptationMoves(
        chain_idx, ReptationDirection::TAIL, n_candidates
    );
    r_forward = forward_result.first; 
    residuals_forward = forward_result.second;

    // Choose a reptation move and generate 50 corresponding *backward* 
    // reptation moves 
    move_idx = 31;
    config_reptated = PolymerConfiguration<double>(
        coords[chain_idx], Units::NANO, 300.0
    ); 
    config_reptated.reptateTowardsTail(r_forward.row(move_idx)); 
    coords_reptated = config_reptated.getSegment(0, length); 
    reverse_result = sampler_cosine_dihedral.generateReptationMoves(
        chain_idx, ReptationDirection::HEAD, n_candidates, coords_reptated
    ); 
    r_reverse = reverse_result.first; 
    residuals_reverse = reverse_result.second; 

    // Check that the 0-th atom is reversion to the original configuration 
    // 
    // Since we were originally reptating towards the tail (reverse = head), 
    // we are looking for the 0-th atom in the original configuration 
    REQUIRE_THAT(
        (r_reverse.row(0) - coords[chain_idx].row(0)).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );  

    // Check that each new atom has a valid distance to the 1st atom in the
    // original configuration
    for (int i = 0; i < n_candidates; ++i)
        REQUIRE((r_reverse.row(i) - coords[chain_idx].row(1)).norm() < fene_params["R0"]); 

    // Check the residual energy of each proposed move 
    //
    // This residual energy is the total non-bonded interaction energy between
    // the new atom and every atom that survives the (reverse) reptation move
    for (int i = 0; i < n_candidates; ++i)
    {
        PolymerConfiguration<double> config_reverse_reptated(config_reptated);
        config_reverse_reptated.reptateTowardsHead(r_reverse.row(i));
        Matrix<double, Dynamic, 3> coords_reverse_reptated = config_reverse_reptated.getSegment(0, length);

        // Check that, if i == 0, reversion was successful 
        if (i == 0)
        {
            for (int j = 0; j < length; ++j)
            {
                REQUIRE_THAT(
                    (coords_reverse_reptated.row(j) - coords[chain_idx].row(j)).norm(), 
                    Catch::Matchers::WithinAbs(0, tol)
                ); 
            }
        }  

        // Re-calculate this residual energy 
        //
        // If we are reptating towards the head (in reverse), this consists of:
        // - the non-bonded interaction energy between the new atom and atoms
        //   2, ..., 9 in the reverse-reptated configuration (the new atom is 
        //   atom 0)
        // - the non-bonded interaction energy between the new atom and every
        //   atom on every other chain 
        //
        // Start with the intra-chain interaction energy
        double residual = 0;  
        for (int j = 2; j < length; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                r_reverse.row(i), coords_reverse_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            residual += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }

        // Then calculate the inter-chain interaction energy 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_reverse.row(i), coords[j].row(k), xlen, ylen, zlen 
                    );
                    double r = dvec.norm(); 
                    residual += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
            }
        }
        REQUIRE_THAT(residuals_reverse(i), Catch::Matchers::WithinRel(residual, tol)); 

        // In other words, the intra-chain interaction energy is also the
        // non-bonded interaction energy between the new atom and atoms 1, ..., 8
        // in the once-reptated configuration 
        residual = 0;
        for (int j = 1; j < length - 1; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                r_reverse.row(i), coords_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            residual += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_reverse.row(i), coords[j].row(k), xlen, ylen, zlen 
                    );
                    double r = dvec.norm(); 
                    residual += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
            }
        }
        REQUIRE_THAT(residuals_reverse(i), Catch::Matchers::WithinRel(residual, tol));
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

    // Try generating 50 *forward* reptation moves at the head of chain 1
    chain_idx = 1;
    forward_result = sampler_gaussian.generateReptationMoves(
        chain_idx, ReptationDirection::HEAD, n_candidates
    );
    r_forward = forward_result.first; 
    residuals_forward = forward_result.second;

    // Choose a reptation move and generate 50 corresponding *backward* 
    // reptation moves 
    move_idx = 40;
    config_reptated = PolymerConfiguration<double>(
        coords[chain_idx], Units::NANO, 300.0
    ); 
    config_reptated.reptateTowardsHead(r_forward.row(move_idx));
    coords_reptated = config_reptated.getSegment(0, length); 
    reverse_result = sampler_gaussian.generateReptationMoves(
        chain_idx, ReptationDirection::TAIL, n_candidates, coords_reptated
    ); 
    r_reverse = reverse_result.first; 
    residuals_reverse = reverse_result.second;

    // Check that the 0-th atom is reversion to the original configuration
    //
    // Since we were originally reptating towards the head (reverse = tail), 
    // we are looking for the 9-th atom in the original configuration 
    tail_idx = length - 1; 
    REQUIRE_THAT(
        (r_reverse.row(0) - coords[chain_idx].row(tail_idx)).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );

    // Check that each new atom has a valid distance to the 8-th atom in the
    // original configuration
    for (int i = 0; i < n_candidates; ++i)
        REQUIRE((r_reverse.row(i) - coords[chain_idx].row(tail_idx - 1)).norm() < fene_params["R0"]);

    // Check the residual energy of each proposed move 
    //
    // This residual energy is the total non-bonded interaction energy between
    // the new atom and every atom that survives the (reverse) reptation move
    for (int i = 0; i < n_candidates; ++i)
    {
        PolymerConfiguration<double> config_reverse_reptated(config_reptated);
        config_reverse_reptated.reptateTowardsTail(r_reverse.row(i));
        Matrix<double, Dynamic, 3> coords_reverse_reptated = config_reverse_reptated.getSegment(0, length); 

        // Check that, if i == 0, reversion was successful
        if (i == 0)
        {
            for (int j = 0; j < length; ++j)
            {
                REQUIRE_THAT(
                    (coords_reverse_reptated.row(j) - coords[chain_idx].row(j)).norm(), 
                    Catch::Matchers::WithinAbs(0, tol)
                ); 
            }
        }

        // Re-calculate the residual energy 
        //
        // If we are reptating towards the tail (in reverse), this consists of:
        // - the non-bonded interaction energy between the new atom and atoms
        //   0, ..., 7 in the reverse-reptated configuration (the new atom is
        //   atom 9)
        // - the non-bonded interaction energy between the new atom and every
        //   atom on every other chain 
        //
        // Start with the intra-chain interaction energy 
        double residual = 0;  
        for (int j = 0; j < length - 2; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                r_reverse.row(i), coords_reverse_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            residual += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }

        // Then calculate the inter-chain interaction energy 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_reverse.row(i), coords[j].row(k), xlen, ylen, zlen 
                    );
                    double r = dvec.norm(); 
                    residual += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
            }
        }
        REQUIRE_THAT(residuals_reverse(i), Catch::Matchers::WithinRel(residual, tol)); 

        // In other words, the intra-chain interaction energy is also the
        // non-bonded interaction energy between the new atom and atoms 1, ..., 8
        // in the once-reptated configuration
        residual = 0; 
        for (int j = 1; j < length - 1; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                r_reverse.row(i), coords_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            residual += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_reverse.row(i), coords[j].row(k), xlen, ylen, zlen 
                    );
                    double r = dvec.norm(); 
                    residual += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
            }
        }
        REQUIRE_THAT(residuals_reverse(i), Catch::Matchers::WithinRel(residual, tol));
    } 
  
    // Try generating 50 *forward* reptation moves at the tail of chain 0
    chain_idx = 0;
    forward_result = sampler_gaussian.generateReptationMoves(
        chain_idx, ReptationDirection::TAIL, n_candidates
    );
    r_forward = forward_result.first; 
    residuals_forward = forward_result.second;

    // Choose a reptation move and generate 50 corresponding *backward* 
    // reptation moves 
    move_idx = 31;
    config_reptated = PolymerConfiguration<double>(
        coords[chain_idx], Units::NANO, 300.0
    ); 
    config_reptated.reptateTowardsTail(r_forward.row(move_idx)); 
    coords_reptated = config_reptated.getSegment(0, length); 
    reverse_result = sampler_gaussian.generateReptationMoves(
        chain_idx, ReptationDirection::HEAD, n_candidates, coords_reptated
    ); 
    r_reverse = reverse_result.first; 
    residuals_reverse = reverse_result.second; 

    // Check that the 0-th atom is reversion to the original configuration 
    // 
    // Since we were originally reptating towards the tail (reverse = head), 
    // we are looking for the 0-th atom in the original configuration 
    REQUIRE_THAT(
        (r_reverse.row(0) - coords[chain_idx].row(0)).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );  

    // Check that each new atom has a valid distance to the 1st atom in the
    // original configuration
    for (int i = 0; i < n_candidates; ++i)
        REQUIRE((r_reverse.row(i) - coords[chain_idx].row(1)).norm() < fene_params["R0"]); 

    // Check the residual energy of each proposed move 
    //
    // This residual energy is the total non-bonded interaction energy between
    // the new atom and every atom that survives the (reverse) reptation move
    for (int i = 0; i < n_candidates; ++i)
    {
        PolymerConfiguration<double> config_reverse_reptated(config_reptated);
        config_reverse_reptated.reptateTowardsHead(r_reverse.row(i));
        Matrix<double, Dynamic, 3> coords_reverse_reptated = config_reverse_reptated.getSegment(0, length);

        // Check that, if i == 0, reversion was successful 
        if (i == 0)
        {
            for (int j = 0; j < length; ++j)
            {
                REQUIRE_THAT(
                    (coords_reverse_reptated.row(j) - coords[chain_idx].row(j)).norm(), 
                    Catch::Matchers::WithinAbs(0, tol)
                ); 
            }
        }  

        // Re-calculate this residual energy 
        //
        // If we are reptating towards the head (in reverse), this consists of:
        // - the non-bonded interaction energy between the new atom and atoms
        //   2, ..., 9 in the reverse-reptated configuration (the new atom is 
        //   atom 0)
        // - the non-bonded interaction energy between the new atom and every
        //   atom on every other chain 
        //
        // Start with the intra-chain interaction energy
        double residual = 0;  
        for (int j = 2; j < length; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                r_reverse.row(i), coords_reverse_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            residual += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }

        // Then calculate the inter-chain interaction energy 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_reverse.row(i), coords[j].row(k), xlen, ylen, zlen 
                    );
                    double r = dvec.norm(); 
                    residual += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
            }
        }
        REQUIRE_THAT(residuals_reverse(i), Catch::Matchers::WithinRel(residual, tol)); 

        // In other words, the intra-chain interaction energy is also the
        // non-bonded interaction energy between the new atom and atoms 1, ..., 8
        // in the once-reptated configuration 
        residual = 0;
        for (int j = 1; j < length - 1; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                r_reverse.row(i), coords_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            residual += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        r_reverse.row(i), coords[j].row(k), xlen, ylen, zlen 
                    );
                    double r = dvec.norm(); 
                    residual += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    ); 
                }
            }
        }
        REQUIRE_THAT(residuals_reverse(i), Catch::Matchers::WithinRel(residual, tol));
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

    // Try reptating by choosing from 10000 reptation candidate moves ... 
    //
    // First generate the moves 
    int chain_idx = 4; 
    const int n_candidates = 10000; 
    PolymerMeltConfiguration<double> melt_config_reptated(melt_config);  
    auto result = sampler_random.moveOnce(
        n_candidates, chain_idx, CBMCMoveType::REPTATION, 0
    );
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
        melt_config_reptated.reptateTowardsHead(chain_idx, forward_moves.row(move_idx)); 
    else 
        melt_config_reptated.reptateTowardsTail(chain_idx, forward_moves.row(move_idx));

    // Check that each new atom has a valid distance to the terminal atom 
    // at the appropriate end 
    for (int i = 0; i < n_candidates; ++i)
    {
        if (direction == ReptationDirection::HEAD)
        { 
            int forward_idx = 0;    // Forward move should involve new atom bonded to 0
            int reverse_idx = 8;    // Reverse move should involve new atom bonded to 8
            REQUIRE((forward_moves.row(i) - coords[chain_idx].row(forward_idx)).norm() < fene_params["R0"]);
            REQUIRE((reverse_moves.row(i) - coords[chain_idx].row(reverse_idx)).norm() < fene_params["R0"]); 
        }
        else 
        {
            int forward_idx = 9;    // Forward move should involve new atom bonded to 9
            int reverse_idx = 1;    // Reverse move should involve new atom bonded to 1
            REQUIRE((forward_moves.row(i) - coords[chain_idx].row(forward_idx)).norm() < fene_params["R0"]);
            REQUIRE((reverse_moves.row(i) - coords[chain_idx].row(reverse_idx)).norm() < fene_params["R0"]); 
        }
    }

    // Check that the 0-th reverse move is reversion to the original configuration
    if (direction == ReptationDirection::HEAD)
    {
        // The new atom should match the 9-th atom in the original configuration 
        REQUIRE_THAT(
            (reverse_moves.row(0) - coords[chain_idx].row(9)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    } 
    else 
    {
        // The new atom should match the 0-th atom in the original configuration
        REQUIRE_THAT(
            (reverse_moves.row(0) - coords[chain_idx].row(0)).norm(), 
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
        PolymerMeltConfiguration<double> melt_config2(melt_config);
        if (direction == ReptationDirection::HEAD)
            melt_config2.reptateTowardsHead(chain_idx, forward_moves.row(i)); 
        else 
            melt_config2.reptateTowardsTail(chain_idx, forward_moves.row(i));
        Matrix<double, Dynamic, 3> coords_reptated = melt_config2.getSegment(chain_idx, 0, length); 

        // Calculate the non-bonded energy difference between the two 
        // configurations  
        double diff = (
            melt_config2.getTotalNonbondedEnergy(lj_params, neighbor_threshold, true) - 
            melt_config.getTotalNonbondedEnergy(lj_params, neighbor_threshold, true)
        );

        // Calculate the non-bonded interaction energy between the new atom
        // and every atom that survives the reptation move
        //
        // If we are reptating towards the head, this is the interaction energy
        // between the new atom and atoms 2, ..., 9 in the reptated configuration
        //
        // If we are reptating towards the tail, this is the interaction energy
        // between the new atom and atoms 0, ..., 7 in the reptated configuration 
        double energy_new = 0; 
        int min_idx = (direction == ReptationDirection::HEAD ? 2 : 0); 
        int max_idx = (direction == ReptationDirection::HEAD ? 9 : 7); 
        for (int j = min_idx; j <= max_idx; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                forward_moves.row(i), coords_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            energy_new += lj<double>(r, lj_params["eps"], lj_params["sigma"], true);  
        }

        // Add in the interactions between the new atom and every atom along 
        // every other chain 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        forward_moves.row(i), coords[j].row(k), xlen, ylen, zlen
                    );
                    double r = dvec.norm(); 
                    energy_new += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    );  
                }
            }
        }

        // Calculate the non-bonded interaction energy between the atom that
        // dies from the reptation move and every atom that survives the 
        // reptation move
        //
        // If we are reptating towards the head:
        // - the old atom is atom 9, and 
        // - this is the interaction energy between the old atom and atoms
        //   0, ..., 7 in the original configuration
        //
        // If we are reptating towards the tail:
        // - the old atom is atom 0, and 
        // - this is the interaction energy between the old atom and atoms
        //   2, ..., 9 in the original configuration
        double energy_old = 0;
        Matrix<double, 3, 1> old_atom = (
            direction == ReptationDirection::HEAD ?
            coords[chain_idx].row(length - 1) : coords[chain_idx].row(0)
        ); 
        min_idx = (direction == ReptationDirection::HEAD ? 0 : 2); 
        max_idx = (direction == ReptationDirection::HEAD ? 7 : 9); 
        for (int j = min_idx; j <= max_idx; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                old_atom, coords[chain_idx].row(j), xlen, ylen, zlen
            ); 
            double r = dvec.norm(); 
            energy_old += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }

        // Add in the interactions between the old atom and every atom along 
        // every other chain 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        old_atom, coords[j].row(k), xlen, ylen, zlen
                    );
                    double r = dvec.norm(); 
                    energy_new += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    );  
                }
            }
        }

        // The energy difference should be exactly the difference between these
        // contributions 
        REQUIRE_THAT(energy_new - energy_old, Catch::Matchers::WithinRel(diff, tol)); 

        // Calculate the corresponding Boltzmann weight  
        weights_forward(i) = exp(-energy_new / kT);
    }

    // For each reverse move ... 
    for (int i = 0; i < n_candidates; ++i)
    {
        // Generate a new reptated configuration from the *reptated*
        // configuration, in the reverse direction 
        PolymerMeltConfiguration<double> melt_config2(melt_config_reptated); 
        if (direction == ReptationDirection::HEAD)
            melt_config2.reptateTowardsTail(chain_idx, reverse_moves.row(i)); 
        else 
            melt_config2.reptateTowardsHead(chain_idx, reverse_moves.row(i));
        Matrix<double, Dynamic, 3> coords_reptated = melt_config_reptated.getSegment(chain_idx, 0, length); 
        Matrix<double, Dynamic, 3> coords_reverse_reptated = melt_config2.getSegment(chain_idx, 0, length); 

        // Calculate the non-bonded energy difference between the two 
        // configurations  
        double diff = (
            melt_config2.getTotalNonbondedEnergy(lj_params, neighbor_threshold, true) - 
            melt_config_reptated.getTotalNonbondedEnergy(lj_params, neighbor_threshold, true)
        );

        // Calculate the non-bonded interaction energy between the new atom
        // and every atom that survives the reptation move
        //
        // If we are (reverse) reptating towards the head, this is the
        // interaction energy between the new atom and atoms 2, ..., 9 in the
        // twice-reptated configuration
        //
        // If we are (reverse) reptating towards the tail, this is the
        // interaction energy between the new atom and atoms 0, ..., 7 in the
        // twice-reptated configuration 
        double energy_new = 0; 
        int min_idx = (reverse_direction == ReptationDirection::HEAD ? 2 : 0); 
        int max_idx = (reverse_direction == ReptationDirection::HEAD ? 9 : 7); 
        for (int j = min_idx; j <= max_idx; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                reverse_moves.row(i), coords_reverse_reptated.row(j),
                xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            energy_new += lj<double>(r, lj_params["eps"], lj_params["sigma"], true);  
        }

        // Add in the interactions between the new atom and every atom along 
        // every other chain 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        reverse_moves.row(i), coords[j].row(k), xlen, ylen, zlen
                    );
                    double r = dvec.norm(); 
                    energy_new += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    );  
                }
            }
        }

        // Calculate the non-bonded interaction energy between the atom that
        // dies from the reptation move and every atom that survives the 
        // reptation move
        //
        // If we are (reverse) reptating towards the head:
        // - the old atom is atom 9 in the once-reptated configuration, and 
        // - this is the interaction energy between the old atom and atoms
        //   0, ..., 7 in the once-reptated configuration
        //
        // If we are (reverse) reptating towards the tail:
        // - the old atom is atom 0 in the once-reptated configuration, and 
        // - this is the interaction energy between the old atom and atoms
        //   2, ..., 9 in the once-reptated configuration
        double energy_old = 0;
        Matrix<double, 3, 1> old_atom = (
            reverse_direction == ReptationDirection::HEAD ?
            coords_reptated.row(length - 1) : coords_reptated.row(0)
        ); 
        min_idx = (reverse_direction == ReptationDirection::HEAD ? 0 : 2); 
        max_idx = (reverse_direction == ReptationDirection::HEAD ? 7 : 9); 
        for (int j = min_idx; j <= max_idx; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                old_atom, coords_reptated.row(j), xlen, ylen, zlen
            ); 
            double r = dvec.norm(); 
            energy_old += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }

        // Add in the interactions between the old atom and every atom along 
        // every other chain 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        old_atom, coords[j].row(k), xlen, ylen, zlen
                    );
                    double r = dvec.norm(); 
                    energy_new += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    );  
                }
            }
        }

        // The energy difference should be exactly the difference between these
        // contributions 
        REQUIRE_THAT(energy_new - energy_old, Catch::Matchers::WithinRel(diff, tol)); 

        // Calculate the corresponding Boltzmann weight 
        weights_reverse(i) = exp(-energy_new / kT);
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
    Matrix<double, Dynamic, 3> coords_reptated = melt_config_reptated.getSegment(chain_idx, 0, length); 
    Matrix<double, Dynamic, 3> coords_result = sampler_random.getCoords(chain_idx);  
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
                (coords_result.row(i) - coords[chain_idx].row(i)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 
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

    // Try reptating by choosing from 10000 reptation candidate moves ... 
    //
    // First generate the moves 
    chain_idx = 4; 
    melt_config_reptated = PolymerMeltConfiguration<double>(
        n_chains, coords, Units::NANO, 300.0, -xmax, xmax, -ymax, ymax, -zmax,
        zmax
    );
    result = sampler_cosine.moveOnce(
        n_candidates, chain_idx, CBMCMoveType::REPTATION, 0
    );
    forward_moves = std::get<0>(result); 
    reverse_moves = std::get<1>(result); 
    move_idx = std::get<2>(result); 
    prob_accept = std::get<3>(result); 
    accepted_move = std::get<4>(result);
    direction = static_cast<ReptationDirection>(
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
        melt_config_reptated.reptateTowardsHead(chain_idx, forward_moves.row(move_idx)); 
    else 
        melt_config_reptated.reptateTowardsTail(chain_idx, forward_moves.row(move_idx));

    // Check that each new atom has a valid distance to the terminal atom 
    // at the appropriate end 
    for (int i = 0; i < n_candidates; ++i)
    {
        if (direction == ReptationDirection::HEAD)
        { 
            int forward_idx = 0;    // Forward move should involve new atom bonded to 0
            int reverse_idx = 8;    // Reverse move should involve new atom bonded to 8
            REQUIRE((forward_moves.row(i) - coords[chain_idx].row(forward_idx)).norm() < fene_params["R0"]);
            REQUIRE((reverse_moves.row(i) - coords[chain_idx].row(reverse_idx)).norm() < fene_params["R0"]); 
        }
        else 
        {
            int forward_idx = 9;    // Forward move should involve new atom bonded to 9
            int reverse_idx = 1;    // Reverse move should involve new atom bonded to 1
            REQUIRE((forward_moves.row(i) - coords[chain_idx].row(forward_idx)).norm() < fene_params["R0"]);
            REQUIRE((reverse_moves.row(i) - coords[chain_idx].row(reverse_idx)).norm() < fene_params["R0"]); 
        }
    }

    // Check that the 0-th reverse move is reversion to the original configuration
    if (direction == ReptationDirection::HEAD)
    {
        // The new atom should match the 9-th atom in the original configuration 
        REQUIRE_THAT(
            (reverse_moves.row(0) - coords[chain_idx].row(9)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    } 
    else 
    {
        // The new atom should match the 0-th atom in the original configuration
        REQUIRE_THAT(
            (reverse_moves.row(0) - coords[chain_idx].row(0)).norm(), 
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
        PolymerMeltConfiguration<double> melt_config2(melt_config);
        if (direction == ReptationDirection::HEAD)
            melt_config2.reptateTowardsHead(chain_idx, forward_moves.row(i)); 
        else 
            melt_config2.reptateTowardsTail(chain_idx, forward_moves.row(i));
        Matrix<double, Dynamic, 3> coords_reptated = melt_config2.getSegment(chain_idx, 0, length); 

        // Calculate the non-bonded energy difference between the two 
        // configurations  
        double diff = (
            melt_config2.getTotalNonbondedEnergy(lj_params, neighbor_threshold, true) - 
            melt_config.getTotalNonbondedEnergy(lj_params, neighbor_threshold, true)
        );

        // Calculate the non-bonded interaction energy between the new atom
        // and every atom that survives the reptation move
        //
        // If we are reptating towards the head, this is the interaction energy
        // between the new atom and atoms 2, ..., 9 in the reptated configuration
        //
        // If we are reptating towards the tail, this is the interaction energy
        // between the new atom and atoms 0, ..., 7 in the reptated configuration 
        double energy_new = 0; 
        int min_idx = (direction == ReptationDirection::HEAD ? 2 : 0); 
        int max_idx = (direction == ReptationDirection::HEAD ? 9 : 7); 
        for (int j = min_idx; j <= max_idx; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                forward_moves.row(i), coords_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            energy_new += lj<double>(r, lj_params["eps"], lj_params["sigma"], true);  
        }

        // Add in the interactions between the new atom and every atom along 
        // every other chain 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        forward_moves.row(i), coords[j].row(k), xlen, ylen, zlen
                    );
                    double r = dvec.norm(); 
                    energy_new += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    );  
                }
            }
        }

        // Calculate the non-bonded interaction energy between the atom that
        // dies from the reptation move and every atom that survives the 
        // reptation move
        //
        // If we are reptating towards the head:
        // - the old atom is atom 9, and 
        // - this is the interaction energy between the old atom and atoms
        //   0, ..., 7 in the original configuration
        //
        // If we are reptating towards the tail:
        // - the old atom is atom 0, and 
        // - this is the interaction energy between the old atom and atoms
        //   2, ..., 9 in the original configuration
        double energy_old = 0;
        Matrix<double, 3, 1> old_atom = (
            direction == ReptationDirection::HEAD ?
            coords[chain_idx].row(length - 1) : coords[chain_idx].row(0)
        ); 
        min_idx = (direction == ReptationDirection::HEAD ? 0 : 2); 
        max_idx = (direction == ReptationDirection::HEAD ? 7 : 9); 
        for (int j = min_idx; j <= max_idx; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                old_atom, coords[chain_idx].row(j), xlen, ylen, zlen
            ); 
            double r = dvec.norm(); 
            energy_old += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }

        // Add in the interactions between the old atom and every atom along 
        // every other chain 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        old_atom, coords[j].row(k), xlen, ylen, zlen
                    );
                    double r = dvec.norm(); 
                    energy_new += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    );  
                }
            }
        }

        // The energy difference should be exactly the difference between these
        // contributions 
        REQUIRE_THAT(energy_new - energy_old, Catch::Matchers::WithinRel(diff, tol)); 

        // Calculate the corresponding Boltzmann weight  
        weights_forward(i) = exp(-energy_new / kT);
    }

    // For each reverse move ... 
    for (int i = 0; i < n_candidates; ++i)
    {
        // Generate a new reptated configuration from the *reptated*
        // configuration, in the reverse direction 
        PolymerMeltConfiguration<double> melt_config2(melt_config_reptated); 
        if (direction == ReptationDirection::HEAD)
            melt_config2.reptateTowardsTail(chain_idx, reverse_moves.row(i)); 
        else 
            melt_config2.reptateTowardsHead(chain_idx, reverse_moves.row(i));
        Matrix<double, Dynamic, 3> coords_reptated = melt_config_reptated.getSegment(chain_idx, 0, length); 
        Matrix<double, Dynamic, 3> coords_reverse_reptated = melt_config2.getSegment(chain_idx, 0, length); 

        // Calculate the non-bonded energy difference between the two 
        // configurations  
        double diff = (
            melt_config2.getTotalNonbondedEnergy(lj_params, neighbor_threshold, true) - 
            melt_config_reptated.getTotalNonbondedEnergy(lj_params, neighbor_threshold, true)
        );

        // Calculate the non-bonded interaction energy between the new atom
        // and every atom that survives the reptation move
        //
        // If we are (reverse) reptating towards the head, this is the
        // interaction energy between the new atom and atoms 2, ..., 9 in the
        // twice-reptated configuration
        //
        // If we are (reverse) reptating towards the tail, this is the
        // interaction energy between the new atom and atoms 0, ..., 7 in the
        // twice-reptated configuration 
        double energy_new = 0; 
        int min_idx = (reverse_direction == ReptationDirection::HEAD ? 2 : 0); 
        int max_idx = (reverse_direction == ReptationDirection::HEAD ? 9 : 7); 
        for (int j = min_idx; j <= max_idx; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                reverse_moves.row(i), coords_reverse_reptated.row(j),
                xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            energy_new += lj<double>(r, lj_params["eps"], lj_params["sigma"], true);  
        }

        // Add in the interactions between the new atom and every atom along 
        // every other chain 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        reverse_moves.row(i), coords[j].row(k), xlen, ylen, zlen
                    );
                    double r = dvec.norm(); 
                    energy_new += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    );  
                }
            }
        }

        // Calculate the non-bonded interaction energy between the atom that
        // dies from the reptation move and every atom that survives the 
        // reptation move
        //
        // If we are (reverse) reptating towards the head:
        // - the old atom is atom 9 in the once-reptated configuration, and 
        // - this is the interaction energy between the old atom and atoms
        //   0, ..., 7 in the once-reptated configuration
        //
        // If we are (reverse) reptating towards the tail:
        // - the old atom is atom 0 in the once-reptated configuration, and 
        // - this is the interaction energy between the old atom and atoms
        //   2, ..., 9 in the once-reptated configuration
        double energy_old = 0;
        Matrix<double, 3, 1> old_atom = (
            reverse_direction == ReptationDirection::HEAD ?
            coords_reptated.row(length - 1) : coords_reptated.row(0)
        ); 
        min_idx = (reverse_direction == ReptationDirection::HEAD ? 0 : 2); 
        max_idx = (reverse_direction == ReptationDirection::HEAD ? 7 : 9); 
        for (int j = min_idx; j <= max_idx; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                old_atom, coords_reptated.row(j), xlen, ylen, zlen
            ); 
            double r = dvec.norm(); 
            energy_old += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }

        // Add in the interactions between the old atom and every atom along 
        // every other chain 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        old_atom, coords[j].row(k), xlen, ylen, zlen
                    );
                    double r = dvec.norm(); 
                    energy_new += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    );  
                }
            }
        }

        // The energy difference should be exactly the difference between these
        // contributions 
        REQUIRE_THAT(energy_new - energy_old, Catch::Matchers::WithinRel(diff, tol)); 

        // Calculate the corresponding Boltzmann weight 
        weights_reverse(i) = exp(-energy_new / kT);
    }

    // Calculate the Rosenbluth factors
    forward_rosenbluth = weights_forward.sum();
    reverse_rosenbluth = weights_reverse.sum();

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
    coords_reptated = melt_config_reptated.getSegment(chain_idx, 0, length); 
    coords_result = sampler_cosine.getCoords(chain_idx);  
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
                (coords_result.row(i) - coords[chain_idx].row(i)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 
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

    // Try reptating by choosing from 10000 reptation candidate moves ... 
    //
    // First generate the moves 
    chain_idx = 4; 
    melt_config_reptated = PolymerMeltConfiguration<double>(
        n_chains, coords, Units::NANO, 300.0, -xmax, xmax, -ymax, ymax, -zmax,
        zmax
    );
    result = sampler_cosine_dihedral.moveOnce(
        n_candidates, chain_idx, CBMCMoveType::REPTATION, 0
    );
    forward_moves = std::get<0>(result); 
    reverse_moves = std::get<1>(result); 
    move_idx = std::get<2>(result); 
    prob_accept = std::get<3>(result); 
    accepted_move = std::get<4>(result);
    direction = static_cast<ReptationDirection>(
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
        melt_config_reptated.reptateTowardsHead(chain_idx, forward_moves.row(move_idx)); 
    else 
        melt_config_reptated.reptateTowardsTail(chain_idx, forward_moves.row(move_idx));

    // Check that each new atom has a valid distance to the terminal atom 
    // at the appropriate end 
    for (int i = 0; i < n_candidates; ++i)
    {
        if (direction == ReptationDirection::HEAD)
        { 
            int forward_idx = 0;    // Forward move should involve new atom bonded to 0
            int reverse_idx = 8;    // Reverse move should involve new atom bonded to 8
            REQUIRE((forward_moves.row(i) - coords[chain_idx].row(forward_idx)).norm() < fene_params["R0"]);
            REQUIRE((reverse_moves.row(i) - coords[chain_idx].row(reverse_idx)).norm() < fene_params["R0"]); 
        }
        else 
        {
            int forward_idx = 9;    // Forward move should involve new atom bonded to 9
            int reverse_idx = 1;    // Reverse move should involve new atom bonded to 1
            REQUIRE((forward_moves.row(i) - coords[chain_idx].row(forward_idx)).norm() < fene_params["R0"]);
            REQUIRE((reverse_moves.row(i) - coords[chain_idx].row(reverse_idx)).norm() < fene_params["R0"]); 
        }
    }

    // Check that the 0-th reverse move is reversion to the original configuration
    if (direction == ReptationDirection::HEAD)
    {
        // The new atom should match the 9-th atom in the original configuration 
        REQUIRE_THAT(
            (reverse_moves.row(0) - coords[chain_idx].row(9)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    } 
    else 
    {
        // The new atom should match the 0-th atom in the original configuration
        REQUIRE_THAT(
            (reverse_moves.row(0) - coords[chain_idx].row(0)).norm(), 
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
        PolymerMeltConfiguration<double> melt_config2(melt_config);
        if (direction == ReptationDirection::HEAD)
            melt_config2.reptateTowardsHead(chain_idx, forward_moves.row(i)); 
        else 
            melt_config2.reptateTowardsTail(chain_idx, forward_moves.row(i));
        Matrix<double, Dynamic, 3> coords_reptated = melt_config2.getSegment(chain_idx, 0, length); 

        // Calculate the non-bonded energy difference between the two 
        // configurations  
        double diff = (
            melt_config2.getTotalNonbondedEnergy(lj_params, neighbor_threshold, true) - 
            melt_config.getTotalNonbondedEnergy(lj_params, neighbor_threshold, true)
        );

        // Calculate the non-bonded interaction energy between the new atom
        // and every atom that survives the reptation move
        //
        // If we are reptating towards the head, this is the interaction energy
        // between the new atom and atoms 2, ..., 9 in the reptated configuration
        //
        // If we are reptating towards the tail, this is the interaction energy
        // between the new atom and atoms 0, ..., 7 in the reptated configuration 
        double energy_new = 0; 
        int min_idx = (direction == ReptationDirection::HEAD ? 2 : 0); 
        int max_idx = (direction == ReptationDirection::HEAD ? 9 : 7); 
        for (int j = min_idx; j <= max_idx; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                forward_moves.row(i), coords_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            energy_new += lj<double>(r, lj_params["eps"], lj_params["sigma"], true);  
        }

        // Add in the interactions between the new atom and every atom along 
        // every other chain 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        forward_moves.row(i), coords[j].row(k), xlen, ylen, zlen
                    );
                    double r = dvec.norm(); 
                    energy_new += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    );  
                }
            }
        }

        // Calculate the non-bonded interaction energy between the atom that
        // dies from the reptation move and every atom that survives the 
        // reptation move
        //
        // If we are reptating towards the head:
        // - the old atom is atom 9, and 
        // - this is the interaction energy between the old atom and atoms
        //   0, ..., 7 in the original configuration
        //
        // If we are reptating towards the tail:
        // - the old atom is atom 0, and 
        // - this is the interaction energy between the old atom and atoms
        //   2, ..., 9 in the original configuration
        double energy_old = 0;
        Matrix<double, 3, 1> old_atom = (
            direction == ReptationDirection::HEAD ?
            coords[chain_idx].row(length - 1) : coords[chain_idx].row(0)
        ); 
        min_idx = (direction == ReptationDirection::HEAD ? 0 : 2); 
        max_idx = (direction == ReptationDirection::HEAD ? 7 : 9); 
        for (int j = min_idx; j <= max_idx; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                old_atom, coords[chain_idx].row(j), xlen, ylen, zlen
            ); 
            double r = dvec.norm(); 
            energy_old += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }

        // Add in the interactions between the old atom and every atom along 
        // every other chain 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        old_atom, coords[j].row(k), xlen, ylen, zlen
                    );
                    double r = dvec.norm(); 
                    energy_new += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    );  
                }
            }
        }

        // The energy difference should be exactly the difference between these
        // contributions 
        REQUIRE_THAT(energy_new - energy_old, Catch::Matchers::WithinRel(diff, tol)); 

        // Calculate the corresponding Boltzmann weight  
        weights_forward(i) = exp(-energy_new / kT);
    }

    // For each reverse move ... 
    for (int i = 0; i < n_candidates; ++i)
    {
        // Generate a new reptated configuration from the *reptated*
        // configuration, in the reverse direction 
        PolymerMeltConfiguration<double> melt_config2(melt_config_reptated); 
        if (direction == ReptationDirection::HEAD)
            melt_config2.reptateTowardsTail(chain_idx, reverse_moves.row(i)); 
        else 
            melt_config2.reptateTowardsHead(chain_idx, reverse_moves.row(i));
        Matrix<double, Dynamic, 3> coords_reptated = melt_config_reptated.getSegment(chain_idx, 0, length); 
        Matrix<double, Dynamic, 3> coords_reverse_reptated = melt_config2.getSegment(chain_idx, 0, length); 

        // Calculate the non-bonded energy difference between the two 
        // configurations  
        double diff = (
            melt_config2.getTotalNonbondedEnergy(lj_params, neighbor_threshold, true) - 
            melt_config_reptated.getTotalNonbondedEnergy(lj_params, neighbor_threshold, true)
        );

        // Calculate the non-bonded interaction energy between the new atom
        // and every atom that survives the reptation move
        //
        // If we are (reverse) reptating towards the head, this is the
        // interaction energy between the new atom and atoms 2, ..., 9 in the
        // twice-reptated configuration
        //
        // If we are (reverse) reptating towards the tail, this is the
        // interaction energy between the new atom and atoms 0, ..., 7 in the
        // twice-reptated configuration 
        double energy_new = 0; 
        int min_idx = (reverse_direction == ReptationDirection::HEAD ? 2 : 0); 
        int max_idx = (reverse_direction == ReptationDirection::HEAD ? 9 : 7); 
        for (int j = min_idx; j <= max_idx; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                reverse_moves.row(i), coords_reverse_reptated.row(j),
                xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            energy_new += lj<double>(r, lj_params["eps"], lj_params["sigma"], true);  
        }

        // Add in the interactions between the new atom and every atom along 
        // every other chain 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        reverse_moves.row(i), coords[j].row(k), xlen, ylen, zlen
                    );
                    double r = dvec.norm(); 
                    energy_new += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    );  
                }
            }
        }

        // Calculate the non-bonded interaction energy between the atom that
        // dies from the reptation move and every atom that survives the 
        // reptation move
        //
        // If we are (reverse) reptating towards the head:
        // - the old atom is atom 9 in the once-reptated configuration, and 
        // - this is the interaction energy between the old atom and atoms
        //   0, ..., 7 in the once-reptated configuration
        //
        // If we are (reverse) reptating towards the tail:
        // - the old atom is atom 0 in the once-reptated configuration, and 
        // - this is the interaction energy between the old atom and atoms
        //   2, ..., 9 in the once-reptated configuration
        double energy_old = 0;
        Matrix<double, 3, 1> old_atom = (
            reverse_direction == ReptationDirection::HEAD ?
            coords_reptated.row(length - 1) : coords_reptated.row(0)
        ); 
        min_idx = (reverse_direction == ReptationDirection::HEAD ? 0 : 2); 
        max_idx = (reverse_direction == ReptationDirection::HEAD ? 7 : 9); 
        for (int j = min_idx; j <= max_idx; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                old_atom, coords_reptated.row(j), xlen, ylen, zlen
            ); 
            double r = dvec.norm(); 
            energy_old += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }

        // Add in the interactions between the old atom and every atom along 
        // every other chain 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        old_atom, coords[j].row(k), xlen, ylen, zlen
                    );
                    double r = dvec.norm(); 
                    energy_new += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    );  
                }
            }
        }

        // The energy difference should be exactly the difference between these
        // contributions 
        REQUIRE_THAT(energy_new - energy_old, Catch::Matchers::WithinRel(diff, tol)); 

        // Calculate the corresponding Boltzmann weight 
        weights_reverse(i) = exp(-energy_new / kT);
    }

    // Calculate the Rosenbluth factors
    forward_rosenbluth = weights_forward.sum();
    reverse_rosenbluth = weights_reverse.sum();

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
    coords_reptated = melt_config_reptated.getSegment(chain_idx, 0, length); 
    coords_result = sampler_cosine_dihedral.getCoords(chain_idx);  
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
                (coords_result.row(i) - coords[chain_idx].row(i)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 
    } 

    // ------------------------------------------------------------------ // 
    // Reptation moves on 10-mers with a Gaussian angle potential and a 
    // harmonic dihedral potential
    // ------------------------------------------------------------------ //
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

    // Try reptating by choosing from 10000 reptation candidate moves ... 
    //
    // First generate the moves 
    chain_idx = 4; 
    melt_config_reptated = PolymerMeltConfiguration<double>(
        n_chains, coords, Units::NANO, 300.0, -xmax, xmax, -ymax, ymax, -zmax,
        zmax
    );
    result = sampler_gaussian.moveOnce(
        n_candidates, chain_idx, CBMCMoveType::REPTATION, 0
    );
    forward_moves = std::get<0>(result); 
    reverse_moves = std::get<1>(result); 
    move_idx = std::get<2>(result); 
    prob_accept = std::get<3>(result); 
    accepted_move = std::get<4>(result);
    direction = static_cast<ReptationDirection>(
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
        melt_config_reptated.reptateTowardsHead(chain_idx, forward_moves.row(move_idx)); 
    else 
        melt_config_reptated.reptateTowardsTail(chain_idx, forward_moves.row(move_idx));

    // Check that each new atom has a valid distance to the terminal atom 
    // at the appropriate end 
    for (int i = 0; i < n_candidates; ++i)
    {
        if (direction == ReptationDirection::HEAD)
        { 
            int forward_idx = 0;    // Forward move should involve new atom bonded to 0
            int reverse_idx = 8;    // Reverse move should involve new atom bonded to 8
            REQUIRE((forward_moves.row(i) - coords[chain_idx].row(forward_idx)).norm() < fene_params["R0"]);
            REQUIRE((reverse_moves.row(i) - coords[chain_idx].row(reverse_idx)).norm() < fene_params["R0"]); 
        }
        else 
        {
            int forward_idx = 9;    // Forward move should involve new atom bonded to 9
            int reverse_idx = 1;    // Reverse move should involve new atom bonded to 1
            REQUIRE((forward_moves.row(i) - coords[chain_idx].row(forward_idx)).norm() < fene_params["R0"]);
            REQUIRE((reverse_moves.row(i) - coords[chain_idx].row(reverse_idx)).norm() < fene_params["R0"]); 
        }
    }

    // Check that the 0-th reverse move is reversion to the original configuration
    if (direction == ReptationDirection::HEAD)
    {
        // The new atom should match the 9-th atom in the original configuration 
        REQUIRE_THAT(
            (reverse_moves.row(0) - coords[chain_idx].row(9)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    } 
    else 
    {
        // The new atom should match the 0-th atom in the original configuration
        REQUIRE_THAT(
            (reverse_moves.row(0) - coords[chain_idx].row(0)).norm(), 
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
        PolymerMeltConfiguration<double> melt_config2(melt_config);
        if (direction == ReptationDirection::HEAD)
            melt_config2.reptateTowardsHead(chain_idx, forward_moves.row(i)); 
        else 
            melt_config2.reptateTowardsTail(chain_idx, forward_moves.row(i));
        Matrix<double, Dynamic, 3> coords_reptated = melt_config2.getSegment(chain_idx, 0, length); 

        // Calculate the non-bonded energy difference between the two 
        // configurations  
        double diff = (
            melt_config2.getTotalNonbondedEnergy(lj_params, neighbor_threshold, true) - 
            melt_config.getTotalNonbondedEnergy(lj_params, neighbor_threshold, true)
        );

        // Calculate the non-bonded interaction energy between the new atom
        // and every atom that survives the reptation move
        //
        // If we are reptating towards the head, this is the interaction energy
        // between the new atom and atoms 2, ..., 9 in the reptated configuration
        //
        // If we are reptating towards the tail, this is the interaction energy
        // between the new atom and atoms 0, ..., 7 in the reptated configuration 
        double energy_new = 0; 
        int min_idx = (direction == ReptationDirection::HEAD ? 2 : 0); 
        int max_idx = (direction == ReptationDirection::HEAD ? 9 : 7); 
        for (int j = min_idx; j <= max_idx; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                forward_moves.row(i), coords_reptated.row(j), xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            energy_new += lj<double>(r, lj_params["eps"], lj_params["sigma"], true);  
        }

        // Add in the interactions between the new atom and every atom along 
        // every other chain 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        forward_moves.row(i), coords[j].row(k), xlen, ylen, zlen
                    );
                    double r = dvec.norm(); 
                    energy_new += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    );  
                }
            }
        }

        // Calculate the non-bonded interaction energy between the atom that
        // dies from the reptation move and every atom that survives the 
        // reptation move
        //
        // If we are reptating towards the head:
        // - the old atom is atom 9, and 
        // - this is the interaction energy between the old atom and atoms
        //   0, ..., 7 in the original configuration
        //
        // If we are reptating towards the tail:
        // - the old atom is atom 0, and 
        // - this is the interaction energy between the old atom and atoms
        //   2, ..., 9 in the original configuration
        double energy_old = 0;
        Matrix<double, 3, 1> old_atom = (
            direction == ReptationDirection::HEAD ?
            coords[chain_idx].row(length - 1) : coords[chain_idx].row(0)
        ); 
        min_idx = (direction == ReptationDirection::HEAD ? 0 : 2); 
        max_idx = (direction == ReptationDirection::HEAD ? 7 : 9); 
        for (int j = min_idx; j <= max_idx; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                old_atom, coords[chain_idx].row(j), xlen, ylen, zlen
            ); 
            double r = dvec.norm(); 
            energy_old += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }

        // Add in the interactions between the old atom and every atom along 
        // every other chain 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        old_atom, coords[j].row(k), xlen, ylen, zlen
                    );
                    double r = dvec.norm(); 
                    energy_new += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    );  
                }
            }
        }

        // The energy difference should be exactly the difference between these
        // contributions 
        REQUIRE_THAT(energy_new - energy_old, Catch::Matchers::WithinRel(diff, tol)); 

        // Calculate the corresponding Boltzmann weight  
        weights_forward(i) = exp(-energy_new / kT);
    }

    // For each reverse move ... 
    for (int i = 0; i < n_candidates; ++i)
    {
        // Generate a new reptated configuration from the *reptated*
        // configuration, in the reverse direction 
        PolymerMeltConfiguration<double> melt_config2(melt_config_reptated); 
        if (direction == ReptationDirection::HEAD)
            melt_config2.reptateTowardsTail(chain_idx, reverse_moves.row(i)); 
        else 
            melt_config2.reptateTowardsHead(chain_idx, reverse_moves.row(i));
        Matrix<double, Dynamic, 3> coords_reptated = melt_config_reptated.getSegment(chain_idx, 0, length); 
        Matrix<double, Dynamic, 3> coords_reverse_reptated = melt_config2.getSegment(chain_idx, 0, length); 

        // Calculate the non-bonded energy difference between the two 
        // configurations  
        double diff = (
            melt_config2.getTotalNonbondedEnergy(lj_params, neighbor_threshold, true) - 
            melt_config_reptated.getTotalNonbondedEnergy(lj_params, neighbor_threshold, true)
        );

        // Calculate the non-bonded interaction energy between the new atom
        // and every atom that survives the reptation move
        //
        // If we are (reverse) reptating towards the head, this is the
        // interaction energy between the new atom and atoms 2, ..., 9 in the
        // twice-reptated configuration
        //
        // If we are (reverse) reptating towards the tail, this is the
        // interaction energy between the new atom and atoms 0, ..., 7 in the
        // twice-reptated configuration 
        double energy_new = 0; 
        int min_idx = (reverse_direction == ReptationDirection::HEAD ? 2 : 0); 
        int max_idx = (reverse_direction == ReptationDirection::HEAD ? 9 : 7); 
        for (int j = min_idx; j <= max_idx; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                reverse_moves.row(i), coords_reverse_reptated.row(j),
                xlen, ylen, zlen
            );
            double r = dvec.norm(); 
            energy_new += lj<double>(r, lj_params["eps"], lj_params["sigma"], true);  
        }

        // Add in the interactions between the new atom and every atom along 
        // every other chain 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        reverse_moves.row(i), coords[j].row(k), xlen, ylen, zlen
                    );
                    double r = dvec.norm(); 
                    energy_new += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    );  
                }
            }
        }

        // Calculate the non-bonded interaction energy between the atom that
        // dies from the reptation move and every atom that survives the 
        // reptation move
        //
        // If we are (reverse) reptating towards the head:
        // - the old atom is atom 9 in the once-reptated configuration, and 
        // - this is the interaction energy between the old atom and atoms
        //   0, ..., 7 in the once-reptated configuration
        //
        // If we are (reverse) reptating towards the tail:
        // - the old atom is atom 0 in the once-reptated configuration, and 
        // - this is the interaction energy between the old atom and atoms
        //   2, ..., 9 in the once-reptated configuration
        double energy_old = 0;
        Matrix<double, 3, 1> old_atom = (
            reverse_direction == ReptationDirection::HEAD ?
            coords_reptated.row(length - 1) : coords_reptated.row(0)
        ); 
        min_idx = (reverse_direction == ReptationDirection::HEAD ? 0 : 2); 
        max_idx = (reverse_direction == ReptationDirection::HEAD ? 7 : 9); 
        for (int j = min_idx; j <= max_idx; ++j)
        {
            Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                old_atom, coords_reptated.row(j), xlen, ylen, zlen
            ); 
            double r = dvec.norm(); 
            energy_old += lj<double>(r, lj_params["eps"], lj_params["sigma"], true); 
        }

        // Add in the interactions between the old atom and every atom along 
        // every other chain 
        for (int j = 0; j < n_chains; ++j)
        {
            if (j != chain_idx)
            {
                for (int k = 0; k < length; ++k)
                {
                    Matrix<double, 3, 1> dvec = periodicDistVec<double>(
                        old_atom, coords[j].row(k), xlen, ylen, zlen
                    );
                    double r = dvec.norm(); 
                    energy_new += lj<double>(
                        r, lj_params["eps"], lj_params["sigma"], true
                    );  
                }
            }
        }

        // The energy difference should be exactly the difference between these
        // contributions 
        REQUIRE_THAT(energy_new - energy_old, Catch::Matchers::WithinRel(diff, tol)); 

        // Calculate the corresponding Boltzmann weight 
        weights_reverse(i) = exp(-energy_new / kT);
    }

    // Calculate the Rosenbluth factors
    forward_rosenbluth = weights_forward.sum();
    reverse_rosenbluth = weights_reverse.sum();

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
    coords_reptated = melt_config_reptated.getSegment(chain_idx, 0, length); 
    coords_result = sampler_gaussian.getCoords(chain_idx);  
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
                (coords_result.row(i) - coords[chain_idx].row(i)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            ); 
        } 
    } 
}

