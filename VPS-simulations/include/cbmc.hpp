/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     2/1/2026
 */

#ifndef CONFIGURATIONAL_BIAS_MONTE_CARLO_HPP
#define CONFIGURATIONAL_BIAS_MONTE_CARLO_HPP

#include <cmath>
#include <string>
#include <utility>
#include <limits>
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/random.hpp>
#include "utils.hpp"

using std::min; 
using boost::multiprecision::min; 
using std::exp; 
using boost::multiprecision::exp;
using std::isnan; 
using boost::multiprecision::isnan; 
using std::isinf; 
using boost::multiprecision::isinf;

enum class TerminalSegmentEnd
{
    HEAD,
    TAIL
};

enum class CBMCMoveType
{
    REPTATION,
    TERMINAL_SEGMENT,
    INTERNAL_SEGMENT 
};

/** ----------------------------------------------------------------- // 
 *                              REPTATION                             // 
 *  ----------------------------------------------------------------- */
/**
 * Generate possible reptation moves from the given configuration.
 *
 * @param config Current polymer configuration.
 * @param direction Reptation direction. 
 * @param n_candidates Number of candidate moves to generate.  
 * @param rng Random number generator. 
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
 * @param neighbor_threshold Distance threshold for identifying neighboring
 *                           (non-bonded) atoms. 
 * @param fene_params FENE parameters.
 * @param angle_mode Angle potential type.  
 * @param angle_params Angle potential parameters. Must include the cosine
 *                     potential parameters (K and theta0) or the dual
 *                     Gaussian mixture potential parameters (A1, A2, w1, w2,
 *                     theta1, theta2). 
 * @param dihedral_params Dihedral angle potential parameters.
 * @returns Arrays of candidate atomic positions for the new atom and the
 *          corresponding reptation energy differences.  
 */
template <typename T>
std::pair<Matrix<T, Dynamic, 3>,
          Matrix<T, Dynamic, 1> > generateReptationMoves(PolymerConfiguration<T>& config,
                                                         const ReptationDirection direction, 
                                                         const int n_candidates,
                                                         boost::random::mt19937& rng,
                                                         boost::random::uniform_01<>& uniform_dist,
                                                         std::unordered_map<std::string, T>& lj_params,
                                                         const T neighbor_threshold, 
                                                         std::unordered_map<std::string, T>& fene_params,
                                                         const AngleMode angle_mode, 
                                                         std::unordered_map<std::string, T>& angle_params, 
                                                         std::unordered_map<std::string, T>& dihedral_params)
{
    const int n = config.getLength(); 
    Matrix<T, Dynamic, 3> coords = config.getSegment(0, n);
    Matrix<T, Dynamic, 3> moves(n_candidates, 3);
    Matrix<T, Dynamic, 1> energy_diffs(n_candidates); 

    // Define the angle sampling function  
    std::function<T(boost::random::mt19937&)> sample_angle;
    if (angle_mode == AngleMode::COSINE)
    {
        sample_angle = [&config, &angle_params, &uniform_dist](boost::random::mt19937& rng_) -> T
        {
            return sampleAngleCosine<T>(
                angle_params["K"], angle_params["theta0"], config.kT, rng_, 
                uniform_dist, 50
            );
        };
    } 
    else     // angle_mode == AngleMode::GAUSSIAN
    {
        sample_angle = [&config, &angle_params, &uniform_dist](boost::random::mt19937& rng_) -> T
        {
            return sampleAngleDualGaussianMixture<T>(
                angle_params["A1"], angle_params["A2"], angle_params["w1"],
                angle_params["w2"], angle_params["theta1"], angle_params["theta2"],
                config.kT, rng_, uniform_dist, 50
            );
        };
    }

    // Generate bond lengths, bond angles, and dihedral angles 
    Matrix<T, Dynamic, 1> lengths(n_candidates),
                          angles(n_candidates), 
                          dihedrals(n_candidates);
    for (int i = 0; i < n_candidates; ++i)
    {
        lengths(i) = sampleFene<T>(
            lj_params["eps"], lj_params["sigma"], fene_params["K"],
            fene_params["R0"], config.kT, rng, uniform_dist, 50 
        );
        angles(i) = sample_angle(rng);
        dihedrals(i) = sampleDihedralHarmonic<T>(
            dihedral_params["K"], config.kT, rng, uniform_dist
        );
    }
     
    if (direction == ReptationDirection::HEAD)    // Reptate towards the head 
    {
        // Generate new candidate atomic positions at the head
        for (int i = 0; i < n_candidates; ++i)
        {
            moves.row(i) = generateNextAtomDihedral<T>(
                coords.row(2), coords.row(1), coords.row(0), lengths(i, 0),
                angles(i, 0), dihedrals(i, 0), rng, uniform_dist,
                (dihedrals(i, 0) > 0 ? 1 : -1)
            );

            // Get the non-bonded energy difference due to reptation
            energy_diffs(i) = config.getReptationNonbondedEnergyDifference(
                ReptationDirection::HEAD, moves.row(i), lj_params, neighbor_threshold
            ); 
        }
    }
    else        // Reptate towards the tail 
    {
        // Generate new candidate atomic positions at the tail 
        for (int i = 0; i < n_candidates; ++i)
        {
            moves.row(i) = generateNextAtomDihedral<T>(
                coords.row(n - 3), coords.row(n - 2), coords.row(n - 1),
                lengths(i, 0), angles(i, 0), dihedrals(i, 0), rng, 
                uniform_dist, (dihedrals(i, 0) > 0 ? 1 : -1)
            );

            // Get the non-bonded energy difference due to reptation
            energy_diffs(i) = config.getReptationNonbondedEnergyDifference(
                ReptationDirection::TAIL, moves.row(i), lj_params, neighbor_threshold
            ); 
        }
    }

    return std::make_pair(moves, energy_diffs); 
}

/**
 * A configurational-bias Monte Carlo function for reptation of a linear
 * off-lattice polymer.  
 *
 * @param config Current polymer configuration.
 * @param n_candidates Number of candidate moves to generate.  
 * @param rng Random number generator. 
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
 * @param neighbor_threshold Distance threshold for identifying neighboring
 *                           (non-bonded) atoms. 
 * @param fene_params FENE parameters.
 * @param angle_mode Angle potential type.  
 * @param angle_params Angle potential parameters. Must include the cosine
 *                     potential parameters (K and theta0) or the dual
 *                     Gaussian mixture potential parameters (A1, A2, w1, w2,
 *                     theta1, theta2). 
 * @param dihedral_params Dihedral angle potential parameters.
 * @returns The chosen move and reptation direction, along with its Metropolis
 *          acceptance probability and whether the move was taken. 
 */
template <typename T>
std::tuple<ReptationDirection,
           Matrix<T, 3, 1>,
           T,
           bool> reptate(PolymerConfiguration<T>& config, const int n_candidates,
                         boost::random::mt19937& rng,
                         boost::random::uniform_01<>& uniform_dist,
                         std::unordered_map<std::string, T>& lj_params,
                         const T neighbor_threshold, 
                         std::unordered_map<std::string, T>& fene_params,
                         const AngleMode angle_mode, 
                         std::unordered_map<std::string, T>& angle_params, 
                         std::unordered_map<std::string, T>& dihedral_params)
{
    const int n = config.getLength(); 
    Matrix<T, Dynamic, 3> coords = config.getSegment(0, n);

    // Identify whether to reptate towards the head or the tail
    const ReptationDirection direction = (
        uniform_dist(rng) < 0.5 ? ReptationDirection::HEAD : ReptationDirection::TAIL
    );

    // Generate forward moves 
    auto forward_result = generateReptationMoves<T>(
        config, direction, n_candidates, rng, uniform_dist, lj_params,
        neighbor_threshold, fene_params, angle_mode, angle_params,
        dihedral_params
    );
    Matrix<T, Dynamic, 3> forward_moves = forward_result.first;
    Matrix<T, Dynamic, 1> forward_diffs = forward_result.second;  

    // Calculate the forward Rosenbluth factor
    Matrix<T, Dynamic, 1> forward_weights = ((-forward_diffs).array() / config.kT).exp().matrix(); 
    T forward_rosenbluth = forward_weights.sum(); 

    // Choose a candidate move 
    std::vector<T> probs; 
    for (int i = 0; i < n_candidates; ++i)
        probs.push_back(forward_weights(i) / forward_rosenbluth); 
    boost::random::discrete_distribution<> dist(probs);  
    int move_idx = dist(rng);
    Matrix<T, 3, 1> move = forward_moves.row(move_idx);

    // Generate a copy of the current polymer configuration and swap in the
    // chosen candidate move 
    PolymerConfiguration<T> config_chosen(config);
    if (direction == ReptationDirection::HEAD)
        config_chosen.reptateTowardsHead(move); 
    else 
        config_chosen.reptateTowardsTail(move); 
    Matrix<T, Dynamic, 3> coords_chosen = config_chosen.getSegment(0, n);

    // Generate reverse moves from the chosen configuration 
    const ReptationDirection reverse_direction = ( 
        direction == ReptationDirection::HEAD ? ReptationDirection::TAIL : ReptationDirection::HEAD
    ); 
    auto reverse_result = generateReptationMoves<T>(
        config_chosen, reverse_direction, n_candidates, rng, uniform_dist,
        lj_params, neighbor_threshold, fene_params, angle_mode, angle_params,
        dihedral_params
    );
    Matrix<T, Dynamic, 3> reverse_moves = reverse_result.first;
    Matrix<T, Dynamic, 1> reverse_diffs = reverse_result.second;

    // Add in the original configuration as one of the reverse moves
    if (reverse_direction == ReptationDirection::HEAD)
    { 
        reverse_moves.row(move_idx) = coords.row(0); 
        reverse_diffs(move_idx) = config_chosen.getReptationNonbondedEnergyDifference(
            ReptationDirection::HEAD, coords.row(0), lj_params, neighbor_threshold
        );
    }
    else 
    {
        reverse_moves.row(move_idx) = coords.row(n - 1); 
        reverse_diffs(move_idx) = config_chosen.getReptationNonbondedEnergyDifference(
            ReptationDirection::TAIL, coords.row(n - 1), lj_params, neighbor_threshold
        ); 
    } 

    // Calculate the reverse Rosenbluth factor
    Matrix<T, Dynamic, 1> reverse_weights = ((-reverse_diffs).array() / config_chosen.kT).exp().matrix(); 
    T reverse_rosenbluth = reverse_weights.sum(); 

    // Calculate the Metropolis acceptance probability
    T prob_accept = min(1.0, forward_rosenbluth / reverse_rosenbluth);

    // Change the polymer configuration according to that probability
    T r = uniform_dist(rng); 
    if (r < prob_accept)
    {
        if (direction == ReptationDirection::HEAD)    // Reptate towards the head
            config.reptateTowardsHead(move);
        else                                          // Reptate towards the tail 
            config.reptateTowardsTail(move);
    }

    // Return the reptation direction, new atom, acceptance probability, and
    // whether the move was taken 
    return std::make_tuple(direction, move, prob_accept, (r < prob_accept)); 
}

/** ----------------------------------------------------------------- // 
 *                     MOVES OF TERMINAL SEGMENTS                     // 
 *  ----------------------------------------------------------------- */
/**
 * Generate possible terminal segment moves from the given configuration.
 *
 * @param config Current polymer configuration.
 * @param direction Choice of terminal segment to move. 
 * @param n_candidates Number of candidate moves to generate.  
 * @param rng Random number generator. 
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
 * @param neighbor_threshold Distance threshold for identifying neighboring
 *                           (non-bonded) atoms. 
 * @param fene_params FENE parameters. 
 * @param angle_mode Angle potential type.  
 * @param angle_params Angle potential parameters. Must include the cosine
 *                     potential parameters (K and theta0) or the dual
 *                     Gaussian mixture potential parameters (A1, A2, w1, w2,
 *                     theta1, theta2). 
 * @param dihedral_params Dihedral angle potential parameters.
 * @returns Arrays of candidate atomic positions for the new segment and the
 *          corresponding reptation energy differences.  
 */
template <typename T, size_t SegmentLength>
std::pair<Matrix<T, Dynamic, 3 * SegmentLength>, 
          Matrix<T, Dynamic, 1> > generateTerminalSegmentMoves(PolymerConfiguration<T>& config,
                                                               const TerminalSegmentEnd direction,
                                                               const int n_candidates,
                                                               boost::random::mt19937& rng,
                                                               boost::random::uniform_01<>& uniform_dist,
                                                               std::unordered_map<std::string, T>& lj_params, 
                                                               const T neighbor_threshold, 
                                                               std::unordered_map<std::string, T>& fene_params,
                                                               const AngleMode angle_mode,  
                                                               std::unordered_map<std::string, T>& angle_params,
                                                               std::unordered_map<std::string, T>& dihedral_params)
{
    const int n = config.getLength(); 
    Matrix<T, Dynamic, 3> coords = config.getSegment(0, n);
    Matrix<T, Dynamic, 3 * SegmentLength> moves(n_candidates, 3 * SegmentLength);
    Matrix<T, Dynamic, 1> energy_diffs(n_candidates); 

    // Define the angle sampling function  
    std::function<T(boost::random::mt19937&)> sample_angle;
    if (angle_mode == AngleMode::COSINE)
    {
        sample_angle = [&config, &angle_params, &uniform_dist](boost::random::mt19937& rng_) -> T
        {
            return sampleAngleCosine<T>(
                angle_params["K"], angle_params["theta0"], config.kT, rng_, 
                uniform_dist, 50
            );
        };
    } 
    else     // angle_mode == AngleMode::GAUSSIAN
    {
        sample_angle = [&config, &angle_params, &uniform_dist](boost::random::mt19937& rng_) -> T
        {
            return sampleAngleDualGaussianMixture<T>(
                angle_params["A1"], angle_params["A2"], angle_params["w1"],
                angle_params["w2"], angle_params["theta1"], angle_params["theta2"],
                config.kT, rng_, uniform_dist, 50
            );
        };
    }

    // Generate bond lengths, bond angles, and dihedral angles 
    Matrix<T, Dynamic, Dynamic> lengths(n_candidates, SegmentLength),
                                angles(n_candidates, SegmentLength),
                                dihedrals(n_candidates, SegmentLength);
    for (int i = 0; i < n_candidates; ++i)
    {
        for (int j = 0; j < SegmentLength; ++j)
        {
            lengths(i, j) = sampleFene<T>(
                lj_params["eps"], lj_params["sigma"], fene_params["K"],
                fene_params["R0"], config.kT, rng, uniform_dist, 50 
            );
            angles(i, j) = sample_angle(rng);
            dihedrals(i, j) = sampleDihedralHarmonic<T>(
                dihedral_params["K"], config.kT, rng, uniform_dist
            );
        }
    }
     
    if (direction == TerminalSegmentEnd::HEAD)    // Move the terminal segment at the head 
    {
        // Generate new candidate atomic positions for the head segment
        for (int i = 0; i < n_candidates; ++i)
        {
            Matrix<T, Dynamic, 3> segment_i(SegmentLength, 3); 

            // Move backwards from atom (SegmentLength) in the polymer  
            for (int j = 0; j < SegmentLength; ++j)
            {
                Matrix<T, 3, 1> r1, r2, r3; 
                if (j == 0)         // Last atom in the segment (closest to the polymer)
                {
                    r1 = coords.row(SegmentLength + 2);
                    r2 = coords.row(SegmentLength + 1); 
                    r3 = coords.row(SegmentLength);  
                }
                else if (j == 1)    // Second-to-last
                {
                    r1 = coords.row(SegmentLength + 1); 
                    r2 = coords.row(SegmentLength); 
                    r3 = segment_i.row(SegmentLength - 1);  
                }
                else if (j == 2)    // Third-to-last
                {
                    r1 = coords.row(SegmentLength); 
                    r2 = segment_i.row(SegmentLength - 2);
                    r3 = segment_i.row(SegmentLength - 1); 
                }
                else 
                {
                    r1 = segment_i.row(SegmentLength - 3); 
                    r2 = segment_i.row(SegmentLength - 2); 
                    r3 = segment_i.row(SegmentLength - 1);  
                }
                int idx = SegmentLength - 1 - j;
                segment_i.row(idx) = generateNextAtomDihedral<T>(
                    r1, r2, r3, lengths(i, j), angles(i, j), dihedrals(i, j),
                    rng, uniform_dist, (dihedrals(i, j) > 0 ? 1 : -1)
                );
                moves(i, Eigen::seqN(3 * idx, 3)) = segment_i.row(idx); 
            }

            // Get the non-bonded energy difference due to segment replacement
            energy_diffs(i) = config.getSegmentReplacementNonbondedEnergyDifference(
                segment_i, 0, lj_params, neighbor_threshold
            ); 
        }
    }
    else        // Move the terminal segment at the tail 
    {
        // Generate new candidate atomic positions for the tail segment
        for (int i = 0; i < n_candidates; ++i)
        {
            Matrix<T, Dynamic, 3> segment_i(SegmentLength, 3); 

            // Move forward from atom (n - SegmentLength) in the polymer 
            for (int j = 0; j < SegmentLength; ++j)
            {
                Matrix<T, 3, 1> r1, r2, r3; 
                if (j == 0)
                {
                    r1 = coords.row(n - SegmentLength - 3);
                    r2 = coords.row(n - SegmentLength - 2); 
                    r3 = coords.row(n - SegmentLength - 1);  
                }
                else if (j == 1)
                {
                    r1 = coords.row(n - SegmentLength - 2); 
                    r2 = coords.row(n - SegmentLength - 1); 
                    r3 = segment_i.row(0); 
                }
                else if (j == 2)
                {
                    r1 = coords.row(n - SegmentLength - 1); 
                    r2 = segment_i.row(0); 
                    r3 = segment_i.row(1);
                }
                else 
                {
                    r1 = segment_i.row(0);
                    r2 = segment_i.row(1); 
                    r3 = segment_i.row(2); 
                }
                segment_i.row(j) = generateNextAtomDihedral<T>(
                    r1, r2, r3, lengths(i, j), angles(i, j), dihedrals(i, j),
                    rng, uniform_dist, (dihedrals(i, j) > 0 ? 1 : -1)
                );
                moves(i, Eigen::seqN(3 * j, 3)) = segment_i.row(j); 
            }

            // Get the non-bonded energy difference due to segment replacement
            energy_diffs(i) = config.getSegmentReplacementNonbondedEnergyDifference(
                segment_i, n - SegmentLength, lj_params, neighbor_threshold
            ); 
        }
    }

    return std::make_pair(moves, energy_diffs); 
}

/**
 * A configurational-bias Monte Carlo function for movement of terminal 
 * segments in a linear off-lattice polymer.  
 *
 * @param config Current polymer configuration.
 * @param n_tries Number of candidate moves to generate.  
 * @param rng Random number generator. 
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
 * @param neighbor_threshold Distance threshold for identifying neighboring
 *                           (non-bonded) atoms. 
 * @param fene_params FENE parameters. 
 * @param angle_mode Angle potential type.  
 * @param angle_params Angle potential parameters. Must include the cosine
 *                     potential parameters (K and theta0) or the dual
 *                     Gaussian mixture potential parameters (A1, A2, w1, w2,
 *                     theta1, theta2). 
 * @param dihedral_params Dihedral angle potential parameters.
 * @returns The chosen move and terminal segment end, along with its Metropolis
 *          acceptance probability and whether the move was taken. 
 */
template <typename T, size_t SegmentLength>
std::tuple<TerminalSegmentEnd,
           Matrix<T, Dynamic, 3>, 
           T,
           bool> moveTerminalSegment(PolymerConfiguration<T>& config,
                                     const int n_candidates,
                                     boost::random::mt19937& rng,
                                     boost::random::uniform_01<>& uniform_dist,
                                     std::unordered_map<std::string, T>& lj_params, 
                                     const T neighbor_threshold, 
                                     std::unordered_map<std::string, T>& fene_params,
                                     const AngleMode angle_mode,  
                                     std::unordered_map<std::string, T>& angle_params,
                                     std::unordered_map<std::string, T>& dihedral_params)
{
    const int n = config.getLength(); 
    Matrix<T, Dynamic, 3> coords = config.getSegment(0, n);

    // Identify whether to move the terminal segment at the head or the tail 
    const TerminalSegmentEnd direction = (
        uniform_dist(rng) < 0.5 ? TerminalSegmentEnd::HEAD : TerminalSegmentEnd::TAIL
    );

    // Generate forward moves
    auto forward_result = generateTerminalSegmentMoves<T, SegmentLength>(
        config, direction, n_candidates, rng, uniform_dist, lj_params,
        neighbor_threshold, fene_params, angle_mode, angle_params,
        dihedral_params
    );
    Matrix<T, Dynamic, 3 * SegmentLength> forward_moves = forward_result.first;
    Matrix<T, Dynamic, 1> forward_diffs = forward_result.second;  

    // Calculate the forward Rosenbluth factor
    Matrix<T, Dynamic, 1> forward_weights = ((-forward_diffs).array() / config.kT).exp().matrix(); 
    T forward_rosenbluth = forward_weights.sum(); 

    // Choose a candidate move 
    std::vector<T> probs; 
    for (int i = 0; i < n_candidates; ++i)
        probs.push_back(forward_weights(i) / forward_rosenbluth); 
    boost::random::discrete_distribution<> dist(probs);  
    int move_idx = dist(rng);
    Matrix<T, Dynamic, 3> move(SegmentLength, 3); 
    for (int i = 0; i < SegmentLength; ++i)
    {
        move(i, 0) = forward_moves(move_idx, 3 * i); 
        move(i, 1) = forward_moves(move_idx, 3 * i + 1); 
        move(i, 2) = forward_moves(move_idx, 3 * i + 2); 
    }

    // Generate a copy of the current polymer configuration and swap in the
    // chosen candidate move 
    PolymerConfiguration<T> config_chosen(config);
    if (direction == TerminalSegmentEnd::HEAD)
        config_chosen.replaceSegment(move, 0); 
    else 
        config_chosen.replaceSegment(move, n - SegmentLength);
    Matrix<T, Dynamic, 3> coords_chosen = config_chosen.getSegment(0, n);

    // Generate reverse moves from the chosen configuration 
    const TerminalSegmentEnd reverse_direction = ( 
        direction == TerminalSegmentEnd::HEAD ? TerminalSegmentEnd::TAIL : TerminalSegmentEnd::HEAD
    );
    auto reverse_result = generateTerminalSegmentMoves<T, SegmentLength>(
        config_chosen, reverse_direction, n_candidates, rng, uniform_dist,
        lj_params, neighbor_threshold, fene_params, angle_mode, angle_params,
        dihedral_params
    );
    Matrix<T, Dynamic, 3 * SegmentLength> reverse_moves = reverse_result.first;
    Matrix<T, Dynamic, 1> reverse_diffs = reverse_result.second;

    // Add in the original configuration as one of the reverse moves
    if (reverse_direction == TerminalSegmentEnd::HEAD)
    {
        Matrix<T, Dynamic, 3> segment = coords(Eigen::seqN(0, SegmentLength), Eigen::all); 
        for (int i = 0; i < SegmentLength; ++i)
        {
            reverse_moves(move_idx, 3 * i) = segment(i, 0); 
            reverse_moves(move_idx, 3 * i + 1) = segment(i, 1); 
            reverse_moves(move_idx, 3 * i + 2) = segment(i, 2); 
        }
        reverse_diffs(move_idx) = config_chosen.getSegmentReplacementNonbondedEnergyDifference(
            segment, 0, lj_params, neighbor_threshold 
        );
    }
    else 
    {
        Matrix<T, Dynamic, 3> segment = coords(
            Eigen::seqN(n - SegmentLength, SegmentLength), Eigen::all
        );
        for (int i = 0; i < SegmentLength; ++i)
        {
            reverse_moves(move_idx, 3 * i) = segment(i, 0); 
            reverse_moves(move_idx, 3 * i + 1) = segment(i, 1); 
            reverse_moves(move_idx, 3 * i + 2) = segment(i, 2); 
        }
        reverse_diffs(move_idx) = config_chosen.getSegmentReplacementNonbondedEnergyDifference(
            segment, n - SegmentLength, lj_params, neighbor_threshold
        ); 
    } 

    // Calculate the reverse Rosenbluth factor
    Matrix<T, Dynamic, 1> reverse_weights = ((-reverse_diffs).array() / config_chosen.kT).exp().matrix(); 
    T reverse_rosenbluth = reverse_weights.sum(); 

    // Calculate the Metropolis acceptance probability
    T prob_accept = min(1.0, forward_rosenbluth / reverse_rosenbluth);

    // Change the polymer configuration according to that probability
    T r = uniform_dist(rng); 
    if (r < prob_accept)
    {
        if (direction == TerminalSegmentEnd::HEAD)    // Move the terminal segment at the head 
            config.replaceSegment(move, 0); 
        else                                          // Move the terminal segment at the tail 
            config.replaceSegment(move, n - SegmentLength); 
    }

    // Return the terminal segment end, new segment, acceptance probability,
    // and whether the move was taken 
    return std::make_tuple(direction, move, prob_accept, (r < prob_accept)); 
}

/** ------------------------------------------------------------------- // 
 *  CONCERTED MOVES OF INTERNAL SEGMENTS (ZAMUNER ET AL. PLOS ONE 2015) //
 *  ------------------------------------------------------------------- */ 
template <typename T, size_t DimIn, size_t DimOut>
using VectorValuedFunction = std::function<Matrix<T, DimOut, 1>(const Ref<const Matrix<T, DimIn, 1> >&)>;

/**
 * Get the Jacobian of a vector-value function at the given point.
 *
 * Each partial derivative is approximated through a finite difference 
 * approximation.
 *
 * @param F Input function. 
 * @param x0 Input point. 
 * @param dx Increment for finite difference approximation. 
 * @returns Jacobian matrix of F at x0.  
 */
template <typename T, size_t DimIn, size_t DimOut>
Matrix<T, DimOut, DimIn> getJacobian(VectorValuedFunction<T, DimIn, DimOut>& F, 
                                     const Ref<const Matrix<T, DimIn, 1> >& x0,
                                     const T dx = 1e-8)
{
    Matrix<T, DimOut, DimIn> J;
    for (int j = 0; j < DimIn; ++j)
    {
        Matrix<T, DimIn, 1> xp(x0), xm(x0); 
        xp(j) += dx; 
        xm(j) -= dx;
        J.col(j) = (F(xp) - F(xm)) / (2 * dx);
    }

    return J; 
}

/**
 * Get orthonormal bases for the tangent space and its orthogonal complement
 * of a manifold, defined by F(x) = 0, at a point, x0.
 *
 * The function maps a vector of dimension DimOut to a vector of dimension
 * DimIn. Therefore, the Jacobian has size (DimOut x DimIn), and the SVD 
 * yields a right singular matrix V of size (DimIn x DimIn). The columns 
 * corresponding to the zero singular value comprise a basis for the tangent
 * space; the remaining columns comprise a basis for the orthogonal complement.
 *
 * @param F Input function; the manifold is defined as F(x) = 0.  
 * @param x0 Input point. 
 * @param dx Increment for finite difference approximation. 
 * @returns Two matrices, the first with columns spanning the tangent space 
 *          of the Jacobian at x0, and the second with columns spanning the 
 *          orthogonal complement. 
 */
template <typename T, size_t DimIn, size_t DimOut>
std::pair<Matrix<T, DimIn, DimIn - DimOut>, Matrix<T, DimIn, DimOut> >
    getTangentAndOrthogonalSpaceBases(VectorValuedFunction<T, DimIn, DimOut>& F, 
                                      const Ref<const Matrix<T, DimIn, 1> >& x0, 
                                      const T dx = 1e-8)
{
    // Get the Jacobian of F at x0
    Matrix<T, DimOut, DimIn> J = getJacobian<T, DimIn, DimOut>(F, x0, dx);

    // Compute the SVD of the Jacobian 
    auto svd = J.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

    // Assume that there are (DimIn - DimOut) zero singular values, corresponding
    // to the dimension of the tangent space
    //
    // Therefore, the last (DimIn - DimOut) right singular vectors comprise a
    // basis for the tangent space
    Matrix<T, DimIn, DimIn> V = svd.matrixV(); 
    Matrix<T, DimIn, DimOut> orthogonal_basis = V(Eigen::all, Eigen::seqN(0, DimOut));
    Matrix<T, DimIn, DimIn - DimOut> tangent_basis = V(Eigen::all, Eigen::seqN(DimOut, DimIn - DimOut));  

    return std::make_pair(tangent_basis, orthogonal_basis);  
}

/**
 * Generate a small perturbation of the input point, x0, that satisfies the 
 * constraints F(x) = 0. 
 *
 * The constraint function, F, maps a vector of dimension DimOut to a vector
 * of dimension DimIn. The input point, x0, and output point have dimension 
 * DimIn. This algorithm perturbs x0 along the tangent space of the Jacobian 
 * of F, then projects the perturbation back onto the manifold where F(x) = 0
 * using a basic Newton's method with Armijo backtracking.  
 *
 * @param F Input function; the manifold is defined as F(x) = 0.  
 * @param x0 Input point.
 * @param tangent_stepsize Perturbation stepsize along the tangent space. 
 * @param dir Perturbation direction.  
 * @param dx Increment for finite difference approximation.
 * @param newton_tol Tolerance for assessing convergence of Newton's method. 
 * @param min_newton_stepsize Minimum stepsize for Newton's method. 
 * @param armijo_const Constant for Armijo condition. Set to 1e-4 by default,
 *                     following Nocedal and Wright (page 33).   
 * @returns Two matrices, the first with columns spanning the tangent space 
 *          of the Jacobian at x0, and the second with columns spanning the 
 *          orthogonal complement. 
 */
template <typename T, size_t DimIn, size_t DimOut>
Matrix<T, DimIn, 1> generateConcertedMove(VectorValuedFunction<T, DimIn, DimOut>& F, 
                                          const Ref<const Matrix<T, DimIn, 1> >& x0, 
                                          const T tangent_stepsize,
                                          const Ref<const Matrix<T, DimIn - DimOut, 1> >& dir, 
                                          const T dx = 1e-8,
                                          const T newton_tol = 1e-8,
                                          const T min_newton_stepsize = 1e-4,
                                          const T armijo_const = 1e-4)
{
    // Get bases for the tangent space and orthogonal complement of the 
    // constraint manifold at x0
    auto result = getTangentAndOrthogonalSpaceBases<T, DimIn, DimOut>(F, x0, dx);
    Matrix<T, DimIn, DimIn - DimOut> Qt = result.first; 
    Matrix<T, DimIn, DimOut> Qp = result.second; 

    // Take a step within the tangent space
    Matrix<T, DimIn, 1> x1 = x0 + tangent_stepsize * Qt * dir;

    // Project the resulting vector onto the orthogonal complement 
    auto func = [&x1, &Qp](const Ref<const Matrix<T, DimOut, 1> >& z) -> Matrix<T, DimIn, 1>
    {
        return x1 + Qp * z; 
    };
    Matrix<T, DimOut, 1> z_curr = Matrix<T, DimOut, 1>::Zero(); 
    Matrix<T, DimIn, 1> z_proj = func(z_curr);
    Matrix<T, DimOut, 1> f_curr = F(z_proj);  
    T residual = f_curr.norm(); 
    while (residual > newton_tol)
    {
        // Compute the Jacobian corresponding to the current step, times 
        // the orthogonal complement matrix  
        Matrix<T, DimOut, DimOut> JQ_curr = getJacobian<T, DimIn, DimOut>(F, z_proj, dx) * Qp;

        // Solve for the corresponding Newton step
        Matrix<T, DimOut, 1> dz = JQ_curr.fullPivLu().solve(-f_curr);

        // Update z by some multiple of dz (determine the multiplier by 
        // Armijo backtracking) 
        T stepsize = 1;
        while (stepsize > min_newton_stepsize)
        {
            Matrix<T, DimOut, 1> z_next = z_curr + stepsize * dz; 
            T residual_next = F(func(z_next)).norm();
            if (residual_next <= (1 - armijo_const * stepsize) * residual)
                break; 
            stepsize *= 0.5; 
        }
        z_curr += stepsize * dz;
        z_proj = func(z_curr);  
        f_curr = F(z_proj); 
        residual = f_curr.norm(); 
    }

    return x1 + Qp * z_curr; 
}

/**
 * Chooses one move, among the given candidate concerted internal segment
 * moves, according to their Rosenbluth weights, and calculate its Metropolis
 * acceptance probability.  
 *
 * @param config Current polymer configuration. 
 * @param moves Array of candidate moves. Each row is a new proposed vector of
 *              segment coordinates.
 * @param idx Index of first atom to be replaced.
 * @param rng Random number generator. 
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
 * @param neighbor_threshold Distance threshold for identifying neighboring
 *                           (non-bonded) atoms. 
 * @param fene_params FENE parameters. 
 * @param angle_mode Angle potential type.  
 * @param angle_params Angle potential parameters. Must include the cosine
 *                     potential parameters (K and theta0) or the dual
 *                     Gaussian mixture potential parameters (A1, A2, w1, w2,
 *                     theta1, theta2). 
 * @param dihedral_params Dihedral angle potential parameters.
 * @returns The chosen candidate move, along with its acceptance probability.  
 */
template <typename T, size_t SegmentLength>
std::pair<Matrix<T, Dynamic, 3>, T> chooseSegmentMove(PolymerConfiguration<T>& config,
                                                      const Ref<const Matrix<T, Dynamic, 3 * SegmentLength> >& moves,
                                                      const int idx,
                                                      boost::random::mt19937& rng,
                                                      boost::random::uniform_01<>& uniform_dist,
                                                      std::unordered_map<std::string, T>& lj_params, 
                                                      const T neighbor_threshold, 
                                                      std::unordered_map<std::string, T>& fene_params,
                                                      const AngleMode angle_mode, 
                                                      std::unordered_map<std::string, T>& angle_params,
                                                      std::unordered_map<std::string, T>& dihedral_params)
{
    // Calculate the energy difference for each candidate move
    const int n_tries = moves.rows();
    Matrix<T, Dynamic, 1> forward_diffs(n_tries); 
    for (int i = 0; i < n_tries; ++i)
    {
        Matrix<T, Dynamic, 3> segment(SegmentLength, 3); 
        for (int j = 0; j < SegmentLength; ++j)
        {
            segment(j, 0) = moves(i, 3 * j); 
            segment(j, 1) = moves(i, 3 * j + 1); 
            segment(j, 2) = moves(i, 3 * j + 2); 
        }
        T diff = config.getSegmentReplacementEnergyDifference(
            segment, idx, lj_params, neighbor_threshold, fene_params, 
            angle_mode, angle_params, dihedral_params
        );
        if (isnan(diff) || isinf(diff))
            forward_diffs(i) = std::numeric_limits<T>::infinity(); 
        else 
            forward_diffs(i) = diff; 
    }

    // Calculate the forward Rosenbluth factor
    Matrix<T, Dynamic, 1> forward_weights(n_tries);
    T forward_rosenbluth = 0; 
    for (int i = 0; i < n_tries; ++i)
    {
        if (isinf(forward_diffs(i)))
        {
            forward_weights(i) = 0;
        } 
        else 
        {
            forward_weights(i) = exp(-forward_diffs(i) / config.kT);
            forward_rosenbluth += forward_weights(i); 
        } 
    }

    // Choose a candidate move 
    std::vector<T> probs; 
    for (int i = 0; i < n_tries; ++i)
        probs.push_back(forward_weights(i) / forward_rosenbluth);
    boost::random::discrete_distribution<> dist(probs);  
    int move_idx = dist(rng);
    Matrix<T, Dynamic, 3> segment_new(SegmentLength, 3);
    for (int i = 0; i < SegmentLength; ++i)
    {
        segment_new(i, 0) = moves(move_idx, 3 * i); 
        segment_new(i, 1) = moves(move_idx, 3 * i + 1); 
        segment_new(i, 2) = moves(move_idx, 3 * i + 2); 
    } 

    // Generate a copy of the current polymer configuration and swap in the
    // chosen candidate move 
    PolymerConfiguration<T> config_new(config); 
    config_new.replaceSegment(segment_new, idx);

    // Calculate the energy differences between this new configuration and 
    // every other candidate move
    Matrix<T, Dynamic, 1> reverse_diffs(n_tries); 
    for (int i = 0; i < n_tries; ++i)
    {
        if (i == move_idx)
        {
            reverse_diffs(i) = -forward_diffs(i); 
        }
        else
        {
            Matrix<T, Dynamic, 3> segment(SegmentLength, 3); 
            for (int j = 0; j < SegmentLength; ++j)
            {
                segment(j, 0) = moves(i, 3 * j); 
                segment(j, 1) = moves(i, 3 * j + 1); 
                segment(j, 2) = moves(i, 3 * j + 2); 
            }
            T diff = config_new.getSegmentReplacementEnergyDifference(
                segment, idx, lj_params, neighbor_threshold, fene_params, 
                angle_mode, angle_params, dihedral_params
            );
            if (isnan(diff) || isinf(diff))
                reverse_diffs(i) = std::numeric_limits<T>::infinity(); 
            else 
                reverse_diffs(i) = diff; 
        }
    }
    
    // Calculate the reverse Rosenbluth factor
    Matrix<T, Dynamic, 1> reverse_weights(n_tries);
    T reverse_rosenbluth = 0; 
    for (int i = 0; i < n_tries; ++i)
    {
        if (isinf(reverse_diffs(i)))
        {
            reverse_weights(i) = 0;
        } 
        else 
        {
            reverse_weights(i) = exp(-reverse_diffs(i) / config.kT);
            reverse_rosenbluth += reverse_weights(i); 
        } 
    }

    // Return the chosen move with its Metropolis acceptance probability
    T prob_accept = 1; 
    if (reverse_rosenbluth > 0)
    {
        prob_accept = min(1.0, forward_rosenbluth / reverse_rosenbluth);
    }
    return std::make_pair(segment_new, prob_accept);
}

/**
 * A configurational-bias Monte Carlo function for internal segments along a
 * linear off-lattice polymer.
 *
 * @param config Current polymer configuration. 
 * @param n_tries Number of candidate moves to generate.
 * @param init_tangent_stepsize Initial stepsize for perturbation in the 
 *                              tangent space. 
 * @param min_tangent_stepsize Minimum stepsize for perturbation in the 
 *                             tangent space.  
 * @param rng Random number generator. 
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
 * @param neighbor_threshold Distance threshold for identifying neighboring
 *                           (non-bonded) atoms. 
 * @param fene_params FENE parameters. 
 * @param angle_mode Angle potential type.  
 * @param angle_params Angle potential parameters. Must include the cosine
 *                     potential parameters (K and theta0) or the dual
 *                     Gaussian mixture potential parameters (A1, A2, w1, w2,
 *                     theta1, theta2). 
 * @param dihedral_params Dihedral angle potential parameters.
 * @param dx Increment for finite difference approximation.
 * @param newton_tol Tolerance for assessing convergence of Newton's method. 
 * @param min_newton_stepsize Minimum stepsize for Newton's method. 
 * @param armijo_const Constant for Armijo condition. Set to 1e-4 by default,
 *                     following Nocedal and Wright (page 33).  
 * @returns True if the move generation procedure resulted in at least one
 *          candidate move, false if not.  
 */
template <typename T, size_t SegmentLength = 7>
bool generateInternalSegmentConcertedMove(PolymerConfiguration<T>& config,
                                          const int n_tries, 
                                          const T init_tangent_stepsize,
                                          const T min_tangent_stepsize,
                                          boost::random::mt19937& rng, 
                                          boost::random::uniform_01<>& uniform_dist,
                                          std::unordered_map<std::string, T>& lj_params, 
                                          const T neighbor_threshold, 
                                          std::unordered_map<std::string, T>& fene_params,
                                          const AngleMode angle_mode,  
                                          std::unordered_map<std::string, T>& angle_params, 
                                          std::unordered_map<std::string, T>& dihedral_params,  
                                          const T dx = 1e-8,
                                          const T newton_tol = 1e-8,
                                          const T min_newton_stepsize = 1e-4,
                                          const T armijo_const = 1e-4)
{
    // Identify the index of the segment to move
    boost::random::uniform_int_distribution<> randint_dist(1, config.getLength() - SegmentLength - 1);
    const int idx = randint_dist(rng);

    // Extract the atomic coordinates of the segment to move
    Matrix<T, 3 * SegmentLength, 1> x0;
    Matrix<T, Dynamic, 3> segment_curr = config.getSegment(idx, SegmentLength);  
    for (int i = 0; i < SegmentLength; ++i)
    {
        x0(3 * i) = segment_curr(i, 0); 
        x0(3 * i + 1) = segment_curr(i, 1); 
        x0(3 * i + 2) = segment_curr(i, 2); 
    }

    // Translate the segment to the origin
    for (int i = 0; i < SegmentLength; ++i)
        x0(Eigen::seqN(3 * i, 3)) -= segment_curr.row(0); 

    // Define the constraints to be satisfied 
    VectorValuedFunction<T, 3 * SegmentLength, 6> F
        = [&x0](const Ref<const Matrix<T, 3 * SegmentLength, 1> >& x) -> Matrix<T, 6, 1>
        {
            Matrix<T, 6, 1> v; 
            v << x(0) - x0(0), 
                 x(1) - x0(1), 
                 x(2) - x0(2), 
                 x(3 * SegmentLength - 3) - x0(3 * SegmentLength - 3), 
                 x(3 * SegmentLength - 2) - x0(3 * SegmentLength - 2), 
                 x(3 * SegmentLength - 1) - x0(3 * SegmentLength - 1);  
            return v; 
        }; 

    // Try generating the given number of moves ...
    Matrix<T, Dynamic, 3 * SegmentLength> moves(n_tries, 3 * SegmentLength); 
    int n_success = 0;  
    for (int i = 0; i < n_tries; ++i)
    {
        T tangent_stepsize = init_tangent_stepsize;
        bool found_move = false;  

        // Generate the concerted move
        //
        // First generate a random direction to move along the tangent space
        Matrix<T, 3 * SegmentLength - 6, 1> dir = randomDir<T, 3 * SegmentLength - 6>(rng, uniform_dist);
        
        // Try perturbing the input point along the sampled direction, then 
        // projecting the perturbed point onto the constraint manifold 
        Matrix<T, 3 * SegmentLength, 1> x1;  
        while (tangent_stepsize > min_tangent_stepsize)
        {
            try
            {
                x1 = generateConcertedMove<T, 3 * SegmentLength, 6>(
                    F, x0, tangent_stepsize, dir, dx, newton_tol,
                    min_newton_stepsize, armijo_const
                );
            }
            catch (const std::runtime_error& e)
            {
                // If the projection fails, try perturbing by a smaller increment
                tangent_stepsize /= 2; 
                continue; 
            }
            found_move = true;
            break;  
        }

        // If projection along the given direction was successful, ... 
        if (found_move)
        {
            // Generate the corresponding atomic coordinates 
            for (int j = 0; j < SegmentLength; ++j)
            {
                moves(n_success, 3 * j) = segment_curr(0, 0) + x1(3 * j); 
                moves(n_success, 3 * j + 1) = segment_curr(0, 1) + x1(3 * j + 1); 
                moves(n_success, 3 * j + 2) = segment_curr(0, 2) + x1(3 * j + 2);
            }
            n_success++;
        }
    }

    // If no perturbations/projections were successful, return false 
    if (n_success == 0)
        return false; 

    // Otherwise, retain only the successful projections 
    moves.conservativeResize(n_success, 3 * SegmentLength);

    // Calculate Rosenbluth weights and choose the next move
    auto result = chooseSegmentMove<T, SegmentLength>(
        config, moves, idx, rng, uniform_dist, lj_params, neighbor_threshold, 
        fene_params, angle_mode, angle_params, dihedral_params
    );

    // Change the polymer configuration with the given Metropolis acceptance
    // probability 
    T prob_accept = result.second;
    T r = uniform_dist(rng); 
    if (r < prob_accept)
    {
        Matrix<T, Dynamic, 3> segment = result.first;
        config.replaceSegment(segment, idx); 
    }

    return true;
}

/**
 * Run configurational-bias Monte Carlo sampling with the given initial 
 * polymer configuration. 
 *
 * This sampling procedure chooses, in each iteration, one of the three
 * moves (reptation, terminal segment move, internal segment move) 
 * probabilistically, according to the given array of weights.
 *
 * The returned array contains a representative sub-sample of the sampled
 * configurations.   
 *
 * @param config Initial polymer configuration.  
 * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
 * @param neighbor_threshold Distance threshold for identifying neighboring
 *                           (non-bonded) atoms. 
 * @param fene_params FENE parameters. 
 * @param angle_mode Angle potential type.  
 * @param angle_params Angle potential parameters. Must include the cosine
 *                     potential parameters (K and theta0) or the dual
 *                     Gaussian mixture potential parameters (A1, A2, w1, w2,
 *                     theta1, theta2). 
 * @param dihedral_params Dihedral angle potential parameters.
 * @param internal_move_params Additional parameters for generating internal
 *                             segment moves, to be passed into
 *                             generateInternalSegmentConcertedMove().  
 * @param rng Random number generator. 
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @param weights Array of probabilities for choosing one of reptation, 
 *                terminal segment move, or internal segment move in each 
 *                iteration.  
 * @param max_iter Maximum number of iterations. 
 * @param n_burnin Number of burn-in iterations. 
 * @param mod Modulus that determines which configurations to keep in the 
 *            returned sample, to reduce auto-correlation. 
 * @returns Representative sub-sample of sampled configurations.  
 */
template <typename T, size_t SegmentLength>
Matrix<T, Dynamic, Dynamic> runCBMC(PolymerConfiguration<T>& config,
                                    std::unordered_map<std::string, T>& lj_params,
                                    const T neighbor_threshold, 
                                    std::unordered_map<std::string, T>& fene_params,
                                    const AngleMode angle_mode,  
                                    std::unordered_map<std::string, T>& angle_params, 
                                    std::unordered_map<std::string, T>& dihedral_params,
                                    std::unordered_map<std::string, T>& internal_move_params, 
                                    boost::random::mt19937& rng,
                                    boost::random::uniform_01<>& uniform_dist,
                                    const Ref<const Matrix<T, 3, 1> >& weights, 
                                    const int n_tries, const int max_iter,
                                    const int n_burnin, const int mod)
{
    // Identify how many configurations will be collected throughout the sampling
    int n_collect = (max_iter - n_burnin) / mod; 
    Matrix<T, Dynamic, Dynamic> coords(n_collect, 3 * config.getLength()); 

    // Run sampling procedure ... 
    int curr_idx = 0; 
    int collect_idx = 0; 
    while (collect_idx < n_collect)
    {
        // Sample a move type 
        T r = uniform_dist(rng);
        CBMCMoveType move_type;         
        if (r < weights(0))
            move_type = CBMCMoveType::REPTATION; 
        else if (r < weights(0) + weights(1))
            move_type = CBMCMoveType::TERMINAL_SEGMENT; 
        else 
            move_type = CBMCMoveType::INTERNAL_SEGMENT; 

        // Generate a corresponding move 
        if (move_type == CBMCMoveType::REPTATION)
        {
            reptate<T>(
                config, n_tries, rng, uniform_dist, lj_params, neighbor_threshold,
                fene_params, angle_mode, angle_params, dihedral_params
            );
        }
        else if (move_type == CBMCMoveType::TERMINAL_SEGMENT)
        {
            moveTerminalSegment<T, SegmentLength>(
                config, n_tries, rng, uniform_dist, lj_params, neighbor_threshold, 
                fene_params, angle_mode, angle_params, dihedral_params
            );  
        } 
        else 
        {
            const T init_tangent_stepsize = internal_move_params["init_tangent_stepsize"]; 
            const T min_tangent_stepsize = internal_move_params["min_tangent_stepsize"]; 
            const T dx = internal_move_params["dx"]; 
            const T newton_tol = internal_move_params["newton_tol"]; 
            const T min_newton_stepsize = internal_move_params["min_newton_stepsize"]; 
            const T armijo_const = internal_move_params["armijo_const"]; 
            generateInternalSegmentConcertedMove<T, SegmentLength>(
                config, n_tries, init_tangent_stepsize, min_tangent_stepsize, 
                rng, uniform_dist, lj_params, neighbor_threshold, fene_params,
                angle_mode, angle_params, dihedral_params, dx, newton_tol,
                min_newton_stepsize, armijo_const 
            ); 
        }

        // Should we collect this configuration? 
        if (curr_idx >= n_burnin && (curr_idx - n_burnin) % mod == 0)
        {
            Matrix<T, Dynamic, 3> curr_coords = config.getSegment(0, config.getLength());
            for (int j = 0; j < config.getLength(); ++j)
            {
                coords(collect_idx, 3 * j) = curr_coords(j, 0); 
                coords(collect_idx, 3 * j + 1) = curr_coords(j, 1); 
                coords(collect_idx, 3 * j + 2) = curr_coords(j, 2); 
            }
            collect_idx++;
        }
        curr_idx++; 
    }

    return coords; 
}

#endif 
