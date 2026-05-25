/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     5/24/2026
 */

#ifndef MULTI_CHAIN_CONFIGURATIONAL_BIAS_MONTE_CARLO_HPP
#define MULTI_CHAIN_CONFIGURATIONAL_BIAS_MONTE_CARLO_HPP

#include <cmath>
#include <string>
#include <utility>
#include <tuple>
#include <limits>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/random.hpp>
#include "utils.hpp"
#include "polymerConfiguration.hpp"
#include "polymerEnsemble.hpp"
#include "polymerMelt.hpp"
#include "cbmc.hpp"

using std::max; 
using boost::multiprecision::max; 
using std::min; 
using boost::multiprecision::min; 
using std::exp; 
using boost::multiprecision::exp;
using std::log; 
using boost::multiprecision::log; 
using std::isnan; 
using boost::multiprecision::isnan; 
using std::isinf; 
using boost::multiprecision::isinf;
using std::ceil; 
using boost::multiprecision::ceil; 

/** ------------------------------------------------------------------- // 
 *   CONFIGURATIONAL-BIAS MONTE CARLO SAMPLER CLASS FOR POLYMER MELTS   //
 *  ------------------------------------------------------------------- */ 
template <typename T>
class PolymerMeltCBMCSampler
{
    private:
        // Current polymer melt configuration
        PolymerMeltConfiguration<T> melt_config;
        int n_chains; 

        // Polymer lengths and atomic coordinates
        std::vector<int> lengths;  
        std::vector<Matrix<T, Dynamic, 3> > r;

        // Domain bounds 
        T xmin; 
        T xmax; 
        T ymin; 
        T ymax; 
        T zmin; 
        T zmax;
        T xlen; 
        T ylen; 
        T zlen;  

        // Potential parameters 
        std::unordered_map<std::string, T> lj_params;
        T neighbor_threshold;  
        std::unordered_map<std::string, T> fene_params;
        AngleMode angle_mode;  
        std::unordered_map<std::string, T> angle_params; 
        std::unordered_map<std::string, T> dihedral_params;

        // Indicators for whether angle or dihedral potentials are trivial 
        bool no_angle_potential; 
        bool no_dihedral_potential; 

        // Random number generator and standard uniform distribution instances 
        boost::random::mt19937 rng; 
        boost::random::uniform_01<> uniform_dist; 

        // Bond length CDF
        Matrix<T, Dynamic, 2> bond_length_cdf; 

        // Bond angle and dihedral angle sampling functions 
        std::function<T()> sample_angle; 
        std::function<T()> sample_dihedral; 

        /**
         * Internal function for updating atomic coordinates after each 
         * move. 
         */
        void updateCoords()
        {
            for (int i = 0; i < this->n_chains; ++i)
                this->r[i] = this->melt_config.getSegment(i, 0, this->lengths[i]); 
        }

    public:
        /**
         * Default constructor. 
         */
        PolymerMeltCBMCSampler(PolymerMeltConfiguration<T>& melt_config,
                               std::unordered_map<std::string, T>& lj_params, 
                               const T neighbor_threshold, 
                               std::unordered_map<std::string, T>& fene_params, 
                               const AngleMode angle_mode, 
                               std::unordered_map<std::string, T>& angle_params, 
                               std::unordered_map<std::string, T>& dihedral_params, 
                               boost::random::mt19937& rng, const T xmin,
                               const T xmax, const T ymin, const T ymax, 
                               const T zmin, const T zmax,
                               const int n_bins = 10000)
        {
            this->melt_config = melt_config;
            this->n_chains = melt_config.numChains(); 
            for (int i = 0; i < this->n_chains; ++i)
            {
                int ni = this->melt_config.getLength(i); 
                this->lengths.push_back(ni); 
                this->r.push_back(this->melt_config.getSegment(i, 0, ni));  
            } 
            this->lj_params = lj_params; 
            this->neighbor_threshold = neighbor_threshold;
            this->fene_params = fene_params; 
            this->angle_mode = angle_mode; 
            this->angle_params = angle_params; 
            this->dihedral_params = dihedral_params;
            this->rng = rng;
            this->xmin = xmin; 
            this->xmax = xmax; 
            this->ymin = ymin; 
            this->ymax = ymax; 
            this->zmin = zmin; 
            this->zmax = zmax; 
            this->xlen = this->xmax - this->xmin; 
            this->ylen = this->ymax - this->ymin; 
            this->zlen = this->zmax - this->zmin; 

            // Are the bond angle and dihedral angle potentials trivial? 
            this->no_angle_potential = (
                this->angle_mode == AngleMode::COSINE && this->angle_params["K"] == 0
            );
            this->no_dihedral_potential = (this->dihedral_params["K"] == 0); 

            // Calculate FENE bond length CDF 
            this->bond_length_cdf = getFeneCDF<T>(
                this->lj_params.at("eps"), 
                this->lj_params.at("sigma"), 
                this->fene_params.at("K"), 
                this->fene_params.at("R0"), 
                this->melt_config.kT,
                n_bins 
            );

            // Define the angle sampling function  
            if (this->no_angle_potential)
            {
                this->sample_angle = [this]() -> T
                {
                    // The density should be proportional to sin(\theta)
                    T u = -1 + 2 * this->uniform_dist(this->rng);
                    return acosSafe<T>(u);  
                };
            }
            else if (this->angle_mode == AngleMode::COSINE)
            {
                this->sample_angle = [this]() -> T
                {
                    return sampleAngleCosine<T>(
                        this->angle_params.at("K"),
                        this->angle_params.at("theta0"),
                        this->melt_config.kT,
                        this->rng, 
                        this->uniform_dist
                    );
                };
            } 
            else     // this->angle_mode == AngleMode::GAUSSIAN
            {
                this->sample_angle = [this]() -> T
                {
                    return sampleAngleDualGaussianMixture<T>(
                        this->angle_params.at("A1"),
                        this->angle_params.at("A2"),
                        this->angle_params.at("w1"),
                        this->angle_params.at("w2"),
                        this->angle_params.at("theta1"),
                        this->angle_params.at("theta2"),
                        this->melt_config.kT,
                        this->rng,
                        this->uniform_dist
                    );
                };
            }

            // Define the dihedral sampling function 
            if (this->no_dihedral_potential)
            {
                this->sample_dihedral = [this]() -> T
                {
                    return (
                        -boost::math::constants::pi<T>() +
                        boost::math::constants::two_pi<T>() * this->uniform_dist(this->rng)
                    ); 
                };
            }
            else if (this->dihedral_params.find("delta") == this->dihedral_params.end())
            {
                this->sample_dihedral = [this]() -> T
                {
                    return sampleDihedralHarmonic<T>(
                        this->dihedral_params.at("K"),
                        this->melt_config.kT, 
                        this->rng, 
                        this->uniform_dist
                    ); 
                };
            } 
            else     // "delta" has been specified as an offset angle 
            {
                this->sample_dihedral = [this]() -> T
                {
                    return sampleDihedralFourierSingleComponent<T>(
                        this->dihedral_params.at("K"),
                        this->dihedral_params.at("delta"), 
                        this->melt_config.kT, 
                        this->rng,
                        this->uniform_dist
                    );
                };
            }
        }

        /**
         * Constructor with pre-computed FENE bond length density.
         */
        PolymerMeltCBMCSampler(PolymerMeltConfiguration<T>& melt_config,
                               std::unordered_map<std::string, T>& lj_params, 
                               const T neighbor_threshold, 
                               std::unordered_map<std::string, T>& fene_params, 
                               const AngleMode angle_mode, 
                               std::unordered_map<std::string, T>& angle_params, 
                               std::unordered_map<std::string, T>& dihedral_params, 
                               boost::random::mt19937& rng, const T xmin,
                               const T xmax, const T ymin, const T ymax, 
                               const T zmin, const T zmax,
                               const Ref<const Matrix<T, Dynamic, 2> >& bond_length_cdf)
        {
            this->melt_config = melt_config;
            this->n_chains = melt_config.numChains();
            for (int i = 0; i < this->n_chains; ++i)
            {
                int ni = this->melt_config.getLength(i); 
                this->lengths.push_back(ni); 
                this->r.push_back(this->melt_config.getSegment(i, 0, ni));  
            } 
            this->lj_params = lj_params; 
            this->neighbor_threshold = neighbor_threshold;
            this->fene_params = fene_params; 
            this->angle_mode = angle_mode; 
            this->angle_params = angle_params; 
            this->dihedral_params = dihedral_params;
            this->rng = rng;
            this->xmin = xmin; 
            this->xmax = xmax; 
            this->ymin = ymin; 
            this->ymax = ymax; 
            this->zmin = zmin; 
            this->zmax = zmax; 
            this->xlen = this->xmax - this->xmin; 
            this->ylen = this->ymax - this->ymin; 
            this->zlen = this->zmax - this->zmin; 
            this->bond_length_cdf = bond_length_cdf;

            // Are the bond angle and dihedral angle potentials trivial? 
            this->no_angle_potential = (
                this->angle_mode == AngleMode::COSINE && this->angle_params["K"] == 0
            );
            this->no_dihedral_potential = (this->dihedral_params["K"] == 0);

            // Define the angle sampling function  
            if (this->no_angle_potential)
            {
                this->sample_angle = [this]() -> T
                {
                    // The density should be proportional to sin(\theta)
                    T u = -1 + 2 * this->uniform_dist(this->rng);
                    return acosSafe<T>(u);  
                };
            }
            else if (this->angle_mode == AngleMode::COSINE)
            {
                this->sample_angle = [this]() -> T
                {
                    return sampleAngleCosine<T>(
                        this->angle_params.at("K"),
                        this->angle_params.at("theta0"),
                        this->melt_config.kT,
                        this->rng, 
                        this->uniform_dist
                    );
                };
            } 
            else     // this->angle_mode == AngleMode::GAUSSIAN
            {
                this->sample_angle = [this]() -> T
                {
                    return sampleAngleDualGaussianMixture<T>(
                        this->angle_params.at("A1"),
                        this->angle_params.at("A2"),
                        this->angle_params.at("w1"),
                        this->angle_params.at("w2"),
                        this->angle_params.at("theta1"),
                        this->angle_params.at("theta2"),
                        this->melt_config.kT,
                        this->rng,
                        this->uniform_dist
                    );
                };
            }

            // Define the dihedral sampling function 
            if (this->no_dihedral_potential)
            {
                this->sample_dihedral = [this]() -> T
                {
                    return (
                        -boost::math::constants::pi<T>() +
                        boost::math::constants::two_pi<T>() * this->uniform_dist(this->rng)
                    ); 
                };
            }
            else if (this->dihedral_params.find("delta") == this->dihedral_params.end())
            {
                this->sample_dihedral = [this]() -> T
                {
                    return sampleDihedralHarmonic<T>(
                        this->dihedral_params.at("K"),
                        this->melt_config.kT, 
                        this->rng, 
                        this->uniform_dist
                    ); 
                };
            } 
            else     // "delta" has been specified as an offset angle 
            {
                this->sample_dihedral = [this]() -> T
                {
                    return sampleDihedralFourierSingleComponent<T>(
                        this->dihedral_params.at("K"),
                        this->dihedral_params.at("delta"), 
                        this->melt_config.kT, 
                        this->rng,
                        this->uniform_dist
                    );
                };
            }
        }

        /**
         * Trivial destructor. 
         */
        ~PolymerMeltCBMCSampler()
        {
        } 

        /**
         * Re-seed the random number generator.
         *
         * @param seed Input seed.  
         */
        void seed(const int seed)
        {
            this->rng.seed(seed); 
        }

        /**
         * Return the current polymer configuration.
         *
         * @returns Current polymer configuration.  
         */
        PolymerConfiguration<T> getConfig(const int i)
        {
            return this->melt_config[i];
        }

        /**
         * Return the current atomic coordinates.
         *
         * @returns Current atomic coordinates.  
         */
        Matrix<T, Dynamic, 3> getCoords(const int i)
        {
            return this->r[i];
        }

        /** -------------------------------------------------------------- // 
         *                    MOVE GENERATION: REPTATION                   // 
         *  -------------------------------------------------------------- */
        /**
         * Generate possible reptation moves from the current configuration
         * for the given polymer in the melt.
         *
         * @param polymer_idx Polymer index. 
         * @param direction Reptation direction. 
         * @param n_candidates Number of candidate moves to generate.  
         * @returns Arrays of candidate atomic positions for the new atom and
         *          the corresponding residual energies.
         */
        std::pair<Matrix<T, Dynamic, 3>,
                  Matrix<T, Dynamic, 1> > generateReptationMoves(const int polymer_idx, 
                                                                 const ReptationDirection direction, 
                                                                 const int n_candidates)
        {
            const int ni = this->lengths[polymer_idx];  
            Matrix<T, Dynamic, 3> moves(n_candidates, 3);
            Matrix<T, Dynamic, 1> residuals(n_candidates); 

            // Generate bond lengths, bond angles, and dihedral angles 
            Matrix<T, Dynamic, 1> bond_lengths(n_candidates),
                                  bond_angles(n_candidates), 
                                  dihedrals(n_candidates);
            for (int i = 0; i < n_candidates; ++i)
            {
                bond_lengths(i) = sampleFene<T>(
                    this->rng, this->uniform_dist, this->bond_length_cdf 
                );
                bond_angles(i) = this->sample_angle();
                dihedrals(i) = this->sample_dihedral(); 
            }
            #ifdef CHECK_CBMC_SAMPLED_VALUES
                for (int i = 0; i < n_candidates; ++i)
                {
                    // Check that the bond lengths are within the desired range 
                    std::stringstream ss; 
                    if (bond_lengths(i) < 1e-6 || bond_lengths(i) > this->fene_params["R0"] - 1e-6)
                    {
                        ss << "Found invalid FENE bond length: " << bond_lengths(i) << std::endl; 
                        throw std::runtime_error(ss.str()); 
                    } 
                    // Check that the bond angles are within [0, 180)
                    if (bond_angles(i) < 0 || bond_angles(i) >= boost::math::constants::pi<T>())
                    {
                        ss << "Found invalid bond angle: " << bond_angles(i) << std::endl; 
                        throw std::runtime_error(ss.str()); 
                    } 
                    // Check that the dihedrals are within [-180, 180)
                    if (abs(dihedrals(i)) > boost::math::constants::pi<T>())
                    {
                        ss << "Found invalid dihedral angle: " << dihedrals(i) << std::endl; 
                        throw std::runtime_error(ss.str()); 
                    } 
                } 
            #endif
             
            if (direction == ReptationDirection::HEAD)    // Reptate towards the head 
            {
                // Generate new candidate atomic positions at the head
                Matrix<T, 3, 1> r1 = this->r[polymer_idx].row(2); 
                Matrix<T, 3, 1> r2 = this->r[polymer_idx].row(1); 
                Matrix<T, 3, 1> r3 = this->r[polymer_idx].row(0); 
                for (int i = 0; i < n_candidates; ++i)
                {
                    moves.row(i) = generateNextAtomDihedral<T>(
                        r1, r2, r3, bond_lengths(i), bond_angles(i), dihedrals(i)
                    );

                    // Get the residual energy 
                    residuals(i) = this->melt_config.getReptationResidualEnergy(
                        polymer_idx, moves.row(i), this->lj_params,
                        this->neighbor_threshold
                    );  
                }
            }
            else        // Reptate towards the tail 
            {
                // Generate new candidate atomic positions at the tail
                Matrix<T, 3, 1> r1 = this->r[polymer_idx].row(ni - 3); 
                Matrix<T, 3, 1> r2 = this->r[polymer_idx].row(ni - 2); 
                Matrix<T, 3, 1> r3 = this->r[polymer_idx].row(ni - 1); 
                for (int i = 0; i < n_candidates; ++i)
                {
                    moves.row(i) = generateNextAtomDihedral<T>(
                        r1, r2, r3, bond_lengths(i), bond_angles(i), dihedrals(i)
                    );

                    // Get the residual energy 
                    residuals(i) = this->melt_config.getReptationResidualEnergy(
                        polymer_idx, moves.row(i), this->lj_params,
                        this->neighbor_threshold
                    );  
                }
            }

            return std::make_pair(moves, residuals);
        }

        /**
         * Generate possible reptation moves from the given configuration,
         * which should differ from the current configuration, for the given
         * polymer in the melt. 
         *
         * This function should be interpreted as yielding *backward* moves 
         * from the given configuration, and *always* includes reversion to
         * the current configuration as a possible move. 
         *
         * @param polymer_idx Polymer index. 
         * @param direction Reptation direction (from the given configuration). 
         * @param n_candidates Number of candidate moves to generate. 
         * @param coords Input array of atomic coordinates.  
         * @returns Arrays of candidate atomic positions for the new atom and
         *          the corresponding residual energies.
         */
        std::pair<Matrix<T, Dynamic, 3>,
                  Matrix<T, Dynamic, 1> > generateReptationMoves(const int polymer_idx,
                                                                 const ReptationDirection direction, 
                                                                 const int n_candidates,
                                                                 const Ref<const Matrix<T, Dynamic, 3> >& coords)
        {
            const int ni = this->lengths[polymer_idx]; 
            Matrix<T, Dynamic, 3> moves(n_candidates, 3);
            Matrix<T, Dynamic, 1> residuals(n_candidates);

            // Generate new configuration with the given coordinates
            std::vector<Matrix<T, Dynamic, 3> > melt_coords; 
            for (int i = 0; i < this->n_chains; ++i)
            {
                if (i != polymer_idx)
                    melt_coords.push_back(this->r[i]); 
                else 
                    melt_coords.push_back(coords); 
            } 
            PolymerMeltConfiguration<T> melt_config_(
                this->n_chains, melt_coords, this->melt_config.getUnits(),
                this->melt_config.getTemp(), this->xmin, this->xmax,
                this->ymin, this->ymax, this->zmin, this->zmax 
            );    

            // Generate bond lengths, bond angles, and dihedral angles 
            Matrix<T, Dynamic, 1> bond_lengths(n_candidates),
                                  bond_angles(n_candidates), 
                                  dihedrals(n_candidates);
            for (int i = 0; i < n_candidates; ++i)
            {
                bond_lengths(i) = sampleFene<T>(
                    this->rng, this->uniform_dist, this->bond_length_cdf 
                );
                bond_angles(i) = this->sample_angle();
                dihedrals(i) = this->sample_dihedral(); 
            }
            #ifdef CHECK_CBMC_SAMPLED_VALUES
                for (int i = 0; i < n_candidates; ++i)
                {
                    // Check that the bond lengths are within the desired range 
                    std::stringstream ss; 
                    if (bond_lengths(i) < 1e-6 || bond_lengths(i) > this->fene_params["R0"] - 1e-6)
                    {
                        ss << "Found invalid FENE bond length: " << bond_lengths(i) << std::endl; 
                        throw std::runtime_error(ss.str()); 
                    } 
                    // Check that the bond angles are within [0, 180)
                    if (bond_angles(i) < 0 || bond_angles(i) >= boost::math::constants::pi<T>())
                    {
                        ss << "Found invalid bond angle: " << bond_angles(i) << std::endl; 
                        throw std::runtime_error(ss.str()); 
                    } 
                    // Check that the dihedrals are within [-180, 180)
                    if (abs(dihedrals(i)) > boost::math::constants::pi<T>())
                    {
                        ss << "Found invalid dihedral angle: " << dihedrals(i) << std::endl; 
                        throw std::runtime_error(ss.str()); 
                    } 
                } 
            #endif
             
            if (direction == ReptationDirection::HEAD)    // Reptate towards the head 
            {
                // Start with reversion to the current configuration
                //
                // We must have reptated toward the tail to get the given
                // configuration
                //
                // Therefore, the new atom here is the 0-th atom in the 
                // current configuration 
                moves.row(0) = this->r[polymer_idx].row(0);

                // Get the residual energy 
                residuals(0) = melt_config_.getReptationResidualEnergy(
                    polymer_idx, moves.row(0), this->lj_params,
                    this->neighbor_threshold
                ); 

                // Generate new candidate atomic positions at the head
                for (int i = 1; i < n_candidates; ++i)
                {
                    moves.row(i) = generateNextAtomDihedral<T>(
                        coords.row(2), coords.row(1), coords.row(0),
                        bond_lengths(i), bond_angles(i), dihedrals(i)
                    );

                    // Get the residual energy 
                    residuals(i) = melt_config_.getReptationResidualEnergy(
                        polymer_idx, moves.row(i), this->lj_params,
                        this->neighbor_threshold
                    );  
                }
            }
            else        // Reptate towards the tail 
            {
                // Start with reversion to the current configuration
                //
                // We must have reptated toward the head to get the given
                // configuration
                //
                // Therefore, the new atom here is the final atom in the 
                // current configuration 
                moves.row(0) = this->r[polymer_idx].row(ni - 1);

                // Get the residual energy 
                residuals(0) = melt_config_.getReptationResidualEnergy(
                    polymer_idx, moves.row(0), this->lj_params,
                    this->neighbor_threshold
                ); 

                // Generate new candidate atomic positions at the tail 
                for (int i = 1; i < n_candidates; ++i)
                {
                    moves.row(i) = generateNextAtomDihedral<T>(
                        coords.row(ni - 3), coords.row(ni - 2), coords.row(ni - 1),
                        bond_lengths(i), bond_angles(i), dihedrals(i)
                    );

                    // Get the residual energy 
                    residuals(i) = melt_config_.getReptationResidualEnergy(
                        polymer_idx, moves.row(i), this->lj_params,
                        this->neighbor_threshold
                    );  
                }
            }

            return std::make_pair(moves, residuals); 
        }

        /** -------------------------------------------------------------- // 
         *                MOVE GENERATION: MULTIMER REPTATION              // 
         *  -------------------------------------------------------------- */
        /**
         * Iteratively generate and select a multimer reptation move from the
         * current configuration for the given polymer in the melt.
         *
         * This function should be interpreted as yielding *forward* moves 
         * from the current configuration. 
         *
         * @param polymer_idx Polymer index. 
         * @param direction Reptation direction.
         * @param n_reptate Multimer length.  
         * @param n_candidates Number of candidate moves to generate per atom.  
         * @returns The chosen multimer reptation move and its corresponding
         *          (total) Rosenbluth weight. 
         */
        std::tuple<Matrix<T, Dynamic, Dynamic>, 
                   Matrix<T, Dynamic, 3>,
                   T> generateForwardMultimerReptationMove(const int polymer_idx,
                                                           const ReptationDirection direction,
                                                           const int n_reptate, 
                                                           const int n_candidates)
        {
            const int ni = this->lengths[polymer_idx]; 

            // Generate bond lengths, bond angles, and dihedral angles 
            Matrix<T, Dynamic, Dynamic> bond_lengths(n_reptate, n_candidates),
                                        bond_angles(n_reptate, n_candidates),
                                        dihedrals(n_reptate, n_candidates);
            for (int i = 0; i < n_reptate; ++i)
            {
                for (int j = 0; j < n_candidates; ++j)
                {
                    bond_lengths(i, j) = sampleFene<T>(
                        this->rng, this->uniform_dist, this->bond_length_cdf 
                    );
                    bond_angles(i, j) = this->sample_angle();
                    dihedrals(i, j) = this->sample_dihedral();
                }
            }
            #ifdef CHECK_CBMC_SAMPLED_VALUES
                for (int i = 0; i < n_reptate; ++i)
                {
                    for (int j = 0; j < n_candidates; ++j)
                    {
                        // Check that the bond lengths are within the desired range 
                        std::stringstream ss; 
                        if (bond_lengths(i, j) < 1e-6 || bond_lengths(i, j) > this->fene_params["R0"] - 1e-6)
                        {
                            ss << "Found invalid FENE bond length: " << bond_lengths(i, j) << std::endl; 
                            throw std::runtime_error(ss.str()); 
                        } 
                        // Check that the bond angles are within [0, 180)
                        if (bond_angles(i, j) < 0 || bond_angles(i, j) >= boost::math::constants::pi<T>())
                        {
                            ss << "Found invalid bond angle: " << bond_angles(i, j) << std::endl; 
                            throw std::runtime_error(ss.str()); 
                        } 
                        // Check that the dihedrals are within [-180, 180)
                        if (abs(dihedrals(i, j)) > boost::math::constants::pi<T>())
                        {
                            ss << "Found invalid dihedral angle: " << dihedrals(i, j) << std::endl; 
                            throw std::runtime_error(ss.str()); 
                        }
                    } 
                } 
            #endif

            // Keep track of the proposed atom positions, the growing segment,
            // and the total Rosenbluth weight
            Matrix<T, Dynamic, Dynamic> candidate_positions(n_candidates, 3 * n_reptate); 
            Matrix<T, Dynamic, 3> segment(0, 3); 
            T log_rosenbluth_total = 0;

            if (direction == ReptationDirection::HEAD)    // Reptate towards the head 
            {
                // For each atom ... 
                for (int i = 0; i < n_reptate; ++i)
                {
                    // Generate a collection of candidate positions for the
                    // i-th atom
                    Matrix<T, Dynamic, 3> candidates_i(n_candidates, 3);
                    Matrix<T, 3, 1> r1, r2, r3; 
                    for (int j = 0; j < n_candidates; ++j)
                    {
                        if (i == 0)
                        {
                            r1 = this->r[polymer_idx].row(2);
                            r2 = this->r[polymer_idx].row(1);
                            r3 = this->r[polymer_idx].row(0); 
                        }
                        else if (i == 1)
                        {
                            r1 = this->r[polymer_idx].row(1); 
                            r2 = this->r[polymer_idx].row(0); 
                            r3 = segment.row(0); 
                        }
                        else if (i == 2)
                        {
                            r1 = this->r[polymer_idx].row(0); 
                            r2 = segment.row(0);
                            r3 = segment.row(1); 
                        }
                        else 
                        {
                            r1 = segment.row(i - 3); 
                            r2 = segment.row(i - 2); 
                            r3 = segment.row(i - 1); 
                        }
                        candidates_i.row(j) = generateNextAtomDihedral<T>(
                            r1, r2, r3, bond_lengths(i, j), bond_angles(i, j), 
                            dihedrals(i, j)
                        );
                    }

                    // Calculate the residual energy for each candidate position
                    Matrix<T, Dynamic, 1> residuals_i(n_candidates); 
                    for (int j = 0; j < n_candidates; ++j)
                    {
                        residuals_i(j) = this->melt_config.getMultimerReptationResidualEnergy(
                            polymer_idx, ReptationDirection::HEAD, n_reptate, i,
                            segment, candidates_i.row(j), this->lj_params,
                            this->neighbor_threshold
                        ); 
                    }

                    // Calculate the corresponding atom position probabilities 
                    Matrix<T, Dynamic, 1> residuals_norm = -residuals_i / this->melt_config.kT;
                    T max_residual = residuals_norm.maxCoeff();  
                    Matrix<T, Dynamic, 1> probs(n_candidates);
                    T prob_total = 0;  
                    for (int j = 0; j < n_candidates; ++j)
                    {
                        probs(j) = exp(residuals_norm(j) - max_residual);
                        prob_total += probs(j); 
                    }
                    probs /= prob_total;

                    // Choose one candidate position
                    boost::random::discrete_distribution<> dist(probs);  
                    int move_idx = dist(this->rng);

                    // Calculate the corresponding Rosenbluth weight for the
                    // i-th atom 
                    T log_rosenbluth_i = max_residual + log(prob_total); 
                    log_rosenbluth_total += log_rosenbluth_i;

                    // Keep track of the generated positions 
                    candidate_positions(Eigen::all, Eigen::seqN(3 * i, 3)) = candidates_i;  

                    // Grow the segment 
                    segment.conservativeResize(i + 1, 3); 
                    segment.row(i) = candidates_i.row(move_idx);
                }
            }
            else        // Reptate towards the tail 
            {
                // For each atom ... 
                for (int i = 0; i < n_reptate; ++i)
                {
                    // Generate a collection of candidate positions for the
                    // i-th atom
                    Matrix<T, Dynamic, 3> candidates_i(n_candidates, 3);
                    Matrix<T, 3, 1> r1, r2, r3; 
                    for (int j = 0; j < n_candidates; ++j)
                    {
                        if (i == 0)
                        {
                            r1 = this->r[polymer_idx].row(ni - 3);
                            r2 = this->r[polymer_idx].row(ni - 2);
                            r3 = this->r[polymer_idx].row(ni - 1); 
                        }
                        else if (i == 1)
                        {
                            r1 = this->r[polymer_idx].row(ni - 2); 
                            r2 = this->r[polymer_idx].row(ni - 1); 
                            r3 = segment.row(0); 
                        }
                        else if (i == 2)
                        {
                            r1 = this->r[polymer_idx].row(ni - 1); 
                            r2 = segment.row(0);
                            r3 = segment.row(1); 
                        }
                        else 
                        {
                            r1 = segment.row(i - 3); 
                            r2 = segment.row(i - 2); 
                            r3 = segment.row(i - 1); 
                        }
                        candidates_i.row(j) = generateNextAtomDihedral<T>(
                            r1, r2, r3, bond_lengths(i, j), bond_angles(i, j), 
                            dihedrals(i, j)
                        );
                    }

                    // Calculate the residual energy for each candidate position
                    Matrix<T, Dynamic, 1> residuals_i(n_candidates); 
                    for (int j = 0; j < n_candidates; ++j)
                    {
                        residuals_i(j) = this->melt_config.getMultimerReptationResidualEnergy(
                            polymer_idx, ReptationDirection::TAIL, n_reptate,
                            i, segment, candidates_i.row(j), this->lj_params,
                            this->neighbor_threshold
                        ); 
                    }

                    // Calculate the corresponding atom position probabilities 
                    Matrix<T, Dynamic, 1> residuals_norm = -residuals_i / this->melt_config.kT;
                    T max_residual = residuals_norm.maxCoeff();  
                    Matrix<T, Dynamic, 1> probs(n_candidates);
                    T prob_total = 0;  
                    for (int j = 0; j < n_candidates; ++j)
                    {
                        probs(j) = exp(residuals_norm(j) - max_residual);
                        prob_total += probs(j); 
                    }
                    probs /= prob_total;

                    // Choose one candidate position
                    boost::random::discrete_distribution<> dist(probs);  
                    int move_idx = dist(this->rng);

                    // Calculate the corresponding Rosenbluth weight for the
                    // i-th atom 
                    T log_rosenbluth_i = max_residual + log(prob_total); 
                    log_rosenbluth_total += log_rosenbluth_i;

                    // Keep track of the generated positions 
                    candidate_positions(Eigen::all, Eigen::seqN(3 * i, 3)) = candidates_i;  

                    // Grow the segment 
                    segment.conservativeResize(i + 1, 3); 
                    segment.row(i) = candidates_i.row(move_idx);
                }
            }

            return std::make_tuple(candidate_positions, segment, log_rosenbluth_total);  
        }

        /**
         * Iteratively calculate the (backward) Rosenbluth factor corresponding
         * to reversion to the current configuration from the given configuration
         * via multimer reptation.
         *
         * @param polymer_idx Polymer index. 
         * @param direction Reptation direction (from the given configuration).
         * @param n_reptate Multimer length.  
         * @param n_candidates Number of candidate moves to generate per atom. 
         * @param coords Input array of atomic coordinates.  
         * @returns The corresponding reverse Rosenbluth factor. 
         */
        std::pair<Matrix<T, Dynamic, Dynamic>, T> getBackwardMultimerReptationRosenbluthWeight(const int polymer_idx,
                                                                                               const ReptationDirection direction,
                                                                                               const int n_reptate,
                                                                                               const int n_candidates,
                                                                                               const Ref<const Matrix<T, Dynamic, 3> >& coords)
        {
            const int ni = this->lengths[polymer_idx]; 

            // Generate new configuration with the given coordinates
            std::vector<Matrix<T, Dynamic, 3> > melt_coords; 
            for (int i = 0; i < this->n_chains; ++i)
            {
                if (i != polymer_idx)
                    melt_coords.push_back(this->r[i]); 
                else 
                    melt_coords.push_back(coords); 
            } 
            PolymerMeltConfiguration<T> melt_config_(
                this->n_chains, melt_coords, this->melt_config.getUnits(),
                this->melt_config.getTemp(), this->xmin, this->xmax, 
                this->ymin, this->ymax, this->zmin, this->zmax
            );

            // Generate bond lengths, bond angles, and dihedral angles 
            Matrix<T, Dynamic, Dynamic> bond_lengths(n_reptate, n_candidates),
                                        bond_angles(n_reptate, n_candidates),
                                        dihedrals(n_reptate, n_candidates);
            for (int i = 0; i < n_reptate; ++i)
            {
                for (int j = 0; j < n_candidates; ++j)
                {
                    bond_lengths(i, j) = sampleFene<T>(
                        this->rng, this->uniform_dist, this->bond_length_cdf 
                    );
                    bond_angles(i, j) = this->sample_angle();
                    dihedrals(i, j) = this->sample_dihedral();  
                }
            }
            #ifdef CHECK_CBMC_SAMPLED_VALUES
                for (int i = 0; i < n_reptate; ++i)
                {
                    for (int j = 0; j < n_candidates; ++j)
                    {
                        // Check that the bond lengths are within the desired range 
                        std::stringstream ss; 
                        if (bond_lengths(i, j) < 1e-6 || bond_lengths(i, j) > this->fene_params["R0"] - 1e-6)
                        {
                            ss << "Found invalid FENE bond length: " << bond_lengths(i, j) << std::endl; 
                            throw std::runtime_error(ss.str()); 
                        } 
                        // Check that the bond angles are within [0, 180)
                        if (bond_angles(i, j) < 0 || bond_angles(i, j) >= boost::math::constants::pi<T>())
                        {
                            ss << "Found invalid bond angle: " << bond_angles(i, j) << std::endl; 
                            throw std::runtime_error(ss.str()); 
                        } 
                        // Check that the dihedrals are within [-180, 180)
                        if (abs(dihedrals(i, j)) > boost::math::constants::pi<T>())
                        {
                            ss << "Found invalid dihedral angle: " << dihedrals(i, j) << std::endl; 
                            throw std::runtime_error(ss.str()); 
                        }
                    } 
                } 
            #endif

            // Keep track of the proposed atom positions and the Rosenbluth
            // weight; we are not generating a new segment
            Matrix<T, Dynamic, Dynamic> candidate_positions(n_candidates, 3 * n_reptate); 
            T log_rosenbluth_total = 0;

            // Extract the segment being re-introduced into the given configuration
            //
            // Note that the reptation direction is from the given configuration,
            // not from the current configuration 
            Matrix<T, Dynamic, 3> segment;
            Matrix<T, Dynamic, 3> curr_coords = this->r[polymer_idx]; 
            if (direction == ReptationDirection::HEAD)
            {
                // If reptating towards the head, we should work through the
                // segment backwards (from the end closer to the fixed part
                // of the chain)
                segment = curr_coords(Eigen::seqN(0, n_reptate), Eigen::all);
                segment = segment.colwise().reverse().eval();  
            } 
            else
            { 
                segment = curr_coords(Eigen::seqN(ni - n_reptate, n_reptate), Eigen::all); 
            }
            
            // Again, the reptation direction is from the given configuration,
            // not from the current configuration 
            if (direction == ReptationDirection::HEAD)    // Reptate towards the head 
            {
                // For each atom ... 
                for (int i = 0; i < n_reptate; ++i)
                {
                    // Generate a collection of candidate positions for the 
                    // i-th atom 
                    Matrix<T, Dynamic, 3> candidates_i(n_candidates, 3);
                    Matrix<T, 3, 1> r1, r2, r3;

                    // Start with reversion to the current configuration
                    //
                    // We must have reptated toward the tail to get the given
                    // configuration
                    //
                    // Therefore, the i-th atom here is the (K - i - 1)-th
                    // of the current configuration  
                    candidates_i.row(0) = curr_coords.row(n_reptate - i - 1); 

                    // Generate every other candidate position 
                    for (int j = 1; j < n_candidates; ++j)
                    {
                        if (i == 0)
                        {
                            r1 = coords.row(2);
                            r2 = coords.row(1);
                            r3 = coords.row(0); 
                        }
                        else if (i == 1)
                        {
                            r1 = coords.row(1); 
                            r2 = coords.row(0); 
                            r3 = segment.row(0); 
                        }
                        else if (i == 2)
                        {
                            r1 = coords.row(0); 
                            r2 = segment.row(0);
                            r3 = segment.row(1); 
                        }
                        else 
                        {
                            r1 = segment.row(i - 3); 
                            r2 = segment.row(i - 2); 
                            r3 = segment.row(i - 1); 
                        }
                        candidates_i.row(j) = generateNextAtomDihedral<T>(
                            r1, r2, r3, bond_lengths(i, j), bond_angles(i, j), 
                            dihedrals(i, j)
                        );
                    }

                    // Calculate the residual energy for each candidate position
                    Matrix<T, Dynamic, 1> residuals_i(n_candidates); 
                    for (int j = 0; j < n_candidates; ++j)
                    {
                        // Use the subsegment of atoms 0, ..., i - 1
                        Matrix<T, Dynamic, 3> subsegment = segment(Eigen::seqN(0, i), Eigen::all); 
                        residuals_i(j) = melt_config_.getMultimerReptationResidualEnergy(
                            polymer_idx, ReptationDirection::HEAD, n_reptate,
                            i, subsegment, candidates_i.row(j), this->lj_params,
                            this->neighbor_threshold
                        ); 
                    }

                    // Calculate the corresponding atom position probabilities 
                    Matrix<T, Dynamic, 1> residuals_norm = -residuals_i / this->melt_config.kT;
                    T max_residual = residuals_norm.maxCoeff();
                    T prob_total = 0; 
                    for (int j = 0; j < n_candidates; ++j)
                        prob_total += exp(residuals_norm(j) - max_residual); 

                    // Calculate the corresponding Rosenbluth weight for the
                    // i-th atom
                    T log_rosenbluth_i = max_residual + log(prob_total); 
                    log_rosenbluth_total += log_rosenbluth_i;

                    // Keep track of the generated positions 
                    candidate_positions(Eigen::all, Eigen::seqN(3 * i, 3)) = candidates_i;  
                }
            }
            else        // Reptate towards the tail 
            {
                // For each atom ... 
                for (int i = 0; i < n_reptate; ++i)
                {
                    // Generate a collection of candidate positions for the 
                    // i-th atom 
                    Matrix<T, Dynamic, 3> candidates_i(n_candidates, 3);
                    Matrix<T, 3, 1> r1, r2, r3;

                    // Start with reversion to the current configuration
                    //
                    // We must have reptated toward the head to get the given
                    // configuration
                    //
                    // Therefore, the i-th atom here is the (ni - K + i)-th
                    // of the current configuration  
                    candidates_i.row(0) = curr_coords.row(ni - n_reptate + i); 

                    // Generate every other candidate position 
                    for (int j = 1; j < n_candidates; ++j)
                    {
                        if (i == 0)
                        {
                            r1 = coords.row(ni - 3); 
                            r2 = coords.row(ni - 2); 
                            r3 = coords.row(ni - 1); 
                        }
                        else if (i == 1)
                        {
                            r1 = coords.row(ni - 2); 
                            r2 = coords.row(ni - 1); 
                            r3 = segment.row(0); 
                        }
                        else if (i == 2)
                        {
                            r1 = coords.row(ni - 1); 
                            r2 = segment.row(0);
                            r3 = segment.row(1); 
                        }
                        else 
                        {
                            r1 = segment.row(i - 3); 
                            r2 = segment.row(i - 2); 
                            r3 = segment.row(i - 1); 
                        }
                        candidates_i.row(j) = generateNextAtomDihedral<T>(
                            r1, r2, r3, bond_lengths(i, j), bond_angles(i, j), 
                            dihedrals(i, j)
                        );
                    }

                    // Calculate the residual energy for each candidate position
                    Matrix<T, Dynamic, 1> residuals_i(n_candidates); 
                    for (int j = 0; j < n_candidates; ++j)
                    {
                        // Use the subsegment of atoms 0, ..., i - 1
                        Matrix<T, Dynamic, 3> subsegment = segment(Eigen::seqN(0, i), Eigen::all); 
                        residuals_i(j) = melt_config_.getMultimerReptationResidualEnergy(
                            polymer_idx, ReptationDirection::TAIL, n_reptate,
                            i, subsegment, candidates_i.row(j), this->lj_params,
                            this->neighbor_threshold
                        ); 
                    }

                    // Calculate the corresponding atom position probabilities 
                    Matrix<T, Dynamic, 1> residuals_norm = -residuals_i / this->melt_config.kT;
                    T max_residual = residuals_norm.maxCoeff();
                    T prob_total = 0; 
                    for (int j = 0; j < n_candidates; ++j)
                        prob_total += exp(residuals_norm(j) - max_residual); 

                    // Calculate the corresponding Rosenbluth weight for the
                    // i-th atom
                    T log_rosenbluth_i = max_residual + log(prob_total); 
                    log_rosenbluth_total += log_rosenbluth_i;

                    // Keep track of the generated positions 
                    candidate_positions(Eigen::all, Eigen::seqN(3 * i, 3)) = candidates_i;  
                }
            }

            return std::make_pair(candidate_positions, log_rosenbluth_total);  
        }

        /** -------------------------------------------------------------- // 
         *                 MOVE GENERATION: TERMINAL SEGMENT               // 
         *  -------------------------------------------------------------- */
        /**
         * Iteratively generate and select a terminal segment move from the
         * current configuration.
         *
         * This function should be interpreted as yielding *forward* moves 
         * from the current configuration. 
         *
         * @param polymer_idx Polymer index. 
         * @param segment_length Segment length.  
         * @param direction Choice of terminal segment to move. 
         * @param n_candidates Number of candidate moves to generate. 
         * @returns The chosen terminal segment move and its corresponding 
         *          (total) Rosenbluth weight. 
         */
        std::tuple<Matrix<T, Dynamic, Dynamic>,
                   Matrix<T, Dynamic, 3>,
                   T> generateForwardTerminalSegmentMove(const int polymer_idx,
                                                         const int segment_length, 
                                                         const TerminalSegmentEnd direction,
                                                         const int n_candidates)
        {
            const int ni = this->lengths[polymer_idx];

            // Generate bond lengths, bond angles, and dihedral angles 
            Matrix<T, Dynamic, Dynamic> bond_lengths(segment_length, n_candidates),
                                        bond_angles(segment_length, n_candidates),
                                        dihedrals(segment_length, n_candidates);
            for (int i = 0; i < segment_length; ++i)
            {
                for (int j = 0; j < n_candidates; ++j)
                {
                    bond_lengths(i, j) = sampleFene<T>(
                        this->rng, this->uniform_dist, this->bond_length_cdf
                    );
                    bond_angles(i, j) = this->sample_angle();
                    dihedrals(i, j) = this->sample_dihedral();  
                }
            }
            #ifdef CHECK_CBMC_SAMPLED_VALUES
                for (int i = 0; i < segment_length; ++i)
                {
                    for (int j = 0; j < n_candidates; ++j)
                    {
                        // Check that the bond lengths are within the desired range 
                        std::stringstream ss; 
                        if (bond_lengths(i, j) < 1e-6 || bond_lengths(i, j) > this->fene_params["R0"] - 1e-6)
                        {
                            ss << "Found invalid FENE bond length: " << bond_lengths(i, j) << std::endl; 
                            throw std::runtime_error(ss.str()); 
                        } 
                        // Check that the bond angles are within [0, 180)
                        if (bond_angles(i, j) < 0 || bond_angles(i, j) >= boost::math::constants::pi<T>())
                        {
                            ss << "Found invalid bond angle: " << bond_angles(i, j) << std::endl; 
                            throw std::runtime_error(ss.str()); 
                        } 
                        // Check that the dihedrals are within [-180, 180)
                        if (abs(dihedrals(i, j)) > boost::math::constants::pi<T>())
                        {
                            ss << "Found invalid dihedral angle: " << dihedrals(i, j) << std::endl; 
                            throw std::runtime_error(ss.str()); 
                        }
                    } 
                } 
            #endif

            // Keep track of the proposed atom positions, the growing segment, 
            // and the total Rosenbluth weight
            Matrix<T, Dynamic, Dynamic> candidate_positions(n_candidates, 3 * segment_length); 
            Matrix<T, Dynamic, 3> segment(0, 3);
            T log_rosenbluth_total = 0;  

            Matrix<T, Dynamic, 3> curr_coords = this->r[polymer_idx];  
            if (direction == TerminalSegmentEnd::HEAD)    // Move the terminal segment at the head 
            {
                // For each atom in the terminal segment ...
                for (int i = 0; i < segment_length; ++i)
                {
                    // Generate a collection of candidate positions for the 
                    // i-th atom 
                    Matrix<T, Dynamic, 3> candidates_i(n_candidates, 3);
                    Matrix<T, 3, 1> r1, r2, r3;  
                    for (int j = 0; j < n_candidates; ++j)
                    {
                        // Last atom in the segment (closest to the polymer)
                        if (i == 0)
                        {
                            r1 = curr_coords.row(segment_length + 2);
                            r2 = curr_coords.row(segment_length + 1); 
                            r3 = curr_coords.row(segment_length);  
                        }
                        else if (i == 1)    // Second-to-last atom in the segment
                        {
                            r1 = curr_coords.row(segment_length + 1); 
                            r2 = curr_coords.row(segment_length); 
                            r3 = segment.row(0);    // Last atom in the segment 
                        }
                        else if (i == 2)    // Third-to-last atom in the segment
                        {
                            r1 = curr_coords.row(segment_length); 
                            r2 = segment.row(0);    // Last atom 
                            r3 = segment.row(1);    // Second-to-last
                        }
                        else 
                        {
                            r1 = segment.row(i - 3); 
                            r2 = segment.row(i - 2); 
                            r3 = segment.row(i - 1); 
                        }
                        candidates_i.row(j) = generateNextAtomDihedral<T>(
                            r1, r2, r3, bond_lengths(i, j), bond_angles(i, j), 
                            dihedrals(i, j)
                        );
                    }

                    // Calculate the residual energy for each candidate position
                    Matrix<T, Dynamic, 1> residuals_i(n_candidates); 
                    for (int j = 0; j < n_candidates; ++j)
                    {
                        residuals_i(j) = this->melt_config.getTerminalSegmentReplacementResidualEnergy(
                            polymer_idx, TerminalSegmentEnd::HEAD,
                            segment_length, i, segment, candidates_i.row(j),
                            this->lj_params, this->neighbor_threshold 
                        ); 
                    }

                    // Calculate the corresponding atom position probabilities 
                    Matrix<T, Dynamic, 1> residuals_norm = -residuals_i / this->melt_config.kT;
                    T max_residual = residuals_norm.maxCoeff();  
                    Matrix<T, Dynamic, 1> probs(n_candidates);
                    T prob_total = 0;  
                    for (int j = 0; j < n_candidates; ++j)
                    {
                        probs(j) = exp(residuals_norm(j) - max_residual);
                        prob_total += probs(j); 
                    }
                    probs /= prob_total; 

                    // Choose one candidate position
                    boost::random::discrete_distribution<> dist(probs);  
                    int move_idx = dist(this->rng);
                    
                    // Calculate the corresponding Rosenbluth weight for the
                    // i-th atom 
                    T log_rosenbluth_i = max_residual + log(prob_total); 
                    log_rosenbluth_total += log_rosenbluth_i;
                    
                    // Keep track of the generated positions 
                    candidate_positions(Eigen::all, Eigen::seqN(3 * i, 3)) = candidates_i;  

                    // Grow the segment 
                    segment.conservativeResize(i + 1, 3); 
                    segment.row(i) = candidates_i.row(move_idx);
                }
            }
            else        // Move the terminal segment at the tail 
            {
                // For each atom in the terminal segment ...
                for (int i = 0; i < segment_length; ++i)
                {
                    // Generate a collection of candidate positions for the 
                    // i-th atom
                    Matrix<T, Dynamic, 3> candidates_i(n_candidates, 3);
                    Matrix<T, 3, 1> r1, r2, r3;  
                    for (int j = 0; j < n_candidates; ++j)
                    {
                        // First atom in the segment (closest to the polymer)
                        //
                        // Since we are moving the last (segment_length) atoms
                        // in the polymer, the last atom that is not moved 
                        // has index ni - segment_length - 1
                        if (i == 0)
                        {
                            r1 = curr_coords.row(ni - segment_length - 3);
                            r2 = curr_coords.row(ni - segment_length - 2); 
                            r3 = curr_coords.row(ni - segment_length - 1);  
                        }
                        else if (i == 1)    // Second atom in the segment
                        {
                            r1 = curr_coords.row(ni - segment_length - 2); 
                            r2 = curr_coords.row(ni - segment_length - 1); 
                            r3 = segment.row(0); 
                        }
                        else if (i == 2)    // Third atom in the segment
                        {
                            r1 = curr_coords.row(ni - segment_length - 1); 
                            r2 = segment.row(0); 
                            r3 = segment.row(1);
                        }
                        else 
                        {
                            r1 = segment.row(i - 3); 
                            r2 = segment.row(i - 2); 
                            r3 = segment.row(i - 1); 
                        }
                        candidates_i.row(j) = generateNextAtomDihedral<T>(
                            r1, r2, r3, bond_lengths(i, j), bond_angles(i, j), 
                            dihedrals(i, j)
                        );
                    }

                    // Calculate the residual energy for each candidate position
                    Matrix<T, Dynamic, 1> residuals_i(n_candidates); 
                    for (int j = 0; j < n_candidates; ++j)
                    {
                        residuals_i(j) = this->melt_config.getTerminalSegmentReplacementResidualEnergy(
                            polymer_idx, TerminalSegmentEnd::TAIL,
                            segment_length, i, segment, candidates_i.row(j),
                            this->lj_params, this->neighbor_threshold 
                        ); 
                    }

                    // Calculate the corresponding atom position probabilities 
                    Matrix<T, Dynamic, 1> residuals_norm = -residuals_i / this->melt_config.kT;
                    T max_residual = residuals_norm.maxCoeff();  
                    Matrix<T, Dynamic, 1> probs(n_candidates);
                    T prob_total = 0;  
                    for (int j = 0; j < n_candidates; ++j)
                    {
                        probs(j) = exp(residuals_norm(j) - max_residual);
                        prob_total += probs(j); 
                    }
                    probs /= prob_total;

                    // Choose one candidate position
                    boost::random::discrete_distribution<> dist(probs);  
                    int move_idx = dist(this->rng);

                    // Calculate the corresponding Rosenbluth weight for the
                    // i-th atom 
                    T log_rosenbluth_i = max_residual + log(prob_total); 
                    log_rosenbluth_total += log_rosenbluth_i;

                    // Keep track of the generated positions 
                    candidate_positions(Eigen::all, Eigen::seqN(3 * i, 3)) = candidates_i;  

                    // Grow the segment 
                    segment.conservativeResize(i + 1, 3);
                    segment.row(i) = candidates_i.row(move_idx);
                }
            }

            return std::make_tuple(candidate_positions, segment, log_rosenbluth_total); 
        }

        /**
         * Iteratively calculate the (backward) Rosenbluth factor corresponding
         * to reversion to the current configuration from the given configuration
         * via a terminal segment move. 
         *
         * @param polymer_idx Polymer index. 
         * @param segment_length Segment length. 
         * @param direction Choice of terminal segment to move. 
         * @param n_candidates Number of candidate moves to generate. 
         * @param coords Input array of atomic coordinates.  
         * @returns Arrays of candidate atomic positions for the new segment
         *          and the corresponding energy differences. 
         */
        std::pair<Matrix<T, Dynamic, Dynamic>, T> getBackwardTerminalSegmentMoveRosenbluthWeight(const int polymer_idx,
                                                                                                 const int segment_length, 
                                                                                                 const TerminalSegmentEnd direction,
                                                                                                 const int n_candidates,
                                                                                                 const Ref<const Matrix<T, Dynamic, 3> >& coords)
        {
            const int ni = this->lengths[polymer_idx];

            // Generate new configuration with the given coordinates
            std::vector<Matrix<T, Dynamic, 3> > melt_coords; 
            for (int i = 0; i < this->n_chains; ++i)
            {
                if (i != polymer_idx)
                    melt_coords.push_back(this->r[i]); 
                else 
                    melt_coords.push_back(coords); 
            } 
            PolymerMeltConfiguration<T> melt_config_(
                this->n_chains, melt_coords, this->melt_config.getUnits(),
                this->melt_config.getTemp(), this->xmin, this->xmax,
                this->ymin, this->ymax, this->zmin, this->zmax 
            );

            // Generate bond lengths, bond angles, and dihedral angles 
            Matrix<T, Dynamic, Dynamic> bond_lengths(segment_length, n_candidates),
                                        bond_angles(segment_length, n_candidates),
                                        dihedrals(segment_length, n_candidates);
            for (int i = 0; i < segment_length; ++i)
            {
                for (int j = 0; j < n_candidates; ++j)
                {
                    bond_lengths(i, j) = sampleFene<T>(
                        this->rng, this->uniform_dist, this->bond_length_cdf
                    );
                    bond_angles(i, j) = this->sample_angle();
                    dihedrals(i, j) = this->sample_dihedral(); 
                }
            }
            #ifdef CHECK_CBMC_SAMPLED_VALUES
                for (int i = 0; i < segment_length; ++i)
                {
                    for (int j = 0; j < n_candidates; ++j)
                    {
                        // Check that the bond lengths are within the desired range 
                        std::stringstream ss; 
                        if (bond_lengths(i, j) < 1e-6 || bond_lengths(i, j) > this->fene_params["R0"] - 1e-6)
                        {
                            ss << "Found invalid FENE bond length: " << bond_lengths(i, j) << std::endl; 
                            throw std::runtime_error(ss.str()); 
                        } 
                        // Check that the bond angles are within [0, 180)
                        if (bond_angles(i, j) < 0 || bond_angles(i, j) >= boost::math::constants::pi<T>())
                        {
                            ss << "Found invalid bond angle: " << bond_angles(i, j) << std::endl; 
                            throw std::runtime_error(ss.str()); 
                        } 
                        // Check that the dihedrals are within [-180, 180)
                        if (abs(dihedrals(i, j)) > boost::math::constants::pi<T>())
                        {
                            ss << "Found invalid dihedral angle: " << dihedrals(i, j) << std::endl; 
                            throw std::runtime_error(ss.str()); 
                        }
                    } 
                } 
            #endif

            // Keep track of the proposed atom positions and the Rosenbluth
            // weight; we are not generating a new segment
            Matrix<T, Dynamic, Dynamic> candidate_positions(n_candidates, 3 * segment_length); 
            T log_rosenbluth_total = 0;

            // Extract the segment being re-introduced into the given configuration
            Matrix<T, Dynamic, 3> segment;
            Matrix<T, Dynamic, 3> curr_coords = this->r[polymer_idx]; 
            if (direction == TerminalSegmentEnd::HEAD)
            {
                // If moving the segment at the head, we should work through
                // the segment backwards (from the end closer to the fixed
                // part of the chain)
                segment = curr_coords(Eigen::seqN(0, segment_length), Eigen::all);
                segment = segment.colwise().reverse().eval();  
            } 
            else
            { 
                segment = curr_coords(Eigen::seqN(ni - segment_length, segment_length), Eigen::all); 
            }
             
            if (direction == TerminalSegmentEnd::HEAD)    // Move the terminal segment at the head 
            {
                // For each atom in the terminal segment ...
                for (int i = 0; i < segment_length; ++i)
                {
                    // Generate a collection of candidate positions for the 
                    // i-th atom 
                    Matrix<T, Dynamic, 3> candidates_i(n_candidates, 3);
                    Matrix<T, 3, 1> r1, r2, r3; 

                    // Start with reversion to the current configuration
                    //
                    // If we are moving the terminal segment at the head, 
                    // the i-th atom is the i-th closest to the rest of the
                    // polymer, i.e., atom at index segment_length - i - 1
                    candidates_i.row(0) = curr_coords.row(segment_length - i - 1); 

                    // Generate every other candidate position 
                    for (int j = 1; j < n_candidates; ++j)
                    {
                        // Last atom in the segment (closest to the polymer)
                        if (i == 0)
                        {
                            r1 = coords.row(segment_length + 2);
                            r2 = coords.row(segment_length + 1); 
                            r3 = coords.row(segment_length);  
                        }
                        else if (i == 1)    // Second-to-last atom in the segment
                        {
                            r1 = coords.row(segment_length + 1); 
                            r2 = coords.row(segment_length); 
                            r3 = segment.row(0);    // Last atom in the segment 
                        }
                        else if (i == 2)    // Third-to-last atom in the segment
                        {
                            r1 = coords.row(segment_length); 
                            r2 = segment.row(0);    // Last atom 
                            r3 = segment.row(1);    // Second-to-last
                        }
                        else 
                        {
                            r1 = segment.row(i - 3); 
                            r2 = segment.row(i - 2); 
                            r3 = segment.row(i - 1); 
                        }
                        candidates_i.row(j) = generateNextAtomDihedral<T>(
                            r1, r2, r3, bond_lengths(i, j), bond_angles(i, j), 
                            dihedrals(i, j)
                        );
                    }

                    // Calculate the residual energy for each candidate position
                    Matrix<T, Dynamic, 1> residuals_i(n_candidates); 
                    for (int j = 0; j < n_candidates; ++j)
                    {
                        // Use the subsegment of atoms 0, ..., i - 1
                        Matrix<T, Dynamic, 3> subsegment = segment(Eigen::seqN(0, i), Eigen::all); 
                        residuals_i(j) = melt_config_.getTerminalSegmentReplacementResidualEnergy(
                            polymer_idx, TerminalSegmentEnd::HEAD,
                            segment_length, i, subsegment, candidates_i.row(j),
                            this->lj_params, this->neighbor_threshold 
                        ); 
                    }

                    // Calculate the corresponding atom position probabilities 
                    Matrix<T, Dynamic, 1> residuals_norm = -residuals_i / this->melt_config.kT;
                    T max_residual = residuals_norm.maxCoeff();
                    T prob_total = 0; 
                    for (int j = 0; j < n_candidates; ++j)
                        prob_total += exp(residuals_norm(j) - max_residual); 

                    // Calculate the corresponding Rosenbluth weight for the
                    // i-th atom
                    T log_rosenbluth_i = max_residual + log(prob_total); 
                    log_rosenbluth_total += log_rosenbluth_i;

                    // Keep track of the generated positions 
                    candidate_positions(Eigen::all, Eigen::seqN(3 * i, 3)) = candidates_i;  
                }
            }
            else        // Move the terminal segment at the tail 
            {
                // For each atom in the terminal segment ...
                for (int i = 0; i < segment_length; ++i)
                {
                    // Generate a collection of candidate positions for the 
                    // i-th atom
                    Matrix<T, Dynamic, 3> candidates_i(n_candidates, 3);
                    Matrix<T, 3, 1> r1, r2, r3;  
                    
                    // Start with reversion to the current configuration
                    //
                    // If we are moving the terminal segment at the tail,
                    // the i-th atom has index n - segment_length + i
                    candidates_i.row(0) = curr_coords.row(ni - segment_length + i);

                    // Generate every other candidate position
                    for (int j = 1; j < n_candidates; ++j)
                    {
                        // First atom in the segment (closest to the polymer)
                        //
                        // Since we are moving the last (segment_length) atoms
                        // in the polymer, the last atom that is not moved 
                        // has index ni - segment_length - 1
                        if (i == 0)
                        {
                            r1 = coords.row(ni - segment_length - 3);
                            r2 = coords.row(ni - segment_length - 2); 
                            r3 = coords.row(ni - segment_length - 1);  
                        }
                        else if (i == 1)    // Second atom in the segment
                        {
                            r1 = coords.row(ni - segment_length - 2); 
                            r2 = coords.row(ni - segment_length - 1); 
                            r3 = segment.row(0); 
                        }
                        else if (i == 2)    // Third atom in the segment
                        {
                            r1 = coords.row(ni - segment_length - 1); 
                            r2 = segment.row(0); 
                            r3 = segment.row(1);
                        }
                        else 
                        {
                            r1 = segment.row(i - 3); 
                            r2 = segment.row(i - 2); 
                            r3 = segment.row(i - 1); 
                        }
                        candidates_i.row(j) = generateNextAtomDihedral<T>(
                            r1, r2, r3, bond_lengths(i, j), bond_angles(i, j), 
                            dihedrals(i, j)
                        );
                    }

                    // Calculate the residual energy for each candidate position
                    Matrix<T, Dynamic, 1> residuals_i(n_candidates); 
                    for (int j = 0; j < n_candidates; ++j)
                    {
                        // Use the subsegment of atoms 0, ..., i - 1
                        Matrix<T, Dynamic, 3> subsegment = segment(Eigen::seqN(0, i), Eigen::all); 
                        residuals_i(j) = melt_config_.getTerminalSegmentReplacementResidualEnergy(
                            polymer_idx, TerminalSegmentEnd::TAIL,
                            segment_length, i, subsegment, candidates_i.row(j),
                            this->lj_params, this->neighbor_threshold 
                        ); 
                    }

                    // Calculate the corresponding atom position probabilities 
                    Matrix<T, Dynamic, 1> residuals_norm = -residuals_i / this->melt_config.kT;
                    T max_residual = residuals_norm.maxCoeff();
                    T prob_total = 0; 
                    for (int j = 0; j < n_candidates; ++j)
                        prob_total += exp(residuals_norm(j) - max_residual); 

                    // Calculate the corresponding Rosenbluth weight for the
                    // i-th atom
                    T log_rosenbluth_i = max_residual + log(prob_total); 
                    log_rosenbluth_total += log_rosenbluth_i;

                    // Keep track of the generated positions 
                    candidate_positions(Eigen::all, Eigen::seqN(3 * i, 3)) = candidates_i;  
                }
            }

            return std::make_pair(candidate_positions, log_rosenbluth_total);  
        }

        /** -------------------------------------------------------------- // 
         *                 CONFIGURATIONAL-BIAS MONTE CARLO                // 
         *  -------------------------------------------------------------- */
        /**
         * Perform one iteration of configurational-bias Monte Carlo. 
         *
         * @param n_candidates Number of candidate moves to generate.
         * @param polymer_idx Polymer index.  
         * @param move_type Move type. 
         * @param segment_length Segment length for multimer reptation and 
         *                       terminal segment moves. 
         * @returns The forward and reverse candidate moves, the index of the
         *          chosen (forward) move, its Metropolis acceptance probability,
         *          whether the move was taken, and other identifying information
         *          regarding the move. 
         */
        std::tuple<Matrix<T, Dynamic, Dynamic>, 
                   Matrix<T, Dynamic, Dynamic>, 
                   int, 
                   T, 
                   CBMCMoveResult, 
                   std::unordered_map<std::string, T> > moveOnce(const int n_candidates,
                                                                 const int polymer_idx,
                                                                 const CBMCMoveType move_type,
                                                                 int segment_length)
        {
            // Perform each move type ... 
            if (move_type == CBMCMoveType::REPTATION)
            {
                // Specify a reptation direction
                const T p = this->uniform_dist(this->rng); 
                ReptationDirection rept_dir = (
                    p < 0.5 ? ReptationDirection::HEAD : ReptationDirection::TAIL
                );

                // Generate forward moves
                auto forward_result = this->generateReptationMoves(
                    polymer_idx, rept_dir, n_candidates
                );
                Matrix<T, Dynamic, 3> forward_moves = forward_result.first;
                Matrix<T, Dynamic, 1> forward_residuals = forward_result.second;

                // Calculate the move probabilities
                Matrix<T, Dynamic, 1> residuals_norm = -forward_residuals / this->melt_config.kT; 
                T max_residual = residuals_norm.maxCoeff();
                Matrix<T, Dynamic, 1> probs(n_candidates);
                T prob_total = 0;  
                for (int i = 0; i < n_candidates; ++i)
                {
                    probs(i) = exp(residuals_norm(i) - max_residual);
                    prob_total += probs(i); 
                }
                probs /= prob_total; 

                // Choose one move out of the candidates 
                boost::random::discrete_distribution<> dist(probs);  
                int move_idx = dist(this->rng);
                Matrix<T, 3, 1> r_new = forward_moves.row(move_idx);

                // Calculate the forward Rosenbluth factor 
                T log_forward_rosenbluth = max_residual + log(prob_total); 

                // Generate a copy of the current configuration and apply 
                // the forward move 
                PolymerMeltConfiguration<T> forward_config(this->melt_config);
                if (rept_dir == ReptationDirection::HEAD)
                    forward_config.reptateTowardsHead(polymer_idx, r_new); 
                else
                    forward_config.reptateTowardsTail(polymer_idx, r_new);
                #ifdef CHECK_CBMC_PLAUSIBLE_FORWARD_MOVE
                    // Calculate the energy of the new configuration 
                    T energy_nonbonded = forward_config.getTotalNonbondedEnergy(
                        this->lj_params, this->neighbor_threshold, true
                    );
                    T energy_bond = forward_config.getTotalBondEnergy(
                        this->fene_params, true, this->lj_params
                    );
                    T energy_angle = forward_config.getTotalBondAngleEnergy( 
                        this->angle_mode, this->angle_params
                    );
                    T energy_dihedral = forward_config.getTotalDihedralAngleEnergy(
                        this->dihedral_params
                    );
                    T energy_total = (
                        energy_nonbonded + energy_bond + energy_angle +
                        energy_dihedral
                    );
                    if (isnan(energy_total) || isinf(energy_total))
                    {
                        throw std::runtime_error(
                            "Proposed reptation move with infinite/undefined energy"
                        );  
                    } 
                #endif

                // Identify the reverse direction 
                ReptationDirection reverse_dir; 
                if (rept_dir == ReptationDirection::HEAD)
                    reverse_dir = ReptationDirection::TAIL; 
                else 
                    reverse_dir = ReptationDirection::HEAD;

                // Generate reverse moves
                Matrix<T, Dynamic, 3> forward_coords = forward_config.getSegment(
                    polymer_idx, 0, this->lengths[polymer_idx]
                );  
                auto reverse_result = this->generateReptationMoves(
                    polymer_idx, reverse_dir, n_candidates, forward_coords
                );
                Matrix<T, Dynamic, 3> reverse_moves = reverse_result.first;
                Matrix<T, Dynamic, 1> reverse_residuals = reverse_result.second;

                // Calculate the reverse Rosenbluth factor
                residuals_norm = -reverse_residuals / this->melt_config.kT;
                max_residual = residuals_norm.maxCoeff(); 
                prob_total = 0; 
                for (int i = 0; i < n_candidates; ++i)
                    prob_total += exp(residuals_norm(i) - max_residual);
                T log_reverse_rosenbluth = max_residual + log(prob_total);  

                // Calculate the Metropolis acceptance probability
                T log_prob_accept = min(
                    0.0, log_forward_rosenbluth - log_reverse_rosenbluth
                );

                // Change the polymer configuration according to that probability
                T r = this->uniform_dist(this->rng);
                CBMCMoveResult move_result;  
                if (log(r) < log_prob_accept)
                {
                    // Reptate towards the head 
                    if (rept_dir == ReptationDirection::HEAD)
                        this->melt_config.reptateTowardsHead(polymer_idx, r_new); 
                    else    // Reptate towards the tail 
                        this->melt_config.reptateTowardsTail(polymer_idx, r_new);
                    this->updateCoords(); 
                    move_result = CBMCMoveResult::ACCEPT; 
                }
                else 
                {
                    move_result = CBMCMoveResult::REJECT; 
                }

                // Return the forward and reverse candidate moves, the index
                // of the chosen move, its Metropolis acceptance probability, 
                // whether the move was taken, and the reptation direction
                Matrix<T, Dynamic, Dynamic> forward_moves_(forward_moves),
                                            reverse_moves_(reverse_moves); 
                std::unordered_map<std::string, T> move_info; 
                move_info["direction"] = (rept_dir == ReptationDirection::HEAD ? 0 : 1);
                return std::make_tuple(
                    forward_moves_, reverse_moves_, move_idx, exp(log_prob_accept),  
                    move_result, move_info
                ); 
            }
            else if (move_type == CBMCMoveType::MULTIMER_REPTATION)
            {
                // Specify a reptation direction
                const T p = this->uniform_dist(this->rng); 
                ReptationDirection rept_dir = (
                    p < 0.5 ? ReptationDirection::HEAD : ReptationDirection::TAIL
                ); 

                // Generate and select a forward move and get its total 
                // Rosenbluth weight
                auto forward_result = this->generateForwardMultimerReptationMove(
                    polymer_idx, rept_dir, segment_length, n_candidates
                );
                Matrix<T, Dynamic, 3> forward_move = std::get<1>(forward_result);
                T log_forward_rosenbluth = std::get<2>(forward_result);

                // Generate a copy of the current configuration and apply 
                // the forward move
                //
                // When reptating towards the head, the atomic coordinates 
                // must be mirrored 
                PolymerMeltConfiguration<T> forward_config(this->melt_config);
                if (rept_dir == ReptationDirection::HEAD)
                    forward_config.reptateTowardsHeadMultimer(
                        polymer_idx, forward_move.colwise().reverse()
                    );
                else
                    forward_config.reptateTowardsTailMultimer(
                        polymer_idx, forward_move
                    );
                #ifdef CHECK_CBMC_PLAUSIBLE_FORWARD_MOVE
                    // Calculate the energy of the new configuration 
                    T energy_nonbonded = forward_config.getTotalNonbondedEnergy(
                        this->lj_params, this->neighbor_threshold, true
                    );
                    T energy_bond = forward_config.getTotalBondEnergy(
                        this->fene_params, true, this->lj_params
                    );
                    T energy_angle = forward_config.getTotalBondAngleEnergy( 
                        this->angle_mode, this->angle_params
                    );
                    T energy_dihedral = forward_config.getTotalDihedralAngleEnergy(
                        this->dihedral_params
                    );
                    T energy_total = (
                        energy_nonbonded + energy_bond + energy_angle +
                        energy_dihedral
                    );
                    if (isnan(energy_total) || isinf(energy_total))
                    {
                        throw std::runtime_error(
                            "Proposed multimer reptation move with infinite/"
                            "undefined energy"
                        );  
                    } 
                #endif

                // Identify the reverse direction 
                ReptationDirection reverse_dir; 
                if (rept_dir == ReptationDirection::HEAD)
                    reverse_dir = ReptationDirection::TAIL; 
                else 
                    reverse_dir = ReptationDirection::HEAD;

                // Generate and select a reverse move and get its total 
                // Rosenbluth weight
                Matrix<T, Dynamic, 3> forward_coords = forward_config.getSegment(
                    polymer_idx, 0, this->lengths[polymer_idx]
                );  
                auto reverse_result = this->getBackwardMultimerReptationRosenbluthWeight(
                    polymer_idx, reverse_dir, segment_length, n_candidates,
                    forward_coords
                ); 
                T log_reverse_rosenbluth = reverse_result.second; 

                // Calculate the Metropolis acceptance probability
                T log_prob_accept = min(
                    0.0, log_forward_rosenbluth - log_reverse_rosenbluth
                );

                // Change the polymer configuration according to that probability
                T r = this->uniform_dist(this->rng);
                CBMCMoveResult move_result;  
                if (log(r) < log_prob_accept)
                {
                    // Reptate towards the head
                    //
                    // Here, again, the atomic coordinates must be mirrored 
                    if (rept_dir == ReptationDirection::HEAD)
                        this->melt_config.reptateTowardsHeadMultimer(
                            polymer_idx, forward_move.colwise().reverse()
                        ); 
                    else    // Reptate towards the tail 
                        this->melt_config.reptateTowardsTailMultimer(
                            polymer_idx, forward_move
                        );
                    this->updateCoords(); 
                    move_result = CBMCMoveResult::ACCEPT;  
                }
                else 
                {
                    move_result = CBMCMoveResult::REJECT; 
                }

                // Return the forward and reverse candidate moves (latter is 
                // ill-defined), the index of the chosen move (0), its Metropolis
                // acceptance probability, whether the move was taken, and the
                // reptation direction
                Matrix<T, Dynamic, Dynamic> forward_move_(forward_move), reverse_move;
                std::unordered_map<std::string, T> move_info; 
                move_info["direction"] = (rept_dir == ReptationDirection::HEAD ? 0 : 1);
                return std::make_tuple(
                    forward_move_, reverse_move, 0, exp(log_prob_accept),
                    move_result, move_info
                ); 
            }
            else      // move_type == CBMCMoveType::TERMINAL_SEGMENT
            {
                // Specify a terminal segment to move
                const T p = this->uniform_dist(this->rng); 
                TerminalSegmentEnd terminal_end = (
                    p < 0.5 ? TerminalSegmentEnd::HEAD : TerminalSegmentEnd::TAIL
                );

                // Generate and select a forward move and get its total 
                // Rosenbluth weight
                auto forward_result = this->generateForwardTerminalSegmentMove(
                    polymer_idx, segment_length, terminal_end, n_candidates
                );
                Matrix<T, Dynamic, 3> forward_move = std::get<1>(forward_result);  
                T log_forward_rosenbluth = std::get<2>(forward_result);

                // Generate a copy of the current configuration and apply 
                // the forward move
                //
                // When moving the terminal segment at the head, the atomic
                // coordinates must be mirrored
                PolymerMeltConfiguration<T> forward_config(this->melt_config);
                if (terminal_end == TerminalSegmentEnd::HEAD)
                    forward_config.replaceSegment(
                        polymer_idx, forward_move.colwise().reverse(), 0
                    );
                else
                    forward_config.replaceSegment(
                        polymer_idx, forward_move,
                        this->lengths[polymer_idx] - segment_length
                    );
                #ifdef CHECK_CBMC_PLAUSIBLE_FORWARD_MOVE
                    // Calculate the energy of the new configuration 
                    T energy_nonbonded = forward_config.getTotalNonbondedEnergy(
                        this->lj_params, this->neighbor_threshold, true
                    );
                    T energy_bond = forward_config.getTotalBondEnergy(
                        this->fene_params, true, this->lj_params
                    );
                    T energy_angle = forward_config.getTotalBondAngleEnergy( 
                        this->angle_mode, this->angle_params
                    );
                    T energy_dihedral = forward_config.getTotalDihedralAngleEnergy(
                        this->dihedral_params
                    );
                    T energy_total = (
                        energy_nonbonded + energy_bond + energy_angle +
                        energy_dihedral
                    );
                    if (isnan(energy_total) || isinf(energy_total))
                    {
                        throw std::runtime_error(
                            "Proposed terminal segment move with infinite/"
                            "undefined energy"
                        );  
                    } 
                #endif

                // Generate and select a reverse move and get its total 
                // Rosenbluth weight
                Matrix<T, Dynamic, 3> forward_coords = forward_config.getSegment(
                    polymer_idx, 0, this->lengths[polymer_idx]
                );  
                auto reverse_result = this->getBackwardTerminalSegmentMoveRosenbluthWeight(
                    polymer_idx, segment_length, terminal_end, n_candidates,
                    forward_coords
                );
                T log_reverse_rosenbluth = reverse_result.second; 

                // Calculate the Metropolis acceptance probability
                T log_prob_accept = min(
                    0.0, log_forward_rosenbluth - log_reverse_rosenbluth
                );

                // Change the polymer configuration according to that probability
                T r = this->uniform_dist(this->rng);
                CBMCMoveResult move_result;  
                if (log(r) < log_prob_accept)
                {
                    // Move terminal segment at the head 
                    //
                    // Here, again, the atomic coordinates must be mirrored 
                    if (terminal_end == TerminalSegmentEnd::HEAD)
                        this->melt_config.replaceSegment(
                            polymer_idx, forward_move.colwise().reverse(), 0
                        ); 
                    else    // Move terminal segment at the tail 
                        this->melt_config.replaceSegment(
                            polymer_idx, forward_move,
                            this->lengths[polymer_idx] - segment_length
                        );
                    this->updateCoords();
                    move_result = CBMCMoveResult::ACCEPT;  
                }
                else 
                {
                    move_result = CBMCMoveResult::REJECT; 
                }

                // Return the forward and reverse candidate moves (latter is 
                // ill-defined), the index of the chosen move (0), its Metropolis
                // acceptance probability, whether the move was taken, and the
                // terminal segment end 
                Matrix<T, Dynamic, Dynamic> forward_move_(forward_move), reverse_move; 
                std::unordered_map<std::string, T> move_info; 
                move_info["terminal_end"] = (terminal_end == TerminalSegmentEnd::HEAD ? 0 : 1);
                return std::make_tuple(
                    forward_move_, reverse_move, 0, exp(log_prob_accept), 
                    move_result, move_info 
                ); 
            }
        }

        /**
         * Run configurational-bias Monte Carlo sampling. 
         *
         * This sampling procedure chooses, in each iteration, one of the
         * three moves (reptation, multimer reptation, terminal segment move)
         * probabilistically, according to the given array of move probabilities. 
         *
         * The returned array contains a representative sub-sample of the
         * sampled configurations.   
         *
         * @param n_candidates Number of candidate moves to generate. 
         * @param move_probs Array of probabilities for choosing each move type.
         * @param multimer_reptation_length Multimer reptation length. 
         * @param terminal_segment_length Segment length for terminal segment
         *                                moves. 
         * @param max_iter Maximum number of iterations. 
         * @param n_burnin Number of burn-in iterations.
         * @param mod_collect Collect only one of every given number of
         *                    configurations in the sample, to reduce 
         *                    auto-correlation.
         * @param mod_write Write accumulated configurations to file once 
         *                  every this many iterations. 
         * @param max_stall Maximum number of consecutive iterations in which
         *                  the sampling "stalls" at one configuration without
         *                  accepting a new move. 
         * @param outfile Output file stream. 
         * @param verbose If true, print intermittent output to stdout.  
         * @returns Representative sub-sample of sampled configurations.  
         */
        std::vector<std::vector<Matrix<T, Dynamic, 3> > > run(const int n_candidates, 
                                                              const Ref<const Matrix<T, 3, 1> >& move_probs,
                                                              const int multimer_reptation_length, 
                                                              const int terminal_segment_length, 
                                                              const int max_iter,
                                                              const int n_burnin,
                                                              const int mod_collect,
                                                              int mod_write, 
                                                              const int max_stall,
                                                              std::ofstream& outfile,  
                                                              const bool verbose = false)
        {
            // Write sampling parameters to file
            outfile << std::setprecision(10); 
            outfile << "## n_chains = " << this->n_chains << std::endl
                    << "## domain_xmin = " << this->xmin << std::endl
                    << "## domain_xmax = " << this->xmax << std::endl
                    << "## domain_ymin = " << this->ymin << std::endl
                    << "## domain_ymax = " << this->ymax << std::endl
                    << "## domain_zmin = " << this->zmin << std::endl
                    << "## domain_zmax = " << this->zmax << std::endl 
                    << "## n_candidates = " << n_candidates << std::endl
                    << "## move_prob_reptation = "
                    << move_probs(0) << std::endl
                    << "## move_prob_multimer_reptation = "
                    << move_probs(1) << std::endl
                    << "## move_prob_terminal_segment = "
                    << move_probs(2) << std::endl
                    << "## multimer_reptation_length = "
                    << multimer_reptation_length << std::endl
                    << "## terminal_segment_length = "
                    << terminal_segment_length << std::endl
                    << "## n_bins_fene_cdf = "
                    << this->bond_length_cdf.rows() - 1 << std::endl 
                    << "## max_iter = " << max_iter << std::endl
                    << "## mod_collect = " << mod_collect << std::endl
                    << "## mod_write = " << mod_write << std::endl
                    << "## max_stall = " << max_stall << std::endl;

            // Write the initial coordinates to file 
            outfile << "# CONFIG\tINIT\n";
            for (int i = 0; i < this->n_chains; ++i)
            {
                for (int j = 0; j < this->lengths[i]; ++j)
                {
                    outfile << i << '\t' << j << '\t' << this->r[i](j, 0) << '\t'
                            << this->r[i](j, 1) << '\t'
                            << this->r[i](j, 2) << std::endl;
                } 
            }

            // Keep track of time for intermittent output to stdout
            auto t_curr = std::chrono::high_resolution_clock::now();  

            // Identify how many configurations will be collected throughout
            // the sampling
            int n_collect = (max_iter - n_burnin) / mod_collect;

            // Maintain a vector of vector of configuration coordinates 
            //
            // Array [i][j] pertains to the i-th configuration of the j-th chain
            std::vector<std::vector<Matrix<T, Dynamic, 3> > > ensemble_coords; 
            for (int i = 0; i < n_collect; ++i)
            {
                std::vector<Matrix<T, Dynamic, 3> > ensemble_coords_i;
                for (int j = 0; j < this->n_chains; ++j)
                {
                    Matrix<T, Dynamic, 3> rj = Matrix<T, Dynamic, 3>::Zero(this->lengths[j], 3); 
                    ensemble_coords_i.push_back(rj); 
                }
                ensemble_coords.push_back(ensemble_coords_i); 
            }

            // Ensure that mod_write is some multiple (>= 10) of mod_collect
            mod_write = max(
                10 * mod_collect, 
                mod_collect * static_cast<int>(ceil(mod_write / mod_collect))
            );  

            // Tabulate average acceptance probabilities for each move type 
            Matrix<T, 3, 1> accept_probs = Matrix<T, 3, 1>::Zero();
            Matrix<int, 3, 1> move_frequencies = Matrix<int, 3, 1>::Zero(); 

            // Tabulate fraction of accepted moves 
            T frac_accept = 0;

            // Define probability distribution over polymers based on their
            // lengths
            std::vector<T> polymer_probs;
            int total_length = 0; 
            for (int i = 0; i < this->n_chains; ++i)
                total_length += this->lengths[i];
            for (int i = 0; i < this->n_chains; ++i)
                polymer_probs.push_back(
                    static_cast<double>(this->lengths[i]) / total_length
                );
            boost::random::discrete_distribution<> polymer_dist(polymer_probs); 

            // Run sampling procedure ... 
            int curr_idx = 0; 
            int collect_idx = 0;
            int n_stall = 0;
            int last_written_idx = -1;  
            while (collect_idx < n_collect)
            {
                // Sample a move type 
                T r = this->uniform_dist(this->rng);
                CBMCMoveType move_type;         
                if (r < move_probs(0))
                    move_type = CBMCMoveType::REPTATION;
                else if (r < move_probs(0) + move_probs(1))
                    move_type = CBMCMoveType::MULTIMER_REPTATION;  
                else
                    move_type = CBMCMoveType::TERMINAL_SEGMENT;

                // Sample a polymer in the melt 
                const int polymer_idx = polymer_dist(this->rng);

                // Generate and accept/reject a corresponding move 
                int segment_length = 0;
                if (move_type == CBMCMoveType::MULTIMER_REPTATION)
                    segment_length = multimer_reptation_length;  
                else if (move_type == CBMCMoveType::TERMINAL_SEGMENT)
                    segment_length = terminal_segment_length; 
                auto result = this->moveOnce(
                    n_candidates, polymer_idx, move_type, segment_length
                );
                T prob_accept = std::get<3>(result);
                CBMCMoveResult accepted_move = std::get<4>(result);  
                auto move_info = std::get<5>(result);

                // Update average acceptance probabilities
                int move_type_idx = static_cast<int>(move_type);  
                accept_probs(move_type_idx) += (
                    (prob_accept - accept_probs(move_type_idx)) / (move_frequencies(move_type_idx) + 1)
                );
                move_frequencies(move_type_idx) += 1; 

                // Update total fraction of accepts 
                double outcome = (accepted_move == CBMCMoveResult::ACCEPT ? 1.0 : 0.0); 
                frac_accept += ((outcome - frac_accept) / (curr_idx + 1));  

                // Update number of consecutive stalling iterations 
                if (outcome == 0)
                    n_stall++;
                else 
                    n_stall = 0;

                // If we have exceeded the number of consecutive iterations 
                // in which the sampling has stalled, return the current 
                // sample 
                if (n_stall > max_stall)
                {
                    std::cout << "[FAIL] Sampling has stalled due to too many "
                              << "consecutive null moves or rejections"
                              << std::endl;
                    std::vector<std::vector<Matrix<T, Dynamic, 3> > > subensemble_coords(
                        ensemble_coords.begin(), ensemble_coords.begin() + collect_idx
                    ); 
                    return subensemble_coords; 
                } 

                // Print intermittent output to stdout, if desired  
                if (verbose && curr_idx % 100 == 0)
                {
                    auto t_next = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed = t_next - t_curr; 
                    std::cout << "... generated configuration " << curr_idx  
                              << ", collected " << collect_idx + 1
                              << "; time elapsed = " << elapsed.count() 
                              << ", accept probs = ["
                              << accept_probs(0) << ", "
                              << accept_probs(1) << ", "
                              << accept_probs(2) << "]"
                              << ", fraction of accepts = "
                              << frac_accept << std::endl;
                    t_curr = t_next; 
                } 

                // Decide whether to collect this configuration 
                if (curr_idx >= n_burnin && (curr_idx - n_burnin) % mod_collect == 0)
                {
                    for (int i = 0; i < this->n_chains; ++i)
                    {
                        for (int j = 0; j < this->lengths[i]; ++j)
                        {
                            ensemble_coords[collect_idx][i](j, 0) = this->r[i](j, 0); 
                            ensemble_coords[collect_idx][i](j, 1) = this->r[i](j, 1); 
                            ensemble_coords[collect_idx][i](j, 2) = this->r[i](j, 2); 
                        }
                    }
                    collect_idx++;
                }

                // Decide whether to write the configurations accumulated
                // thus far
                if (curr_idx >= n_burnin && (curr_idx - n_burnin) % mod_write == 0)
                {
                    for (int i = last_written_idx + 1; i < collect_idx; ++i)
                    {
                        // Instantiate polymer configuration and calculate 
                        // its energy and radius of gyration ... 
                        std::vector<Matrix<T, Dynamic, 3> > coords_i(ensemble_coords[i]);
                        PolymerMeltConfiguration<T> melt_config_i(
                            this->n_chains, coords_i, this->melt_config.getUnits(),
                            this->melt_config.getTemp(), this->xmin, this->xmax,
                            this->ymin, this->ymax, this->zmin, this->zmax
                        );

                        // Exclude LJ terms between consecutive atoms from
                        // the non-bonded energy
                        T energy_nonbonded = melt_config_i.getTotalNonbondedEnergy(
                            this->lj_params, this->neighbor_threshold, true
                        );

                        // Include LJ terms between consecutive atoms in 
                        // the bond energy
                        T energy_bond = melt_config_i.getTotalBondEnergy(
                            this->fene_params, true, this->lj_params
                        );

                        // Calculate bond angle and dihedral energies 
                        T energy_angle = melt_config_i.getTotalBondAngleEnergy( 
                            this->angle_mode, this->angle_params
                        );
                        T energy_dihedral = melt_config_i.getTotalDihedralAngleEnergy(
                            this->dihedral_params
                        );

                        // Sum up these contributions 
                        T energy_total = (
                            energy_nonbonded + energy_bond + energy_angle +
                            energy_dihedral
                        );
                        if (isnan(energy_total) || isinf(energy_total))
                        {
                            std::stringstream ss; 
                            ss << "Found accepted configuration " << "(" << i
                               << ") with NaN or infinite energy\n"
                               << "- Nonbonded energy: " << energy_nonbonded
                               << std::endl
                               << "- Bond energy: " << energy_bond
                               << std::endl
                               << "- Angle energy: " << energy_angle
                               << std::endl
                               << "- Dihedral energy: " << energy_dihedral
                               << std::endl
                               << "- Total energy: " << energy_total
                               << std::endl;
                            throw std::runtime_error(ss.str()); 
                        } 

                        // Write coordinates and energy terms to file
                        outfile << "# CONFIG\t" << i << std::endl
                                << "# ENERGY_TOTAL\t" << energy_total << std::endl
                                << "# ENERGY_NONBONDED\t" << energy_nonbonded << std::endl
                                << "# ENERGY_BOND\t" << energy_bond << std::endl
                                << "# ENERGY_ANGLE\t" << energy_angle << std::endl
                                << "# ENERGY_DIHEDRAL\t" << energy_dihedral << std::endl;
                        for (int j = 0; j < this->n_chains; ++j)
                        { 
                            for (int k = 0; k < this->lengths[j]; ++k)
                            {
                                outfile << j << '\t' << k << '\t'
                                        << ensemble_coords[i][j](k, 0) << '\t'
                                        << ensemble_coords[i][j](k, 1) << '\t'
                                        << ensemble_coords[i][j](k, 2) << std::endl;
                            } 
                        } 
                    } 
                    last_written_idx = collect_idx - 1;
                } 
                curr_idx++; 
            }

            // Write remaining configurations to file 
            for (int i = last_written_idx + 1; i < collect_idx; ++i)
            {
                // Instantiate polymer configuration and calculate its energy
                // and radius of gyration ... 
                std::vector<Matrix<T, Dynamic, 3> > coords_i(ensemble_coords[i]);
                PolymerMeltConfiguration<T> melt_config_i(
                    this->n_chains, coords_i, this->melt_config.getUnits(),
                    this->melt_config.getTemp(), this->xmin, this->xmax,
                    this->ymin, this->ymax, this->zmin, this->zmax
                );

                // Exclude LJ terms between consecutive atoms from the
                // non-bonded energy
                T energy_nonbonded = melt_config_i.getTotalNonbondedEnergy(
                    this->lj_params, this->neighbor_threshold, true
                );

                // Include LJ terms between consecutive atoms in the bond energy
                T energy_bond = melt_config_i.getTotalBondEnergy(
                    this->fene_params, true, this->lj_params
                );

                // Calculate bond angle and dihedral energies 
                T energy_angle = melt_config_i.getTotalBondAngleEnergy( 
                    this->angle_mode, this->angle_params
                );
                T energy_dihedral = melt_config_i.getTotalDihedralAngleEnergy(
                    this->dihedral_params
                );

                // Sum up these contributions 
                T energy_total = (
                    energy_nonbonded + energy_bond + energy_angle +
                    energy_dihedral
                );
                if (isnan(energy_total) || isinf(energy_total))
                {
                    std::stringstream ss; 
                    ss << "Found accepted configuration " << "(" << i
                       << ") with NaN or infinite energy\n"
                       << "- Nonbonded energy: " << energy_nonbonded
                       << std::endl
                       << "- Bond energy: " << energy_bond
                       << std::endl
                       << "- Angle energy: " << energy_angle
                       << std::endl
                       << "- Dihedral energy: " << energy_dihedral
                       << std::endl
                       << "- Total energy: " << energy_total
                       << std::endl;
                    throw std::runtime_error(ss.str()); 
                } 

                // Write coordinates and energy terms to file
                outfile << "# CONFIG\t" << i << std::endl
                        << "# ENERGY_TOTAL\t" << energy_total << std::endl
                        << "# ENERGY_NONBONDED\t" << energy_nonbonded << std::endl
                        << "# ENERGY_BOND\t" << energy_bond << std::endl
                        << "# ENERGY_ANGLE\t" << energy_angle << std::endl
                        << "# ENERGY_DIHEDRAL\t" << energy_dihedral << std::endl; 
                for (int j = 0; j < this->n_chains; ++j)
                { 
                    for (int k = 0; k < this->lengths[j]; ++k)
                    {
                        outfile << j << '\t' << k << '\t'
                                << ensemble_coords[i][j](k, 0) << '\t'
                                << ensemble_coords[i][j](k, 1) << '\t'
                                << ensemble_coords[i][j](k, 2) << std::endl;
                    } 
                } 
            } 

            return ensemble_coords; 
        }

        /**
         * Restart configurational-bias Monte Carlo sampling from the final
         * configuration in the given configurations file.
         *
         * The sampling procedure uses the same parameters that were used 
         * to generate the configurations in the given file.  
         *
         * @param filename Input filename.
         * @param n_collect Number of configurations to sample; this determines
         *                  the number of sampling iterations. 
         * @param outfile Output file stream. 
         * @param verbose If true, print intermittent output to stdout.  
         * @returns Representative sub-sample of sampled configurations.  
         */
        std::vector<std::vector<Matrix<T, Dynamic, 3> > > run(const std::string& filename,
                                                              const int n_collect,
                                                              std::ofstream& outfile,  
                                                              const bool verbose = false)
        {
            int n_candidates; 
            Matrix<T, 3, 1> move_probs;
            int n_bins, multimer_reptation_length, terminal_segment_length,
                mod_collect, mod_write, max_stall; 
            const int n_burnin = 0;    // Set burn-in to zero

            // Parse the given file
            auto result = parseMeltFinalConfig(
                filename, this->melt_config.getUnits(), this->melt_config.getTemp()
            );
            this->melt_config = result.first;
            std::unordered_map<std::string, T> params = result.second;
            n_candidates = static_cast<int>(params["n_candidates"]);
            move_probs << params["move_prob_reptation"],
                          params["move_prob_multimer_reptation"], 
                          params["move_prob_terminal_segment"]; 
            n_bins = static_cast<int>(params["n_bins_fene_cdf"]); 
            multimer_reptation_length = static_cast<int>(params["multimer_reptation_length"]);  
            terminal_segment_length = static_cast<int>(params["terminal_segment_length"]); 
            mod_collect = static_cast<int>(params["mod_collect"]); 
            mod_write = static_cast<int>(params["mod_write"]); 
            max_stall = static_cast<int>(params["max_stall"]);

            // Re-generate the FENE bond length CDF 
            this->bond_length_cdf = getFeneCDF<T>(
                this->lj_params.at("eps"), 
                this->lj_params.at("sigma"), 
                this->fene_params.at("K"), 
                this->fene_params.at("R0"), 
                this->melt_config.kT,
                n_bins 
            );   

            // Fix number of sampling iterations 
            const int max_iter = n_collect * mod_collect; 
            
            // Update the stored coordinates
            this->updateCoords(); 

            // Run the sampling
            return this->run(
                n_candidates, move_probs, multimer_reptation_length,
                terminal_segment_length, max_iter, n_burnin, mod_collect,
                mod_write, max_stall, outfile, verbose 
            );  
        }

        /**
         * Restart configurational-bias Monte Carlo sampling from the final
         * configuration in the given .lammpstrj file.
         *
         * Here, since the final configuration is from a .lammpstrj file, 
         * the burn-in is not automatically set to zero. 
         *
         * @param filename Input filename.
         * @param n_candidates Number of candidate moves to generate. 
         * @param move_probs Array of probabilities for choosing each move type.
         * @param multimer_reptation_length Multimer reptation length. 
         * @param terminal_segment_length Segment length for terminal segment
         *                                moves. 
         * @param max_iter Maximum number of iterations. 
         * @param n_burnin Number of burn-in iterations.
         * @param mod_collect Collect only one of every given number of
         *                    configurations in the sample, to reduce 
         *                    auto-correlation.
         * @param mod_write Write accumulated configurations to file once 
         *                  every this many iterations. 
         * @param max_stall Maximum number of consecutive iterations in which
         *                  the sampling "stalls" at one configuration without
         *                  accepting a new move. 
         * @param outfile Output file stream. 
         * @param verbose If true, print intermittent output to stdout.  
         * @returns Representative sub-sample of sampled configurations.  
         */
        std::vector<std::vector<Matrix<T, Dynamic, 3> > > run(const std::string& filename,
                                                              const int n_candidates, 
                                                              const Ref<const Matrix<T, 3, 1> >& move_probs,
                                                              const int multimer_reptation_length, 
                                                              const int terminal_segment_length, 
                                                              const int max_iter,
                                                              const int n_burnin,
                                                              const int mod_collect,
                                                              int mod_write, 
                                                              const int max_stall,
                                                              std::ofstream& outfile,  
                                                              const bool verbose = false)
        {
            // Parse the given file and extract the final configuration
            //
            // Note that the timestep does not matter here 
            auto result = parseMeltLammpstrj(
                filename, 1.0, this->melt_config.getUnits(),
                this->melt_config.getTemp()
            );
            int n_configs = result.first.size(); 
            this->melt_config = result.first[n_configs - 1];

            // Update the stored coordinates
            this->updateCoords(); 

            // Run the sampling
            return this->run(
                n_candidates, move_probs, multimer_reptation_length,
                terminal_segment_length, max_iter, n_burnin, mod_collect,
                mod_write, max_stall, outfile, verbose 
            );  
        }
};

#endif 
