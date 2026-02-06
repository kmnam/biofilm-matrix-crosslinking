/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     2/3/2026
 */

#ifndef CONFIGURATIONAL_BIAS_MONTE_CARLO_HPP
#define CONFIGURATIONAL_BIAS_MONTE_CARLO_HPP

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

using std::min; 
using boost::multiprecision::min; 
using std::exp; 
using boost::multiprecision::exp;
using std::isnan; 
using boost::multiprecision::isnan; 
using std::isinf; 
using boost::multiprecision::isinf;
using std::ceil; 
using boost::multiprecision::ceil; 

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

/** ------------------------------------------------------------------- // 
 *  CONCERTED MOVES OF INTERNAL SEGMENTS (ZAMUNER ET AL. PLOS ONE 2015) //
 *  ------------------------------------------------------------------- */ 
template <typename T, size_t DimIn, size_t DimOut>
using VectorValuedFunction = std::function<Matrix<T, DimOut, 1>(const Ref<const Matrix<T, DimIn, 1> >&)>;

template <typename T>
using DynamicVectorValuedFunction = std::function<Matrix<T, Dynamic, 1>(const Ref<const Matrix<T, Dynamic, 1> >&)>; 

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
 * Get the Jacobian of a vector-value function at the given point.
 *
 * Each partial derivative is approximated through a finite difference 
 * approximation.
 *
 * This is a version of the above function that takes and returns dynamically
 * sized matrices. 
 *
 * @param F Input function. 
 * @param x0 Input point. 
 * @param dx Increment for finite difference approximation. 
 * @returns Jacobian matrix of F at x0.  
 */
template <typename T>
Matrix<T, Dynamic, Dynamic> getJacobian(DynamicVectorValuedFunction<T>& F, 
                                        const Ref<const Matrix<T, Dynamic, 1> >& x0,
                                        const T dx = 1e-8)
{
    // Get the input/output dimensions of F
    const int dim_in = x0.size(); 
    const int dim_out = F(x0).size();

    // Compute the Jacobian entries 
    Matrix<T, Dynamic, Dynamic> J(dim_out, dim_in);
    for (int j = 0; j < dim_in; ++j)
    {
        Matrix<T, Dynamic, 1> xp(x0), xm(x0); 
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
 * Get orthonormal bases for the tangent space and its orthogonal complement
 * of a manifold, defined by F(x) = 0, at a point, x0.
 *
 * This is a version of the above function that takes and returns dynamically
 * sized matrices.  
 *
 * @param F Input function; the manifold is defined as F(x) = 0.  
 * @param x0 Input point. 
 * @param dx Increment for finite difference approximation. 
 * @returns Two matrices, the first with columns spanning the tangent space 
 *          of the Jacobian at x0, and the second with columns spanning the 
 *          orthogonal complement. 
 */
template <typename T>
std::pair<Matrix<T, Dynamic, Dynamic>, Matrix<T, Dynamic, Dynamic> >
    getTangentAndOrthogonalSpaceBases(DynamicVectorValuedFunction<T>& F, 
                                      const Ref<const Matrix<T, Dynamic, 1> >& x0, 
                                      const T dx = 1e-8)
{
    // Get the input/output dimensions of F
    const int dim_in = x0.size(); 
    const int dim_out = F(x0).size();

    // Get the Jacobian of F at x0
    Matrix<T, Dynamic, Dynamic> J = getJacobian<T>(F, x0, dx); 

    // Compute the SVD of the Jacobian 
    auto svd = J.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

    // Assume that there are (dim_in - dim_out) zero singular values,
    // corresponding to the dimension of the tangent space
    //
    // Therefore, the last (dim_in - dim_out) right singular vectors comprise
    // a basis for the tangent space
    Matrix<T, Dynamic, Dynamic> V = svd.matrixV(); 
    Matrix<T, Dynamic, Dynamic> orthogonal_basis = V(Eigen::all, Eigen::seqN(0, dim_out)); 
    Matrix<T, Dynamic, Dynamic> tangent_basis = V(Eigen::all, Eigen::seqN(dim_out, dim_in - dim_out)); 

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
 * @param max_newton_iter Maximum number of Newton iterations.  
 * @param armijo_const Constant for Armijo condition. Set to 1e-4 by default,
 *                     following Nocedal and Wright (page 33).   
 * @returns The perturbed vector, together with a residual estimating its 
 *          distance from the constraint manifold. 
 */
template <typename T, size_t DimIn, size_t DimOut>
std::pair<Matrix<T, DimIn, 1>, T> perturbAndProject(VectorValuedFunction<T, DimIn, DimOut>& F, 
                                                    const Ref<const Matrix<T, DimIn, 1> >& x0, 
                                                    const T tangent_stepsize,
                                                    const Ref<const Matrix<T, DimIn - DimOut, 1> >& dir, 
                                                    const T dx = 1e-8,
                                                    const T newton_tol = 1e-8,
                                                    const T min_newton_stepsize = 1e-4,
                                                    const int max_newton_iter = 1000,
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
    int iter = 0;  
    while (residual > newton_tol && iter < max_newton_iter)
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
        iter++;  
    }

    return std::make_pair(x1 + Qp * z_curr, residual); 
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
 * This is a version of the above function that:
 * 1) takes and returns dynamically sized matrices, and 
 * 2) uses pre-computed bases for the tangent space and orthogonal complement.
 *
 * @param F Input function; the manifold is defined as F(x) = 0.  
 * @param x0 Input point.
 * @param Qt Matrix whose columns form a basis for the tangent space of the
 *           Jacobian of F. 
 * @param Qp Matrix whose columns form a basis for the orthogonal complement
 *           of the tangent space of the Jacobian of F. 
 * @param tangent_stepsize Perturbation stepsize along the tangent space. 
 * @param dir Perturbation direction.  
 * @param dx Increment for finite difference approximation.
 * @param newton_tol Tolerance for assessing convergence of Newton's method. 
 * @param min_newton_stepsize Minimum stepsize for Newton's method.
 * @param max_newton_iter Maximum number of Newton iterations.  
 * @param armijo_const Constant for Armijo condition. Set to 1e-4 by default,
 *                     following Nocedal and Wright (page 33).   
 * @returns The perturbed vector, together with a residual estimating its 
 *          distance from the constraint manifold. 
 */
template <typename T>
std::pair<Matrix<T, Dynamic, 1>, T> perturbAndProject(DynamicVectorValuedFunction<T>& F, 
                                                      const Ref<const Matrix<T, Dynamic, 1> >& x0,
                                                      const Ref<const Matrix<T, Dynamic, Dynamic> >& Qt, 
                                                      const Ref<const Matrix<T, Dynamic, Dynamic> >& Qp,  
                                                      const T tangent_stepsize,
                                                      const Ref<const Matrix<T, Dynamic, 1> >& dir, 
                                                      const T dx = 1e-8,
                                                      const T newton_tol = 1e-8,
                                                      const T min_newton_stepsize = 1e-4,
                                                      const int max_newton_iter = 1000,
                                                      const T armijo_const = 1e-4)
{
    // Check input/output dimensions for F, Qt, and Qp, if desired 
    //
    // Note that:
    // - Qt should be (dim_in, dim_in - dim_out)
    // - Qp should be (dim_in, dim_out)
    const int dim_in = Qt.rows(); 
    const int dim_out = Qp.cols();
    #ifdef DEBUG_CHECK_MATRIX_DIMS
        assert(Qt.cols() == dim_in - dim_out);
        assert(Qp.rows() == dim_in);
        assert(x0.size() == dim_in); 
        assert(F(x0).size() == dim_out); 
        assert(dir.size() == dim_in - dim_out); 
    #endif 

    // Take a step within the tangent space
    Matrix<T, Dynamic, 1> x1 = x0 + tangent_stepsize * Qt * dir;

    // Project the resulting vector onto the orthogonal complement
    //
    // This function takes a vector of size (dim_out) and returns a vector of 
    // size (dim_in) 
    auto func = [&x1, &Qp](const Ref<const Matrix<T, Dynamic, 1> >& z) -> Matrix<T, Dynamic, 1>
    {
        return x1 + Qp * z; 
    };
    Matrix<T, Dynamic, 1> z_curr = Matrix<T, Dynamic, 1>::Zero(dim_out);    // Size dim_out 
    Matrix<T, Dynamic, 1> z_proj = func(z_curr);                            // Size dim_in
    Matrix<T, Dynamic, 1> f_curr = F(z_proj);                               // Size dim_out 
    T residual = f_curr.norm();
    int iter = 0;  
    while (residual > newton_tol && iter < max_newton_iter)
    {
        // Compute the Jacobian corresponding to the current step, times 
        // the orthogonal complement matrix 
        //
        // This matrix should be (dim_out, dim_out) 
        Matrix<T, Dynamic, Dynamic> JQ_curr = getJacobian<T>(F, z_proj, dx) * Qp;

        // Solve for the corresponding Newton step
        //
        // This vector should be size (dim_out)
        Matrix<T, Dynamic, 1> dz = JQ_curr.fullPivLu().solve(-f_curr);

        // Update z by some multiple of dz (determine the multiplier by 
        // Armijo backtracking) 
        T stepsize = 1;
        while (stepsize > min_newton_stepsize)
        {
            Matrix<T, Dynamic, 1> z_next = z_curr + stepsize * dz;    // Size dim_out
            T residual_next = F(func(z_next)).norm();
            if (residual_next <= (1 - armijo_const * stepsize) * residual)
                break; 
            stepsize *= 0.5; 
        }
        z_curr += stepsize * dz;
        z_proj = func(z_curr);       // Size dim_in 
        f_curr = F(z_proj);          // Size dim_out
        residual = f_curr.norm();
        iter++;  
    }

    return std::make_pair(x1 + Qp * z_curr, residual);     // Size dim_in 
}

/** ------------------------------------------------------------------- // 
 *             CONFIGURATIONAL-BIAS MONTE CARLO SAMPLER CLASS           //
 *  ------------------------------------------------------------------- */ 
template <typename T>
class PolymerCBMCSampler
{
    private:
        // Current polymer configuration
        PolymerConfiguration<T> config;

        // Length and atomic coordinates 
        int length; 
        Matrix<T, Dynamic, 3> r;

        // Potential parameters 
        std::unordered_map<std::string, T> lj_params;
        T neighbor_threshold;  
        std::unordered_map<std::string, T> fene_params;
        AngleMode angle_mode;  
        std::unordered_map<std::string, T> angle_params; 
        std::unordered_map<std::string, T> dihedral_params; 

        // Random number generator and standard uniform distribution instances 
        boost::random::mt19937 rng; 
        boost::random::uniform_01<> uniform_dist;  

        /**
         * Internal function for updating atomic coordinates after each 
         * move. 
         */
        void updateCoords()
        {
            this->r = this->config.getSegment(0, this->length); 
        }

    public:
        /**
         * Default constructor. 
         */
        PolymerCBMCSampler(PolymerConfiguration<T>& config,
                           std::unordered_map<std::string, T>& lj_params, 
                           const T neighbor_threshold, 
                           std::unordered_map<std::string, T>& fene_params, 
                           const AngleMode angle_mode, 
                           std::unordered_map<std::string, T>& angle_params, 
                           std::unordered_map<std::string, T>& dihedral_params, 
                           boost::random::mt19937& rng)
        {
            this->config = config;
            this->length = this->config.getLength();
            this->r = this->config.getSegment(0, this->length); 
            this->lj_params = lj_params; 
            this->neighbor_threshold = neighbor_threshold;
            this->fene_params = fene_params; 
            this->angle_mode = angle_mode; 
            this->angle_params = angle_params; 
            this->dihedral_params = dihedral_params;
            this->rng = rng;  
        }

        /**
         * Trivial destructor. 
         */
        ~PolymerCBMCSampler()
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
        PolymerConfiguration<T> getConfig()
        {
            return this->config; 
        }

        /**
         * Return the current atomic coordinates.
         *
         * @returns Current atomic coordinates.  
         */
        Matrix<T, Dynamic, 3> getCoords()
        {
            return this->r;
        }

        /** -------------------------------------------------------------- // 
         *                    MOVE GENERATION: REPTATION                   // 
         *  -------------------------------------------------------------- */
        /**
         * Generate possible reptation moves from the current configuration.
         *
         * @param direction Reptation direction. 
         * @param n_candidates Number of candidate moves to generate.  
         * @returns Arrays of candidate atomic positions for the new atom and
         *          the corresponding reptation energy differences.  
         */
        std::pair<Matrix<T, Dynamic, Dynamic>,
                  Matrix<T, Dynamic, 1> > generateReptationMoves(const ReptationDirection direction, 
                                                                 const int n_candidates)
        {
            const int n = this->length; 
            Matrix<T, Dynamic, Dynamic> moves(n_candidates, 3);
            Matrix<T, Dynamic, 1> energy_diffs(n_candidates); 

            // Define the angle sampling function  
            std::function<T()> sample_angle;
            if (this->angle_mode == AngleMode::COSINE)
            {
                sample_angle = [this]() -> T
                {
                    return sampleAngleCosine<T>(
                        this->angle_params.at("K"),
                        this->angle_params.at("theta0"),
                        this->config.kT,
                        this->rng, 
                        this->uniform_dist, 50
                    );
                };
            } 
            else     // this->angle_mode == AngleMode::GAUSSIAN
            {
                sample_angle = [this]() -> T
                {
                    return sampleAngleDualGaussianMixture<T>(
                        this->angle_params.at("A1"),
                        this->angle_params.at("A2"),
                        this->angle_params.at("w1"),
                        this->angle_params.at("w2"),
                        this->angle_params.at("theta1"),
                        this->angle_params.at("theta2"),
                        this->config.kT,
                        this->rng,
                        this->uniform_dist, 50
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
                    this->lj_params.at("eps"), this->lj_params.at("sigma"),
                    this->fene_params.at("K"), this->fene_params.at("R0"),
                    this->config.kT, this->rng, this->uniform_dist, 50 
                );
                angles(i) = sample_angle();
                dihedrals(i) = sampleDihedralHarmonic<T>(
                    this->dihedral_params.at("K"), this->config.kT, this->rng,
                    this->uniform_dist
                );
            }
             
            if (direction == ReptationDirection::HEAD)    // Reptate towards the head 
            {
                // Generate new candidate atomic positions at the head
                for (int i = 0; i < n_candidates; ++i)
                {
                    moves.row(i) = generateNextAtomDihedral<T>(
                        this->r.row(2), this->r.row(1), this->r.row(0),
                        lengths(i, 0), angles(i, 0), dihedrals(i, 0),
                        this->rng, this->uniform_dist,
                        (dihedrals(i, 0) > 0 ? 1 : -1)
                    );

                    // Get the non-bonded energy difference due to reptation
                    energy_diffs(i) = this->config.getReptationNonbondedEnergyDifference(
                        ReptationDirection::HEAD, moves.row(i),
                        this->lj_params, this->neighbor_threshold
                    ); 
                }
            }
            else        // Reptate towards the tail 
            {
                // Generate new candidate atomic positions at the tail 
                for (int i = 0; i < n_candidates; ++i)
                {
                    moves.row(i) = generateNextAtomDihedral<T>(
                        this->r.row(n - 3), this->r.row(n - 2), this->r.row(n - 1),
                        lengths(i, 0), angles(i, 0), dihedrals(i, 0),
                        this->rng, this->uniform_dist,
                        (dihedrals(i, 0) > 0 ? 1 : -1)
                    );

                    // Get the non-bonded energy difference due to reptation
                    energy_diffs(i) = this->config.getReptationNonbondedEnergyDifference(
                        ReptationDirection::TAIL, moves.row(i),
                        this->lj_params, this->neighbor_threshold
                    ); 
                }
            }

            return std::make_pair(moves, energy_diffs); 
        }

        /**
         * Generate possible reptation moves from the given configuration
         * (which may differ from the current configuration).
         *
         * @param direction Reptation direction. 
         * @param n_candidates Number of candidate moves to generate. 
         * @param coords Input array of atomic coordinates.  
         * @returns Arrays of candidate atomic positions for the new atom and
         *          the corresponding reptation energy differences.  
         */
        std::pair<Matrix<T, Dynamic, Dynamic>,
                  Matrix<T, Dynamic, 1> > generateReptationMoves(const ReptationDirection direction, 
                                                                 const int n_candidates,
                                                                 const Ref<const Matrix<T, Dynamic, 3> >& coords)
        {
            const int n = this->length; 
            Matrix<T, Dynamic, Dynamic> moves(n_candidates, 3);
            Matrix<T, Dynamic, 1> energy_diffs(n_candidates);

            // Generate new configuration with the given coordinates 
            PolymerConfiguration<T> config_(
                coords, this->config.getUnits(), this->config.getTemp()
            );    

            // Define the angle sampling function  
            std::function<T()> sample_angle;
            if (this->angle_mode == AngleMode::COSINE)
            {
                sample_angle = [this]() -> T
                {
                    return sampleAngleCosine<T>(
                        this->angle_params.at("K"),
                        this->angle_params.at("theta0"),
                        this->config.kT,
                        this->rng, 
                        this->uniform_dist, 50
                    );
                };
            } 
            else     // this->angle_mode == AngleMode::GAUSSIAN
            {
                sample_angle = [this]() -> T
                {
                    return sampleAngleDualGaussianMixture<T>(
                        this->angle_params.at("A1"),
                        this->angle_params.at("A2"),
                        this->angle_params.at("w1"),
                        this->angle_params.at("w2"),
                        this->angle_params.at("theta1"),
                        this->angle_params.at("theta2"),
                        this->config.kT,
                        this->rng,
                        this->uniform_dist, 50
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
                    this->lj_params.at("eps"), this->lj_params.at("sigma"),
                    this->fene_params.at("K"), this->fene_params.at("R0"),
                    config_.kT, this->rng, this->uniform_dist, 50 
                );
                angles(i) = sample_angle();
                dihedrals(i) = sampleDihedralHarmonic<T>(
                    this->dihedral_params.at("K"), config_.kT, this->rng,
                    this->uniform_dist
                );
            }
             
            if (direction == ReptationDirection::HEAD)    // Reptate towards the head 
            {
                // Generate new candidate atomic positions at the head
                for (int i = 0; i < n_candidates; ++i)
                {
                    moves.row(i) = generateNextAtomDihedral<T>(
                        coords.row(2), coords.row(1), coords.row(0),
                        lengths(i, 0), angles(i, 0), dihedrals(i, 0),
                        this->rng, this->uniform_dist,
                        (dihedrals(i, 0) > 0 ? 1 : -1)
                    );

                    // Get the non-bonded energy difference due to reptation
                    energy_diffs(i) = config_.getReptationNonbondedEnergyDifference(
                        ReptationDirection::HEAD, moves.row(i),
                        this->lj_params, this->neighbor_threshold
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
                        lengths(i, 0), angles(i, 0), dihedrals(i, 0),
                        this->rng, this->uniform_dist,
                        (dihedrals(i, 0) > 0 ? 1 : -1)
                    );

                    // Get the non-bonded energy difference due to reptation
                    energy_diffs(i) = config_.getReptationNonbondedEnergyDifference(
                        ReptationDirection::TAIL, moves.row(i),
                        this->lj_params, this->neighbor_threshold
                    ); 
                }
            }

            return std::make_pair(moves, energy_diffs); 
        }

        /** -------------------------------------------------------------- // 
         *                 MOVE GENERATION: TERMINAL SEGMENT               // 
         *  -------------------------------------------------------------- */
        /**
         * Generate possible terminal segment moves from the current 
         * configuration.
         *
         * @param direction Choice of terminal segment to move. 
         * @param n_candidates Number of candidate moves to generate. 
         * @param segment_length Segment length.  
         * @returns Arrays of candidate atomic positions for the new segment
         *          and the corresponding energy differences. 
         */
        std::pair<Matrix<T, Dynamic, Dynamic>,
                  Matrix<T, Dynamic, 1> > generateTerminalSegmentMoves(const int segment_length, 
                                                                       const TerminalSegmentEnd direction,
                                                                       const int n_candidates)
        {
            const int n = this->length;
            Matrix<T, Dynamic, Dynamic> moves(n_candidates, 3 * segment_length); 
            Matrix<T, Dynamic, 1> energy_diffs(n_candidates);

            // Define the angle sampling function  
            std::function<T()> sample_angle;
            if (this->angle_mode == AngleMode::COSINE)
            {
                sample_angle = [this]() -> T
                {
                    return sampleAngleCosine<T>(
                        this->angle_params.at("K"),
                        this->angle_params.at("theta0"),
                        this->config.kT,
                        this->rng, 
                        this->uniform_dist, 50
                    );
                };
            } 
            else     // this->angle_mode == AngleMode::GAUSSIAN
            {
                sample_angle = [this]() -> T
                {
                    return sampleAngleDualGaussianMixture<T>(
                        this->angle_params.at("A1"),
                        this->angle_params.at("A2"),
                        this->angle_params.at("w1"),
                        this->angle_params.at("w2"),
                        this->angle_params.at("theta1"),
                        this->angle_params.at("theta2"),
                        this->config.kT,
                        this->rng,
                        this->uniform_dist, 50
                    );
                };
            }

            // Generate bond lengths, bond angles, and dihedral angles 
            Matrix<T, Dynamic, Dynamic> lengths(n_candidates, segment_length),
                                        angles(n_candidates, segment_length),
                                        dihedrals(n_candidates, segment_length);
            for (int i = 0; i < n_candidates; ++i)
            {
                for (int j = 0; j < segment_length; ++j)
                {
                    lengths(i, j) = sampleFene<T>(
                        this->lj_params.at("eps"), this->lj_params.at("sigma"),
                        this->fene_params.at("K"), this->fene_params.at("R0"),
                        this->config.kT, this->rng, this->uniform_dist, 50 
                    );
                    angles(i, j) = sample_angle();
                    dihedrals(i, j) = sampleDihedralHarmonic<T>(
                        this->dihedral_params.at("K"), this->config.kT,
                        this->rng, this->uniform_dist
                    );
                }
            }
             
            if (direction == TerminalSegmentEnd::HEAD)    // Move the terminal segment at the head 
            {
                // Generate new candidate atomic positions for the head segment
                for (int i = 0; i < n_candidates; ++i)
                {
                    Matrix<T, Dynamic, 3> segment_i(segment_length, 3); 

                    // Move backwards from atom (segment_length) in the polymer  
                    for (int j = 0; j < segment_length; ++j)
                    {
                        Matrix<T, 3, 1> r1, r2, r3; 
                        if (j == 0)         // Last atom in the segment (closest to the polymer)
                        {
                            r1 = this->r.row(segment_length + 2);
                            r2 = this->r.row(segment_length + 1); 
                            r3 = this->r.row(segment_length);  
                        }
                        else if (j == 1)    // Second-to-last
                        {
                            r1 = this->r.row(segment_length + 1); 
                            r2 = this->r.row(segment_length); 
                            r3 = segment_i.row(segment_length - 1);  
                        }
                        else if (j == 2)    // Third-to-last
                        {
                            r1 = this->r.row(segment_length); 
                            r2 = segment_i.row(segment_length - 1);
                            r3 = segment_i.row(segment_length - 2); 
                        }
                        else 
                        {
                            r1 = segment_i.row(segment_length - 1 - j + 3); 
                            r2 = segment_i.row(segment_length - 1 - j + 2); 
                            r3 = segment_i.row(segment_length - 1 - j + 1);  
                        }
                        int idx = segment_length - 1 - j;
                        segment_i.row(idx) = generateNextAtomDihedral<T>(
                            r1, r2, r3, lengths(i, j), angles(i, j),
                            dihedrals(i, j), this->rng, this->uniform_dist,
                            (dihedrals(i, j) > 0 ? 1 : -1)
                        );
                        moves(i, Eigen::seqN(3 * idx, 3)) = segment_i.row(idx); 
                    }

                    // Get the non-bonded energy difference due to segment replacement
                    energy_diffs(i) = this->config.getSegmentReplacementNonbondedEnergyDifference(
                        segment_i, 0, this->lj_params, this->neighbor_threshold
                    ); 
                }
            }
            else        // Move the terminal segment at the tail 
            {
                // Generate new candidate atomic positions for the tail segment
                for (int i = 0; i < n_candidates; ++i)
                {
                    Matrix<T, Dynamic, 3> segment_i(segment_length, 3); 

                    // Move forward from atom (n - segment_length) in the polymer 
                    for (int j = 0; j < segment_length; ++j)
                    {
                        Matrix<T, 3, 1> r1, r2, r3; 
                        if (j == 0)
                        {
                            r1 = this->r.row(n - segment_length - 3);
                            r2 = this->r.row(n - segment_length - 2); 
                            r3 = this->r.row(n - segment_length - 1);  
                        }
                        else if (j == 1)
                        {
                            r1 = this->r.row(n - segment_length - 2); 
                            r2 = this->r.row(n - segment_length - 1); 
                            r3 = segment_i.row(0); 
                        }
                        else if (j == 2)
                        {
                            r1 = this->r.row(n - segment_length - 1); 
                            r2 = segment_i.row(0); 
                            r3 = segment_i.row(1);
                        }
                        else 
                        {
                            r1 = segment_i.row(j - 3);
                            r2 = segment_i.row(j - 2); 
                            r3 = segment_i.row(j - 1); 
                        }
                        segment_i.row(j) = generateNextAtomDihedral<T>(
                            r1, r2, r3, lengths(i, j), angles(i, j),
                            dihedrals(i, j), this->rng, this->uniform_dist,
                            (dihedrals(i, j) > 0 ? 1 : -1)
                        );
                        moves(i, Eigen::seqN(3 * j, 3)) = segment_i.row(j); 
                    }

                    // Get the non-bonded energy difference due to segment replacement
                    energy_diffs(i) = this->config.getSegmentReplacementNonbondedEnergyDifference(
                        segment_i, n - segment_length, this->lj_params,
                        this->neighbor_threshold
                    ); 
                }
            }

            return std::make_pair(moves, energy_diffs); 
        }

        /**
         * Generate possible terminal segment moves from the given 
         * configuration (which may differ from the current configuration).
         *
         * @param direction Choice of terminal segment to move. 
         * @param n_candidates Number of candidate moves to generate. 
         * @param segment_length Segment length. 
         * @param coords Input array of atomic coordinates.  
         * @returns Arrays of candidate atomic positions for the new segment
         *          and the corresponding energy differences. 
         */
        std::pair<Matrix<T, Dynamic, Dynamic>,
                  Matrix<T, Dynamic, 1> > generateTerminalSegmentMoves(const int segment_length, 
                                                                       const TerminalSegmentEnd direction,
                                                                       const int n_candidates,
                                                                       const Ref<const Matrix<T, Dynamic, 3> >& coords)
        {
            const int n = this->length;
            Matrix<T, Dynamic, Dynamic> moves(n_candidates, 3 * segment_length); 
            Matrix<T, Dynamic, 1> energy_diffs(n_candidates);

            // Generate new configuration with the given coordinates 
            PolymerConfiguration<T> config_(
                coords, this->config.getUnits(), this->config.getTemp()
            );    

            // Define the angle sampling function  
            std::function<T()> sample_angle;
            if (this->angle_mode == AngleMode::COSINE)
            {
                sample_angle = [this]() -> T
                {
                    return sampleAngleCosine<T>(
                        this->angle_params.at("K"),
                        this->angle_params.at("theta0"),
                        this->config.kT,
                        this->rng, 
                        this->uniform_dist, 50
                    );
                };
            } 
            else     // this->angle_mode == AngleMode::GAUSSIAN
            {
                sample_angle = [this]() -> T
                {
                    return sampleAngleDualGaussianMixture<T>(
                        this->angle_params.at("A1"),
                        this->angle_params.at("A2"),
                        this->angle_params.at("w1"),
                        this->angle_params.at("w2"),
                        this->angle_params.at("theta1"),
                        this->angle_params.at("theta2"),
                        this->config.kT,
                        this->rng,
                        this->uniform_dist, 50
                    );
                };
            }

            // Generate bond lengths, bond angles, and dihedral angles 
            Matrix<T, Dynamic, Dynamic> lengths(n_candidates, segment_length),
                                        angles(n_candidates, segment_length),
                                        dihedrals(n_candidates, segment_length);
            for (int i = 0; i < n_candidates; ++i)
            {
                for (int j = 0; j < segment_length; ++j)
                {
                    lengths(i, j) = sampleFene<T>(
                        this->lj_params.at("eps"), this->lj_params.at("sigma"),
                        this->fene_params.at("K"), this->fene_params.at("R0"),
                        config_.kT, this->rng, this->uniform_dist, 50 
                    );
                    angles(i, j) = sample_angle();
                    dihedrals(i, j) = sampleDihedralHarmonic<T>(
                        this->dihedral_params.at("K"), config_.kT,
                        this->rng, this->uniform_dist
                    );
                }
            }
             
            if (direction == TerminalSegmentEnd::HEAD)    // Move the terminal segment at the head 
            {
                // Generate new candidate atomic positions for the head segment
                for (int i = 0; i < n_candidates; ++i)
                {
                    Matrix<T, Dynamic, 3> segment_i(segment_length, 3); 

                    // Move backwards from atom (segment_length) in the polymer  
                    for (int j = 0; j < segment_length; ++j)
                    {
                        Matrix<T, 3, 1> r1, r2, r3; 
                        if (j == 0)         // Last atom in the segment (closest to the polymer)
                        {
                            r1 = coords.row(segment_length + 2);
                            r2 = coords.row(segment_length + 1); 
                            r3 = coords.row(segment_length);  
                        }
                        else if (j == 1)    // Second-to-last
                        {
                            r1 = coords.row(segment_length + 1); 
                            r2 = coords.row(segment_length); 
                            r3 = segment_i.row(segment_length - 1);  
                        }
                        else if (j == 2)    // Third-to-last
                        {
                            r1 = coords.row(segment_length); 
                            r2 = segment_i.row(segment_length - 1);
                            r3 = segment_i.row(segment_length - 2); 
                        }
                        else 
                        {
                            r1 = segment_i.row(segment_length - 1 - j + 3); 
                            r2 = segment_i.row(segment_length - 1 - j + 2); 
                            r3 = segment_i.row(segment_length - 1 - j + 1);  
                        }
                        int idx = segment_length - 1 - j;
                        segment_i.row(idx) = generateNextAtomDihedral<T>(
                            r1, r2, r3, lengths(i, j), angles(i, j),
                            dihedrals(i, j), this->rng, this->uniform_dist,
                            (dihedrals(i, j) > 0 ? 1 : -1)
                        );
                        moves(i, Eigen::seqN(3 * idx, 3)) = segment_i.row(idx); 
                    }

                    // Get the non-bonded energy difference due to segment replacement
                    energy_diffs(i) = config_.getSegmentReplacementNonbondedEnergyDifference(
                        segment_i, 0, this->lj_params, this->neighbor_threshold
                    ); 
                }
            }
            else        // Move the terminal segment at the tail 
            {
                // Generate new candidate atomic positions for the tail segment
                for (int i = 0; i < n_candidates; ++i)
                {
                    Matrix<T, Dynamic, 3> segment_i(segment_length, 3); 

                    // Move forward from atom (n - segment_length) in the polymer 
                    for (int j = 0; j < segment_length; ++j)
                    {
                        Matrix<T, 3, 1> r1, r2, r3; 
                        if (j == 0)
                        {
                            r1 = coords.row(n - segment_length - 3);
                            r2 = coords.row(n - segment_length - 2); 
                            r3 = coords.row(n - segment_length - 1);  
                        }
                        else if (j == 1)
                        {
                            r1 = coords.row(n - segment_length - 2); 
                            r2 = coords.row(n - segment_length - 1); 
                            r3 = segment_i.row(0); 
                        }
                        else if (j == 2)
                        {
                            r1 = coords.row(n - segment_length - 1); 
                            r2 = segment_i.row(0); 
                            r3 = segment_i.row(1);
                        }
                        else 
                        {
                            r1 = segment_i.row(j - 3);
                            r2 = segment_i.row(j - 2); 
                            r3 = segment_i.row(j - 1); 
                        }
                        segment_i.row(j) = generateNextAtomDihedral<T>(
                            r1, r2, r3, lengths(i, j), angles(i, j),
                            dihedrals(i, j), this->rng, this->uniform_dist,
                            (dihedrals(i, j) > 0 ? 1 : -1)
                        );
                        moves(i, Eigen::seqN(3 * j, 3)) = segment_i.row(j); 
                    }

                    // Get the non-bonded energy difference due to segment replacement
                    energy_diffs(i) = config_.getSegmentReplacementNonbondedEnergyDifference(
                        segment_i, n - segment_length, this->lj_params,
                        this->neighbor_threshold
                    ); 
                }
            }

            return std::make_pair(moves, energy_diffs); 
        }

        /** -------------------------------------------------------------- // 
         *                 MOVE GENERATION: INTERNAL SEGMENT               // 
         *  -------------------------------------------------------------- */
        /**
         * Generate possible internal segment moves from the current 
         * configuration. 
         *
         * @param n_candidates Number of candidate moves to generate. 
         * @param segment_length Segment length.
         * @param segment_idx Index of first atom in the segment. 
         * @param max_iter Maximum number of move generation attempts. 
         * @param init_tangent_stepsize Initial stepsize for perturbation in
         *                              the tangent space. 
         * @param min_tangent_stepsize Minimum stepsize for perturbation in the 
         *                             tangent space.  
         * @param dx Increment for finite difference approximation.
         * @param newton_tol Tolerance for assessing convergence of Newton's
         *                   method. 
         * @param min_newton_stepsize Minimum stepsize for Newton's method.
         * @param max_newton_iter Maximum number of Newton iterations.  
         * @param armijo_const Constant for Armijo condition. Set to 1e-4 by
         *                     default, following Nocedal and Wright (page 33).  
         * @returns Arrays of candidate atomic positions for the new segment
         *          and the corresponding energy differences. 
         */
        std::pair<Matrix<T, Dynamic, Dynamic>, 
                  Matrix<T, Dynamic, 1> > generateInternalSegmentMoves(const int n_candidates, 
                                                                       const int segment_length, 
                                                                       const int segment_idx,
                                                                       const int max_iter,
                                                                       const T init_tangent_stepsize, 
                                                                       const T min_tangent_stepsize,
                                                                       const T dx = 1e-8,  
                                                                       const T newton_tol = 1e-8,
                                                                       const T min_newton_stepsize = 1e-4,
                                                                       const int max_newton_iter = 1000, 
                                                                       const T armijo_const = 1e-4) 
        {
            // Specify the constraint manifold for the internal segment move 
            //
            // This manifold is given by F(x) = 0, where F is a map from
            // R^(3 * segment_length) to R^6
            DynamicVectorValuedFunction<T> F; 
            F = [this, &segment_idx, &segment_length](const Ref<const Matrix<T, Dynamic, 1> >& x) -> Matrix<T, Dynamic, 1>
            {
                Matrix<T, Dynamic, 1> v(6);

                // First three coordinates fix the left endpoint of the
                // internal segment
                Matrix<T, 3, 1> xl = x(Eigen::seqN(0, 3)); 
                v(0) = xl(0) - this->r(segment_idx, 0); 
                v(1) = xl(1) - this->r(segment_idx, 1); 
                v(2) = xl(2) - this->r(segment_idx, 2); 

                // Last three coordinates fix the right endpoint of the 
                // internal segment 
                Matrix<T, 3, 1> xr = x(Eigen::seqN(3 * (segment_length - 1), 3));
                v(3) = xr(0) - this->r(segment_idx + segment_length - 1, 0); 
                v(4) = xr(1) - this->r(segment_idx + segment_length - 1, 1); 
                v(5) = xr(2) - this->r(segment_idx + segment_length - 1, 2); 

                return v;  
            };

            // Calculate bases for the tangent space of the manifold F(x) = 0
            // and its orthogonal complement
            Matrix<T, Dynamic, 1> x0(3 * segment_length); 
            for (int i = 0; i < segment_length; ++i)
            {
                x0(3 * i) = this->r(segment_idx + i, 0); 
                x0(3 * i + 1) = this->r(segment_idx + i, 1); 
                x0(3 * i + 2) = this->r(segment_idx + i, 2); 
            }
            auto bases = getTangentAndOrthogonalSpaceBases<T>(F, x0, dx); 
            Matrix<T, Dynamic, Dynamic> Qt = bases.first; 
            Matrix<T, Dynamic, Dynamic> Qp = bases.second; 

            // Try generating internal moves ... 
            Matrix<T, Dynamic, Dynamic> moves(n_candidates, 3 * segment_length);
            Matrix<T, Dynamic, 1> energy_diffs(n_candidates);  
            int n_success = 0;  
            for (int i = 0; i < max_iter; ++i)
            {
                T tangent_stepsize = init_tangent_stepsize;
                bool found_move = false;  

                // Generate a candidate concerted move
                //
                // First generate a random direction to move along the tangent
                // space (dimension = 3 * segment_length - 6)
                Matrix<T, Dynamic, 1> dir = randomDir<T>(
                    3 * segment_length - 6, this->rng, this->uniform_dist
                ); 
                
                // Try perturbing the input point along the sampled direction,
                // then projecting the perturbed point onto the constraint
                // manifold 
                Matrix<T, Dynamic, 1> x1(3 * segment_length); 
                while (tangent_stepsize > min_tangent_stepsize)
                {
                    auto perturb_result = perturbAndProject<T>(
                        F, x0, Qt, Qp, tangent_stepsize, dir, dx, newton_tol,
                        min_newton_stepsize, max_newton_iter, armijo_const
                    );
                    x1 = perturb_result.first;
                    T residual = perturb_result.second;  
                    
                    // If the projection fails to achieve the desired Newton
                    // tolerance, try perturbing by a smaller increment
                    if (residual > newton_tol)
                    {
                        tangent_stepsize /= 2;
                    }
                    else 
                    { 
                        found_move = true;
                        break; 
                    } 
                }

                // If projection along the given direction was successful, ... 
                if (found_move)
                {
                    // Generate the corresponding atomic coordinates
                    Matrix<T, Dynamic, 3> segment(segment_length, 3); 
                    for (int j = 0; j < segment_length; ++j)
                    {
                        segment(j, 0) = x1(3 * j);
                        segment(j, 1) = x1(3 * j + 1); 
                        segment(j, 2) = x1(3 * j + 2);  
                        moves(n_success, 3 * j) = x1(3 * j); 
                        moves(n_success, 3 * j + 1) = x1(3 * j + 1); 
                        moves(n_success, 3 * j + 2) = x1(3 * j + 2);
                    }

                    // Calculate the corresponding energy difference 
                    energy_diffs(n_success) = this->config.getSegmentReplacementEnergyDifference(
                        segment, segment_idx, this->lj_params,
                        this->neighbor_threshold, this->fene_params, 
                        this->angle_mode, this->angle_params,
                        this->dihedral_params
                    );  
                    n_success++;
                }

                // If we have reached the desired number of candidates, quit 
                if (n_success == n_candidates)
                    break; 
            }

            // Retain only the successful projections 
            moves.conservativeResize(n_success, 3 * segment_length); 
            energy_diffs.conservativeResize(n_success);

            return std::make_pair(moves, energy_diffs);  
        }

        /**
         * Generate possible internal segment moves from the given
         * configuration (which may differ from the current configuration). 
         *
         * @param n_candidates Number of candidate moves to generate. 
         * @param segment_length Segment length.
         * @param segment_idx Index of first atom in the segment.
         * @param coords Input array of atomic coordinates.  
         * @param max_iter Maximum number of move generation attempts. 
         * @param init_tangent_stepsize Initial stepsize for perturbation in
         *                              the tangent space. 
         * @param min_tangent_stepsize Minimum stepsize for perturbation in the 
         *                             tangent space.  
         * @param dx Increment for finite difference approximation.
         * @param newton_tol Tolerance for assessing convergence of Newton's
         *                   method. 
         * @param min_newton_stepsize Minimum stepsize for Newton's method.
         * @param max_newton_iter Maximum number of Newton iterations.  
         * @param armijo_const Constant for Armijo condition. Set to 1e-4 by
         *                     default, following Nocedal and Wright (page 33).  
         * @returns Arrays of candidate atomic positions for the new segment
         *          and the corresponding energy differences. 
         */
        std::pair<Matrix<T, Dynamic, Dynamic>, 
                  Matrix<T, Dynamic, 1> > generateInternalSegmentMoves(const int n_candidates, 
                                                                       const int segment_length, 
                                                                       const int segment_idx,
                                                                       const Ref<const Matrix<T, Dynamic, 3> >& coords, 
                                                                       const int max_iter,
                                                                       const T init_tangent_stepsize, 
                                                                       const T min_tangent_stepsize,
                                                                       const T dx = 1e-8,  
                                                                       const T newton_tol = 1e-8,
                                                                       const T min_newton_stepsize = 1e-4, 
                                                                       const int max_newton_iter = 1000,
                                                                       const T armijo_const = 1e-4)
        {
            // Generate new configuration with the given coordinates 
            PolymerConfiguration<T> config_(
                coords, this->config.getUnits(), this->config.getTemp()
            );    

            // Specify the constraint manifold for the internal segment move 
            //
            // This manifold is given by F(x) = 0, where F is a map from
            // R^(3 * segment_length) to R^6
            DynamicVectorValuedFunction<T> F; 
            F = [&coords, &segment_idx, &segment_length](const Ref<const Matrix<T, Dynamic, 1> >& x) -> Matrix<T, Dynamic, 1>
            {
                Matrix<T, Dynamic, 1> v(6);

                // First three coordinates fix the left endpoint of the
                // internal segment
                Matrix<T, 3, 1> xl = x(Eigen::seqN(0, 3)); 
                v(0) = xl(0) - coords(segment_idx, 0); 
                v(1) = xl(1) - coords(segment_idx, 1); 
                v(2) = xl(2) - coords(segment_idx, 2); 

                // Last three coordinates fix the right endpoint of the 
                // internal segment 
                Matrix<T, 3, 1> xr = x(Eigen::seqN(3 * (segment_length - 1), 3));
                v(3) = xr(0) - coords(segment_idx + segment_length - 1, 0); 
                v(4) = xr(1) - coords(segment_idx + segment_length - 1, 1); 
                v(5) = xr(2) - coords(segment_idx + segment_length - 1, 2); 

                return v;  
            };

            // Calculate bases for the tangent space of the manifold F(x) = 0
            // and its orthogonal complement
            Matrix<T, Dynamic, 1> x0(3 * segment_length); 
            for (int i = 0; i < segment_length; ++i)
            {
                x0(3 * i) = coords(segment_idx + i, 0); 
                x0(3 * i + 1) = coords(segment_idx + i, 1); 
                x0(3 * i + 2) = coords(segment_idx + i, 2); 
            }
            auto bases = getTangentAndOrthogonalSpaceBases<T>(F, x0, dx); 
            Matrix<T, Dynamic, Dynamic> Qt = bases.first; 
            Matrix<T, Dynamic, Dynamic> Qp = bases.second; 

            // Try generating internal moves ... 
            Matrix<T, Dynamic, Dynamic> moves(n_candidates, 3 * segment_length);
            Matrix<T, Dynamic, 1> energy_diffs(n_candidates);  
            int n_success = 0;  
            for (int i = 0; i < max_iter; ++i)
            {
                T tangent_stepsize = init_tangent_stepsize;
                bool found_move = false;  

                // Generate a candidate concerted move
                //
                // First generate a random direction to move along the tangent
                // space (dimension = 3 * segment_length - 6)
                Matrix<T, Dynamic, 1> dir = randomDir<T>(
                    3 * segment_length - 6, this->rng, this->uniform_dist
                ); 
                
                // Try perturbing the input point along the sampled direction,
                // then projecting the perturbed point onto the constraint
                // manifold 
                Matrix<T, Dynamic, 1> x1(3 * segment_length); 
                while (tangent_stepsize > min_tangent_stepsize)
                {
                    auto perturb_result = perturbAndProject<T>(
                        F, x0, Qt, Qp, tangent_stepsize, dir, dx, newton_tol,
                        min_newton_stepsize, max_newton_iter, armijo_const
                    );
                    x1 = perturb_result.first;
                    T residual = perturb_result.second;  
                    
                    // If the projection fails to achieve the desired Newton
                    // tolerance, try perturbing by a smaller increment
                    if (residual > newton_tol)
                    {
                        tangent_stepsize /= 2;
                    }
                    else 
                    { 
                        found_move = true;
                        break; 
                    } 
                }

                // If projection along the given direction was successful, ... 
                if (found_move)
                {
                    // Generate the corresponding atomic coordinates
                    Matrix<T, Dynamic, 3> segment(segment_length, 3); 
                    for (int j = 0; j < segment_length; ++j)
                    {
                        segment(j, 0) = x1(3 * j);
                        segment(j, 1) = x1(3 * j + 1); 
                        segment(j, 2) = x1(3 * j + 2);  
                        moves(n_success, 3 * j) = x1(3 * j); 
                        moves(n_success, 3 * j + 1) = x1(3 * j + 1); 
                        moves(n_success, 3 * j + 2) = x1(3 * j + 2);
                    }

                    // Calculate the corresponding energy difference 
                    energy_diffs(n_success) = config_.getSegmentReplacementEnergyDifference(
                        segment, segment_idx, this->lj_params,
                        this->neighbor_threshold, this->fene_params, 
                        this->angle_mode, this->angle_params,
                        this->dihedral_params
                    );  
                    n_success++;
                }

                // If we have reached the desired number of candidates, quit 
                if (n_success == n_candidates)
                    break; 
            }

            // Retain only the successful projections 
            moves.conservativeResize(n_success, 3 * segment_length); 
            energy_diffs.conservativeResize(n_success);

            return std::make_pair(moves, energy_diffs);  
        }

        /** -------------------------------------------------------------- // 
         *                 CONFIGURATIONAL-BIAS MONTE CARLO                // 
         *  -------------------------------------------------------------- */
        /**
         * @param n_candidates Number of candidate moves to generate. 
         * @param move_type Move type. 
         * @param segment_length Segment length for terminal/internal segment
         *                       moves. Fixed to 1 for reptation. 
         * @param move_params Additional parameters for internal segment moves.
         * @returns The forward and reverse candidate moves, the index of the
         *          chosen (forward) move, its Metropolis acceptance probability,
         *          whether the move was taken, and other identifying information
         *          regarding the move. 
         */
        std::tuple<Matrix<T, Dynamic, Dynamic>, 
                   Matrix<T, Dynamic, Dynamic>, 
                   int, 
                   T, 
                   bool, 
                   std::unordered_map<std::string, T> > moveOnce(const int n_candidates,
                                                                 const CBMCMoveType move_type,
                                                                 int segment_length,
                                                                 const std::unordered_map<std::string, T>& move_params)
        {
            const int n = this->length;

            // Fix segment length 
            if (move_type == CBMCMoveType::REPTATION)
                segment_length = 1; 

            // Specify a reptation direction if desired 
            ReptationDirection rept_dir = ReptationDirection::HEAD;  
            if (move_type == CBMCMoveType::REPTATION)
            {
                const T p = this->uniform_dist(this->rng); 
                rept_dir = (p < 0.5 ? ReptationDirection::HEAD : ReptationDirection::TAIL);
            }

            // Specify a terminal segment to move if desired 
            TerminalSegmentEnd terminal_end = TerminalSegmentEnd::HEAD;  
            if (move_type == CBMCMoveType::TERMINAL_SEGMENT)
            {
                const T p = this->uniform_dist(this->rng); 
                terminal_end = (p < 0.5 ? TerminalSegmentEnd::HEAD : TerminalSegmentEnd::TAIL); 
            }

            // Specify an internal segment to move if desired 
            int segment_idx = 1; 
            if (move_type == CBMCMoveType::INTERNAL_SEGMENT)
            {
                boost::random::uniform_int_distribution<>
                    randint_dist(1, this->length - segment_length - 1); 
                segment_idx = randint_dist(this->rng);
            }

            // Generate forward moves ...
            Matrix<T, Dynamic, Dynamic> forward_moves, reverse_moves;  
            Matrix<T, Dynamic, 1> forward_diffs, reverse_diffs;  
            if (move_type == CBMCMoveType::REPTATION)
            {
                auto forward_result = this->generateReptationMoves(
                    rept_dir, n_candidates
                );
                forward_moves = forward_result.first;
                forward_diffs = forward_result.second; 
            }
            else if (move_type == CBMCMoveType::TERMINAL_SEGMENT)
            {
                auto forward_result = this->generateTerminalSegmentMoves(
                    segment_length, terminal_end, n_candidates
                );
                forward_moves = forward_result.first;
                forward_diffs = forward_result.second;  
            }
            else    // move_type == CBMCMoveType::INTERNAL_SEGMENT
            {
                // Specify required parameters
                const T init_tangent_stepsize = move_params.at("init_tangent_stepsize"); 
                const T min_tangent_stepsize = move_params.at("min_tangent_stepsize");

                // Specify optional parameters
                T dx, newton_tol, min_newton_stepsize, armijo_const; 
                int max_iter, max_newton_iter; 
                try
                {
                    max_iter = static_cast<int>(move_params.at("max_iter")); 
                } 
                catch (const std::out_of_range& e)
                {
                    max_iter = 2 * n_candidates; 
                }
                try
                {
                    dx = move_params.at("dx");
                }
                catch (const std::out_of_range& e)
                {
                    dx = 1e-8;
                } 
                try
                {
                    newton_tol = move_params.at("newton_tol");
                } 
                catch (const std::out_of_range& e)
                {
                    newton_tol = 1e-8;
                } 
                try
                {
                    min_newton_stepsize = move_params.at("min_newton_stepsize");
                } 
                catch (const std::out_of_range& e)
                {
                    min_newton_stepsize = 1e-4;
                }
                try
                {
                    max_newton_iter = static_cast<int>(move_params.at("max_newton_iter"));
                } 
                catch (const std::out_of_range& e)
                {
                    max_newton_iter = 1000;
                } 
                try
                {
                    armijo_const = move_params.at("armijo_const");
                } 
                catch (const std::out_of_range& e)
                {
                    armijo_const = 1e-4;
                }

                // Generate internal segment moves  
                auto forward_result = this->generateInternalSegmentMoves(
                    n_candidates, segment_length, segment_idx, max_iter,
                    init_tangent_stepsize, min_tangent_stepsize, dx, newton_tol,
                    min_newton_stepsize, max_newton_iter, armijo_const 
                );

                // If the desired number of candidate moves was not generated, 
                // then simply remain at the current configuration
                if (forward_result.first.rows() < n_candidates)
                {
                    forward_moves.resize(1, 3 * segment_length); 
                    reverse_moves.resize(1, 3 * segment_length);
                    for (int i = 0; i < segment_length; ++i)
                    {
                        forward_moves(0, Eigen::seqN(3 * i, 3)) = this->r.row(segment_idx + i);
                        reverse_moves(0, Eigen::seqN(3 * i, 3)) = this->r.row(segment_idx + i);  
                    }
                    std::unordered_map<std::string, T> move_info; 
                    move_info["segment_idx"] = segment_idx;
                    move_info["proposed_new_move"] = false; 
                    return std::make_tuple(
                        forward_moves, reverse_moves, 0, 1.0, true, move_info
                    ); 
                }

                // Otherwise, keep track of the candidate moves 
                forward_moves = forward_result.first;
                forward_diffs = forward_result.second;  
            }

            // Calculate the forward Rosenbluth factor
            Matrix<T, Dynamic, 1> forward_weights
                = ((-forward_diffs).array() / this->config.kT).exp().matrix(); 
            T forward_rosenbluth = forward_weights.sum();

            // Choose one move out of the candidates 
            std::vector<T> probs; 
            for (int i = 0; i < n_candidates; ++i)
                probs.push_back(forward_weights(i) / forward_rosenbluth); 
            boost::random::discrete_distribution<> dist(probs);  
            int move_idx = dist(this->rng);
            Matrix<T, Dynamic, 3> move(segment_length, 3); 
            for (int i = 0; i < segment_length; ++i)
            {
                move(i, 0) = forward_moves(move_idx, 3 * i); 
                move(i, 1) = forward_moves(move_idx, 3 * i + 1); 
                move(i, 2) = forward_moves(move_idx, 3 * i + 2); 
            }

            // Generate a copy of the current polymer configuration and swap
            // in the chosen candidate move 
            PolymerConfiguration<T> config_chosen(this->config);
            if (move_type == CBMCMoveType::REPTATION)
            {
                if (rept_dir == ReptationDirection::HEAD)
                {
                    Matrix<T, 3, 1> r_new; 
                    r_new << move(0, 0), move(0, 1), move(0, 2); 
                    config_chosen.reptateTowardsHead(r_new); 
                } 
                else
                {
                    Matrix<T, 3, 1> r_new; 
                    r_new << move(0, 0), move(0, 1), move(0, 2); 
                    config_chosen.reptateTowardsTail(r_new); 
                }
            }
            else if (move_type == CBMCMoveType::TERMINAL_SEGMENT)
            {
                if (terminal_end == TerminalSegmentEnd::HEAD)
                    config_chosen.replaceSegment(move, 0); 
                else 
                    config_chosen.replaceSegment(move, n - segment_length);
            }
            else   // move_type == CBMCMoveType::INTERNAL_SEGMENT
            {
                config_chosen.replaceSegment(move, segment_idx); 
            }
            Matrix<T, Dynamic, 3> coords_chosen = config_chosen.getSegment(0, n);

            // Generate reverse moves from the chosen configuration
            if (move_type == CBMCMoveType::REPTATION)
            {
                // Identify the reverse direction 
                ReptationDirection reverse_dir; 
                if (rept_dir == ReptationDirection::HEAD)
                    reverse_dir = ReptationDirection::TAIL; 
                else 
                    reverse_dir = ReptationDirection::HEAD;

                // Generate reverse moves and energy differences  
                auto reverse_result = this->generateReptationMoves(
                    reverse_dir, n_candidates, coords_chosen 
                );
                reverse_moves = reverse_result.first;
                reverse_diffs = reverse_result.second;
            }
            else if (move_type == CBMCMoveType::TERMINAL_SEGMENT)
            {
                // Generate reverse moves and energy differences
                auto reverse_result = this->generateTerminalSegmentMoves(
                    segment_length, terminal_end, n_candidates, coords_chosen
                );
                reverse_moves = reverse_result.first;
                reverse_diffs = reverse_result.second;  
            }
            else    // move_type == CBMCMoveType::INTERNAL_SEGMENT
            {
                // Specify required parameters
                const T init_tangent_stepsize = move_params.at("init_tangent_stepsize"); 
                const T min_tangent_stepsize = move_params.at("min_tangent_stepsize");

                // Specify optional parameters
                T dx, newton_tol, min_newton_stepsize, armijo_const; 
                int max_iter, max_newton_iter; 
                try
                {
                    max_iter = static_cast<int>(move_params.at("max_iter")); 
                } 
                catch (const std::out_of_range& e)
                {
                    max_iter = 2 * n_candidates; 
                }
                try
                {
                    dx = move_params.at("dx");
                }
                catch (const std::out_of_range& e)
                {
                    dx = 1e-8;
                } 
                try
                {
                    newton_tol = move_params.at("newton_tol");
                } 
                catch (const std::out_of_range& e)
                {
                    newton_tol = 1e-8;
                } 
                try
                {
                    min_newton_stepsize = move_params.at("min_newton_stepsize");
                } 
                catch (const std::out_of_range& e)
                {
                    min_newton_stepsize = 1e-4;
                }
                try
                {
                    max_newton_iter = static_cast<int>(move_params.at("max_newton_iter"));
                } 
                catch (const std::out_of_range& e)
                {
                    max_newton_iter = 1000;
                } 
                try
                {
                    armijo_const = move_params.at("armijo_const");
                } 
                catch (const std::out_of_range& e)
                {
                    armijo_const = 1e-4;
                }

                // Generate internal segment moves  
                auto reverse_result = this->generateInternalSegmentMoves(
                    n_candidates, segment_length, segment_idx, coords_chosen, 
                    max_iter, init_tangent_stepsize, min_tangent_stepsize,
                    dx, newton_tol, min_newton_stepsize, max_newton_iter,
                    armijo_const 
                );

                // If the desired number of candidate moves was not generated, 
                // then simply remain at the current configuration
                if (reverse_result.first.rows() < n_candidates)
                {
                    forward_moves.resize(1, 3 * segment_length); 
                    reverse_moves.resize(1, 3 * segment_length);
                    for (int i = 0; i < segment_length; ++i)
                    {
                        forward_moves(0, Eigen::seqN(3 * i, 3)) = this->r.row(segment_idx + i);
                        reverse_moves(0, Eigen::seqN(3 * i, 3)) = this->r.row(segment_idx + i);  
                    }
                    std::unordered_map<std::string, T> move_info; 
                    move_info["segment_idx"] = segment_idx;
                    move_info["proposed_new_move"] = false; 
                    return std::make_tuple(
                        forward_moves, reverse_moves, 0, 1.0, true, move_info
                    ); 
                }

                // Otherwise, keep track of the candidate moves 
                reverse_moves = reverse_result.first;
                reverse_diffs = reverse_result.second;
            }

            // Add in the original configuration as one of the reverse moves
            if (move_type == CBMCMoveType::REPTATION)
            {
                if (rept_dir == ReptationDirection::HEAD)
                { 
                    reverse_moves.row(move_idx) = this->r.row(n - 1); 
                    reverse_diffs(move_idx)
                        = config_chosen.getReptationNonbondedEnergyDifference(
                            ReptationDirection::TAIL, this->r.row(n - 1),
                            this->lj_params, this->neighbor_threshold
                        );
                }
                else 
                {
                    reverse_moves.row(move_idx) = this->r.row(0); 
                    reverse_diffs(move_idx)
                        = config_chosen.getReptationNonbondedEnergyDifference(
                            ReptationDirection::HEAD, this->r.row(0),
                            this->lj_params, this->neighbor_threshold
                        ); 
                }
            }
            else if (move_type == CBMCMoveType::TERMINAL_SEGMENT)
            {
                if (terminal_end == TerminalSegmentEnd::HEAD)
                {
                    // Extract the original configuration along the head segment 
                    Matrix<T, Dynamic, 3> segment = this->r(
                        Eigen::seqN(0, segment_length), Eigen::all
                    ); 
                    for (int i = 0; i < segment_length; ++i)
                    {
                        reverse_moves(move_idx, 3 * i) = segment(i, 0); 
                        reverse_moves(move_idx, 3 * i + 1) = segment(i, 1); 
                        reverse_moves(move_idx, 3 * i + 2) = segment(i, 2); 
                    }
                    reverse_diffs(move_idx)
                        = config_chosen.getSegmentReplacementNonbondedEnergyDifference(
                            segment, 0, this->lj_params, this->neighbor_threshold 
                        );
                }
                else 
                {
                    // Extract the original configuration along the tail segment
                    Matrix<T, Dynamic, 3> segment = this->r(
                        Eigen::seqN(n - segment_length, segment_length), Eigen::all
                    );
                    for (int i = 0; i < segment_length; ++i)
                    {
                        reverse_moves(move_idx, 3 * i) = segment(i, 0); 
                        reverse_moves(move_idx, 3 * i + 1) = segment(i, 1); 
                        reverse_moves(move_idx, 3 * i + 2) = segment(i, 2); 
                    }
                    reverse_diffs(move_idx)
                        = config_chosen.getSegmentReplacementNonbondedEnergyDifference(
                            segment, n - segment_length, this->lj_params,
                            this->neighbor_threshold
                        ); 
                }
            }
            else    // move_type == CBMCMoveType::INTERNAL_SEGMENT
            {
                // Extract the original configuration along the internal segment
                Matrix<T, Dynamic, 3> segment = this->r(
                    Eigen::seqN(segment_idx, segment_length), Eigen::all
                ); 
                for (int i = 0; i < segment_length; ++i)
                {
                    reverse_moves(move_idx, 3 * i) = segment(i, 0); 
                    reverse_moves(move_idx, 3 * i + 1) = segment(i, 1); 
                    reverse_moves(move_idx, 3 * i + 2) = segment(i, 2); 
                }
                reverse_diffs(move_idx)
                    = config_chosen.getSegmentReplacementNonbondedEnergyDifference(
                        segment, segment_idx, this->lj_params,
                        this->neighbor_threshold 
                    );
            }

            // Calculate the reverse Rosenbluth factor
            Matrix<T, Dynamic, 1> reverse_weights
                = ((-reverse_diffs).array() / config_chosen.kT).exp().matrix(); 
            T reverse_rosenbluth = reverse_weights.sum();

            // Calculate the Metropolis acceptance probability
            T prob_accept = min(1.0, forward_rosenbluth / reverse_rosenbluth);

            // Change the polymer configuration according to that probability
            T r = this->uniform_dist(this->rng); 
            if (r < prob_accept)
            {
                if (move_type == CBMCMoveType::REPTATION)
                {
                    // Reptate towards the head 
                    if (rept_dir == ReptationDirection::HEAD)
                    {
                        Matrix<T, 3, 1> r_new; 
                        r_new << move(0, 0), move(0, 1), move(0, 2); 
                        this->config.reptateTowardsHead(r_new); 
                    } 
                    else    // Reptate towards the tail 
                    {
                        Matrix<T, 3, 1> r_new; 
                        r_new << move(0, 0), move(0, 1), move(0, 2); 
                        this->config.reptateTowardsTail(r_new); 
                    }
                }
                else if (move_type == CBMCMoveType::TERMINAL_SEGMENT)
                {
                    // Move the terminal segment at the head 
                    if (terminal_end == TerminalSegmentEnd::HEAD)
                        this->config.replaceSegment(move, 0); 
                    else    // Move the terminal segment at the tail 
                        this->config.replaceSegment(move, n - segment_length);
                }
                if (move_type == CBMCMoveType::INTERNAL_SEGMENT)
                {
                    // Move the internal segment
                    this->config.replaceSegment(move, segment_idx);  
                }
            }

            // Update stored atomic coordinates 
            this->updateCoords(); 

            // Return the forward and reverse moves, the chosen move, the 
            // acceptance probability, whether the move was taken, and all 
            // additional information about the move (depending on move type)
            std::unordered_map<std::string, T> move_info; 
            if (move_type == CBMCMoveType::REPTATION)
            {
                move_info["direction"] = (rept_dir == ReptationDirection::HEAD ? 0 : 1);
            }
            else if (move_type == CBMCMoveType::TERMINAL_SEGMENT)
            {
                move_info["terminal_end"] = (terminal_end == TerminalSegmentEnd::HEAD ? 0 : 1);
            } 
            else
            { 
                move_info["segment_idx"] = segment_idx;
                move_info["proposed_new_move"] = true; 
            } 
            return std::make_tuple(
                forward_moves, reverse_moves, move_idx, prob_accept,
                (r < prob_accept), move_info
            ); 
        }

        /**
         * Run configurational-bias Monte Carlo sampling. 
         *
         * This sampling procedure chooses, in each iteration, one of the
         * three moves (reptation, terminal segment move, internal segment
         * move) probabilistically, according to the given array of move 
         * probabilities. 
         *
         * The returned array contains a representative sub-sample of the
         * sampled configurations.   
         *
         * @param n_candidates Number of candidate moves to generate. 
         * @param internal_move_params Additional parameters for generating
         *                             internal segment moves.  
         * @param move_probs Array of probabilities for choosing each move type.
         * @param terminal_segment_length Segment length for terminal segment
         *                                moves. 
         * @param internal_segment_length Segment length for internal segment 
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
        Matrix<T, Dynamic, Dynamic> run(const int n_candidates, 
                                        std::unordered_map<std::string, T>& internal_move_params, 
                                        const Ref<const Matrix<T, 3, 1> >& move_probs,
                                        const int terminal_segment_length, 
                                        const int internal_segment_length,
                                        const int max_iter, const int n_burnin,
                                        const int mod_collect, int mod_write, 
                                        const int max_stall,
                                        std::ofstream& outfile,  
                                        const bool verbose = false)
        {
            // Keep track of time for intermittent output to stdout
            auto t_curr = std::chrono::high_resolution_clock::now();  

            // Identify how many configurations will be collected throughout
            // the sampling
            int n_collect = (max_iter - n_burnin) / mod; 
            Matrix<T, Dynamic, Dynamic> ensemble_coords(n_collect, 3 * this->length);

            // Ensure that mod_write is some multiple (>= 10) of mod_collect
            mod_write = max(
                10 * mod_collect, 
                mod_collect * static_cast<int>(ceil(mod_write / mod_collect))
            );  

            // Tabulate average acceptance probabilities for each move type 
            Matrix<T, 3, 1> accept_probs = Matrix<T, 3, 1>::Zero();

            // Tabulate probability of null moves for internal segment moves 
            T internal_null_move_prob = 0; 

            // Tabulate fraction of non-null moves 
            T frac_nonnull = 0;  

            // Run sampling procedure ... 
            int curr_idx = 0; 
            int collect_idx = 0;
            int n_stall = 0; 
            int last_written_idx = -1;  
            while (collect_idx < n_collect)
            {
                // Sample a move type 
                T r = uniform_dist(rng);
                CBMCMoveType move_type;         
                if (r < move_probs(0))
                    move_type = CBMCMoveType::REPTATION; 
                else if (r < move_probs(0) + move_probs(1))
                    move_type = CBMCMoveType::TERMINAL_SEGMENT; 
                else 
                    move_type = CBMCMoveType::INTERNAL_SEGMENT; 

                // Generate and accept/reject a corresponding move 
                int segment_length = 0; 
                if (move_type == CBMCMoveType::TERMINAL_SEGMENT)
                    segment_length = terminal_segment_length; 
                else if (move_type == CBMCMoveType::INTERNAL_SEGMENT)
                    segment_length = internal_segment_length;
                auto result = this->moveOnce(
                    n_candidates, move_type, segment_length, internal_move_params
                );

                // Update average acceptance probabilities 
                T prob_accept = std::get<3>(result);
                if (move_type == CBMCMoveType::REPTATION)
                    accept_probs(0) += ((prob_accept - accept_probs(0)) / (curr_idx + 1));
                else if (move_type == CBMCMoveType::TERMINAL_SEGMENT)
                    accept_probs(1) += ((prob_accept - accept_probs(1)) / (curr_idx + 1));
                else 
                    accept_probs(2) += ((prob_accept - accept_probs(2)) / (curr_idx + 1)); 

                // Update probability of null internal segment moves
                auto move_info = std::get<5>(result);
                if (move_type == CBMCMoveType::INTERNAL_SEGMENT)
                {
                    double null_move = (move_info["proposed_new_move"] > 0 ? 0.0 : 1.0); 
                    internal_null_move_prob += (
                        (null_move - internal_null_move_prob) / (curr_idx + 1)
                    );
                }

                // Update total fraction of non-null moves
                bool accepted_move = std::get<4>(result); 
                bool made_null_move = (
                    !accepted_move || (
                        move_type == CBMCMoveType::INTERNAL_SEGMENT &&
                        move_info["proposed_new_move"] == 0
                    )
                );
                frac_nonnull += (((!made_null_move) - frac_nonnull) / (curr_idx + 1));  

                // Update number of consecutive stalling iterations 
                if (made_null_move)
                    n_stall++;
                else 
                    n_stall = 0;

                // If we have exceeded the number of consecutive iterations 
                // in which the sampling has stalled, return the current 
                // sample 
                if (n_stall > max_stall)
                {
                    std::cout << "[WARN] Sampling has stalled due to too many "
                              << "consecutive rejections" << std::endl;
                    ensemble_coords.conservativeResize(collect_idx, 3 * this->length);  
                    return ensemble_coords; 
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
                              << ", null internal move prob = "
                              << internal_null_move_prob
                              << ", fraction of non-null moves = "
                              << frac_nonnull << std::endl;
                    t_curr = t_next; 
                } 

                // Decide whether to collect this configuration 
                if (curr_idx >= n_burnin && (curr_idx - n_burnin) % mod_collect == 0)
                {
                    for (int j = 0; j < this->length; ++j)
                    {
                        ensemble_coords(collect_idx, 3 * j) = this->r(j, 0); 
                        ensemble_coords(collect_idx, 3 * j + 1) = this->r(j, 1); 
                        ensemble_coords(collect_idx, 3 * j + 2) = this->r(j, 2); 
                    }
                    collect_idx++;
                }

                // Decide whether to write the configurations accumulated
                // thus far
                if (curr_idx >= n_burnin && (curr_idx - n_burnin) % mod_write == 0)
                {
                    for (int i = last_written_idx + 1; i < collect_idx; ++i)
                    {
                        // Calculate configuration energy and radius of gyration
                        T energy = config.getTotalEnergy(
                            this->lj_params, this->neighbor_threshold, 
                            this->fene_params, this->angle_mode,
                            this->angle_params, this->dihedral_params
                        );
                        T radius = config.radiusOfGyration(); 

                        // Write coordinates, energy, and radius of gyration to file  
                        outfile << "CONFIG\t" << i << std::endl
                                << "# ENERGY\t" << energy << std::endl 
                                << "# RADIUS_OF_GYRATION\t" << radius << std::endl; 
                        for (int j = 0; j < length; ++j)
                        {
                            outfile << ensemble_coords(i, 3 * j) << '\t'
                                    << ensemble_coords(i, 3 * j + 1) << '\t' 
                                    << ensemble_coords(i, 3 * j + 2) << std::endl; 
                        }  
                    } 
                    last_written_idx = collect_idx - 1;
                } 
                curr_idx++; 
            }

            // Write remaining configurations to file 
            for (int i = last_written_idx + 1; i < collect_idx; ++i)
            {
                // Calculate configuration energy and radius of gyration
                T energy = config.getTotalEnergy(
                    this->lj_params, this->neighbor_threshold, 
                    this->fene_params, this->angle_mode,
                    this->angle_params, this->dihedral_params
                );
                T radius = config.radiusOfGyration(); 

                // Write coordinates, energy, and radius of gyration to file  
                outfile << "CONFIG\t" << i << std::endl
                        << "# ENERGY\t" << energy << std::endl 
                        << "# RADIUS_OF_GYRATION\t" << radius << std::endl; 
                for (int j = 0; j < length; ++j)
                {
                    outfile << ensemble_coords(i, 3 * j) << '\t'
                            << ensemble_coords(i, 3 * j + 1) << '\t' 
                            << ensemble_coords(i, 3 * j + 2) << std::endl; 
                }  
            } 

            return ensemble_coords; 
        }
};

#endif 
