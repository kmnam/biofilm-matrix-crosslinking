/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     3/24/2026
 */

#ifndef POLYMER_UTILS_HPP 
#define POLYMER_UTILS_HPP 

#include <cmath>
#include <fstream>
#include <string>
#include <limits>
#include <unordered_map>
#include <functional>
#include <Eigen/Dense>
#include <boost/json/src.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/random.hpp>

using std::abs;
using boost::multiprecision::abs; 
using std::log; 
using boost::multiprecision::log; 
using std::pow; 
using boost::multiprecision::pow;
using std::exp; 
using boost::multiprecision::exp; 
using std::sqrt;
using boost::multiprecision::sqrt;
using std::min; 
using boost::multiprecision::min; 
using std::sin; 
using boost::multiprecision::sin; 
using std::cos;
using boost::multiprecision::cos;
using std::acos; 
using boost::multiprecision::acos; 
using std::atan2; 
using boost::multiprecision::atan2;
using std::fmod; 
using boost::multiprecision::fmod; 

using namespace Eigen;

/**
 * Sample a value from the standard normal distribution with the Box-Muller
 * method. 
 *
 * @param rng Random number generator. 
 * @param uniform_dist Pre-defined instance of standard uniform distribution. 
 * @returns Sampled value. 
 */
template <typename T>
T standardNormal(boost::random::mt19937& rng, boost::random::uniform_01<>& uniform_dist)
{
    T u = static_cast<T>(uniform_dist(rng)); 
    T v = static_cast<T>(uniform_dist(rng));
    T c = sqrt(-2 * log(u));
    return c * cos(boost::math::constants::two_pi<T>() * v);  
}

/**
 * Generate a random unit vector. 
 *
 * @param rng Random number generator. 
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @returns Sampled unit vector.  
 */
template <typename T, size_t Dim = 3>
Matrix<T, Dim, 1> randomDir(boost::random::mt19937& rng,
                            boost::random::uniform_01<>& uniform_dist)
{
    Matrix<T, Dim, 1> v;
    for (int i = 0; i < Dim; ++i) 
        v(i) = standardNormal<T>(rng, uniform_dist);
    return v / v.norm(); 
}

/**
 * A version of `randomDir()` that returns a dynamically sized vector of the
 * given size.
 *
 * @param rng Random number generator. 
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @returns Sampled unit vector.  
 */
template <typename T>
Matrix<T, Dynamic, 1> randomDir(const int dim, boost::random::mt19937& rng, 
                                boost::random::uniform_01<>& uniform_dist)
{
    Matrix<T, Dynamic, 1> v(dim);
    for (int i = 0; i < dim; ++i) 
        v(i) = standardNormal<T>(rng, uniform_dist);
    return v / v.norm(); 
}

/**
 * A safe version of acos() that accounts for (slightly) out-of-range input
 * values. 
 *
 * @param x Input value. 
 * @returns Arccosine of input value.  
 */
template <typename T>
T acosSafe(const T& x)
{
    if (x >= 1)
        return 0; 
    else if (x <= -1)
        return boost::math::constants::pi<T>(); 
    else 
        return acos(x);  
}

/**
 * Generate the rotation matrix corresponding to rotation about the given 
 * axis by the given angle. 
 *
 * @param u Rotation axis. 
 * @param theta Angle. 
 * @returns Rotation matrix.  
 */
template <typename T>
Matrix<T, 3, 3> getRotation(const Ref<const Matrix<T, 3, 1> >& u, const T theta)
{
    // Use Rodrigues' formula for axis-angle rotation
    T c1 = cos(theta); 
    T c2 = 1 - c1; 
    T s = sin(theta); 
    Matrix<T, 3, 3> rot;  
    rot << c1 + u(0) * u(0) * c2,           // First row
           u(0) * u(1) * c2 - u(2) * s,
           u(0) * u(2) * c2 + u(1) * s,
           u(1) * u(0) * c2 + u(2) * s,     // Second row
           c1 + u(1) * u(1) * c2,
           u(1) * u(2) * c2 - u(0) * s,
           u(2) * u(0) * c2 - u(1) * s,    // Third row 
           u(2) * u(1) * c2 + u(0) * s,
           c1 + u(2) * u(2) * c2;

    return rot;  
}

/**
 * Sample a value from the von Mises distribution with the given mean and
 * concentration parameter.
 *
 * This is an implementation of Best & Fisher's algorithm. 
 *
 * @param mu Mean. 
 * @param kappa Concentration parameter.
 * @param rng Random number generator.
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @returns A sampled value from the von Mises distribution.
 */
template <typename T>
T vonMises(const T mu, const T kappa, boost::random::mt19937& rng,
           boost::random::uniform_01<>& uniform_dist)
{
    T tau = 1 + sqrt(1 + 4 * kappa * kappa);
    T rho = (tau - sqrt(2 * tau)) / (2 * kappa);
    T r = (1 + rho * rho) / (2 * rho);
    T z, f, c; 
    bool reject = true; 
    while (reject)
    {
        T u1 = uniform_dist(rng); 
        T u2 = uniform_dist(rng);
        z = cos(boost::math::constants::pi<T>() * u1);
        f = (1 + r * z) / (r + z); 
        c = kappa * (r - f); 
        if (c * (2 - c) - u2 > 0)
            reject = false;
        else if (log(c / u2) + 1 - c >= 0)
            reject = false;
    }

    // Get final angle and ensure that its value is between [0, 2*\pi)
    T u3 = uniform_dist(rng);
    T theta;
    if (u3 > 0.5)
        theta = fmod(acosSafe<T>(f) + mu, boost::math::constants::two_pi<T>());
    else if (u3 < 0.5)
        theta = fmod(-acosSafe<T>(f) + mu, boost::math::constants::two_pi<T>());
    else 
        theta = mu;

    // Then change the domain to [-\pi, \pi)
    if (theta > boost::math::constants::pi<T>())
        return theta - boost::math::constants::two_pi<T>(); 
    else 
        return theta;  
}

/**
 * Generate an orthonormal basis of R^3 that contains the given vector `u`. 
 *
 * @param u Input vector. 
 * @param rng Random number generator. 
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @returns Remaining two vectors in the basis. 
 */
template <typename T>
Matrix<T, 2, 3> generateOrthonormalBasis3D(const Ref<const Matrix<T, 3, 1> >& u, 
                                           boost::random::mt19937& rng,
                                           boost::random::uniform_01<>& uniform_dist)
{
    // First sample a random unit vector ... 
    Matrix<T, 3, 1> v = randomDir<T>(rng, uniform_dist); 

    // ... then project it onto u and normalize ... 
    Matrix<T, 3, 1> proj = v.dot(u) * u;
    v -= proj; 
    T v_norm = v.norm(); 
    v /= v_norm;

    // ... then get the cross product of u and v and normalize 
    Matrix<T, 3, 1> cross = u.cross(v); 
    Matrix<T, 3, 1> w = cross / cross.norm();

    // Return the two vectors 
    Matrix<T, 2, 3> vw; 
    vw.row(0) = v.transpose(); 
    vw.row(1) = w.transpose(); 
    return vw; 
}

/**
 * Get the dihedral angle along the given four-atom segment. 
 *
 * @param r1 First atom. 
 * @param r2 Second atom. 
 * @param r3 Third atom. 
 * @param r4 Fourth atom. 
 * @returns Dihedral angle. 
 */
template <typename T>
T getDihedral(const Ref<const Matrix<T, 3, 1> >& r1, const Ref<const Matrix<T, 3, 1> >& r2, 
              const Ref<const Matrix<T, 3, 1> >& r3, const Ref<const Matrix<T, 3, 1> >& r4)
{
    Matrix<T, 3, 1> u1 = r2 - r1; 
    Matrix<T, 3, 1> u2 = r3 - r2; 
    Matrix<T, 3, 1> u3 = r4 - r3; 
    return atan2(u2.norm() * u1.dot(u2.cross(u3)), (u1.cross(u2)).dot(u2.cross(u3))); 
}

/**
 * Generate a random position for the next atom in a 3-atom segment, given 
 * a desired bond length and bond angle. 
 *
 * @param r1 First atom. 
 * @param r2 Second atom.
 * @param length Bond length between r2 and the new atom. 
 * @param angle Bond angle. 
 * @param rng Random number generator. 
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @returns Position of the third atom. 
 */
template <typename T>
Matrix<T, 3, 1> generateNextAtom(const Ref<const Matrix<T, 3, 1> >& r1, 
                                 const Ref<const Matrix<T, 3, 1> >& r2,
                                 const T length, const T angle, 
                                 boost::random::mt19937& rng, 
                                 boost::random::uniform_01<>& uniform_dist)
{
    // If the bond angle at atom 2 is \pi, then the position of atom 3 is fixed
    if (abs(abs(angle) - boost::math::constants::pi<T>()) < 1e-6)
    {
        Matrix<T, 3, 1> u = r2 - r1; 
        return r2 + length * (u / u.norm()); 
    }

    // Get the distance vector and direction from atom 1 to atom 2 
    Matrix<T, 3, 1> u = r2 - r1;
    u /= u.norm(); 

    // Randomly sample an orthonormal basis that contains the 2-1 direction 
    // vector 
    //
    // The other two vectors in this basis span the plane normal to the 2-1
    // direction vector (up to translation)
    Matrix<T, 2, 3> basis = generateOrthonormalBasis3D<T>(-u, rng, uniform_dist);
    Matrix<T, 3, 1> v = basis.row(0).transpose(); 
    Matrix<T, 3, 1> w = basis.row(1).transpose();

    // Rotate the direction vector from atom 2 to atom 1 about atom 2 by 
    // the given angle within the plane normal to w, which must contain u
    //
    // To do this, we rotate the vector (1, 0, 0) in the xy-plane, and 
    // perform a change of basis in which x <-> -u, y <-> v, and z <-> w
    //
    // This yields a vector that is orthogonal to w and has the desired 
    // angle from -u 
    Matrix<T, 2, 2> rot; 
    rot << cos(angle), -sin(angle),
           sin(angle),  cos(angle);
    Matrix<T, 2, 1> e; 
    e << 1, 0;  
    Matrix<T, 3, 3> trans; 
    trans << -u(0), v(0), w(0), 
             -u(1), v(1), w(1), 
             -u(2), v(2), w(2);
    Matrix<T, 3, 1> e_rot = Matrix<T, 3, 1>::Zero();
    e_rot.head(2) = rot * e; 
    Matrix<T, 3, 1> u_new = trans * e_rot;

    // Get the position of the next atom 
    return r2 + length * u_new;  
}

/**
 * Generate one of two possible positions for the next atom in a 4-atom 
 * segment, given the desired bond length, bond angle, and dihedral angle.
 *
 * @param r1 First atom. 
 * @param r2 Second atom.
 * @param r3 Third atom. 
 * @param length Bond length between r3 and the new atom. 
 * @param angle Bond angle along the 3-atom segment given by r2, r3, and the
 *              new atom.
 * @param dihedral Dihedral angle along the entire 4-atom segment. 
 * @param rng Random number generator. 
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @param sign If positive, assume a positive dihedral angle; if negative, 
 *             assume a negative dihedral angle; if zero, choose one or the 
 *             other randomly. 
 * @returns Position of the fourth atom. 
 */
template <typename T>
Matrix<T, 3, 1> generateNextAtomDihedral(const Ref<const Matrix<T, 3, 1> >& r1, 
                                         const Ref<const Matrix<T, 3, 1> >& r2,
                                         const Ref<const Matrix<T, 3, 1> >& r3, 
                                         const T length,
                                         const T angle,
                                         const T dihedral,  
                                         boost::random::mt19937& rng,
                                         boost::random::uniform_01<>& uniform_dist,
                                         int sign = 0)
{
    // If the bond angle at atom 3 is \pi, then the position of atom 4 is fixed
    // and the dihedral angle should be ignored
    if (abs(abs(angle) - boost::math::constants::pi<T>()) < 1e-6)
    {
        Matrix<T, 3, 1> u2 = r3 - r2;
        return r3 + length * (u2 / u2.norm()); 
    }

    // Get the distance vector and direction from atom 1 to atom 2 and from
    // atom 2 to atom 3 
    Matrix<T, 3, 1> u1 = r2 - r1;
    Matrix<T, 3, 1> u2 = r3 - r2;
    Matrix<T, 3, 1> v1 = u1 / u1.norm(); 
    Matrix<T, 3, 1> v2 = u2 / u2.norm(); 
    Matrix<T, 3, 1> x = -v2; 

    // Get the cross product between u1 and u2, which is normal to the plane
    // containing atoms 1, 2, 3
    Matrix<T, 3, 1> y = v1.cross(v2); 
    y /= y.norm();  

    // Get the cross product between z and x; this yields a right-handed 
    // orthonormal basis (x, y, z)
    Matrix<T, 3, 1> z = x.cross(y); 

    // Get the position of the next atom
    if (sign == 0)
        sign = (uniform_dist(rng) < 0.5 ? 1 : -1);
    else if (sign > 0)
        sign = 1; 
    else 
        sign = -1; 
    Matrix<T, 3, 1> w = (
        cos(angle) * x + sin(angle) * (cos(dihedral) * z + sign * sin(dihedral) * y)
    );  
    return r3 + length * w; 
}

/**
 * Return the Lennard-Jones potential for the given distance. 
 *
 * @param r Input distance.
 * @param eps Energy parameter. 
 * @param sigma Length-scale parameter. 
 * @param wca If true, use the repulsive component of the Weeks-Chandler-
 *            Andersen decomposition. 
 * @returns Lennard-Jones potential value. 
 */
template <typename T>
T lj(const T r, const T eps, const T sigma, const bool wca = false)
{
    T ratio = pow(sigma / r, 6);
    if (wca)
    {
        if (r < pow(2, 1./6.) * sigma)
            return 4 * eps * (ratio * ratio - ratio) + eps; 
        else 
            return 0; 
    }
    else
    { 
        return 4 * eps * (ratio * ratio - ratio);
    } 
}

/**
 * Return the FENE bond potential for the given distance. 
 *
 * The Lennard-Jones term is omitted in this calculation. 
 *
 * @param r Input distance (bond length). 
 * @param K Energy parameter. 
 * @param R0 Maximum bond length. 
 * @returns FENE bond potential value. 
 */
template <typename T>
T bondFene(const T r, const T K, const T R0)
{
    if (r >= R0)
    {
        return std::numeric_limits<T>::infinity();
    } 
    else
    {
        T ratio = r / R0;
        return -0.5 * K * R0 * R0 * log(1 - ratio * ratio); 
    }
}

/**
 * Return the cosine/delta angle potential for the given bond angle. 
 *
 * @param theta Input angle. 
 * @param K Energy parameter. 
 * @param theta0 Equilibrium angle. 
 * @returns Potential value. 
 */
template <typename T>
T angleCosine(const T theta, const T K, const T theta0)
{
    return K * (1 - cos(theta - theta0)); 
}

/**
 * Return the two-component Gaussian mixture potential for the given bond 
 * angle. 
 *
 * @param theta Input angle. 
 * @param A1, A2 Weights of the two Gaussian components.
 * @param w1, w2 Twice the standard deviations of the two Gaussian components. 
 * @param theta1, theta2 Means of the two Gaussian components. 
 * @param kT Boltzmann's constant times temperature (in the appropriate units). 
 * @returns Potential value.  
 */
template <typename T>
T angleDualGaussianMixture(const T theta, const T A1, const T A2, const T w1, 
                           const T w2, const T theta1, const T theta2, 
                           const T kT)
{
    T prob = 0; 
    T dtheta1 = theta - theta1; 
    T dtheta2 = theta - theta2;
    prob += A1 * exp(-2 * dtheta1 * dtheta1 / (w1 * w1)) / (w1 * sqrt(boost::math::constants::half_pi<T>()));
    prob += A2 * exp(-2 * dtheta2 * dtheta2 / (w2 * w2)) / (w2 * sqrt(boost::math::constants::half_pi<T>()));
    
    return -kT * log(prob);  
}

/**
 * Return the harmonic potential for the given dihedral angle. 
 *
 * @param phi Input angle. 
 * @param K Energy parameter. 
 * @param d Phase parameter (either -1 or +1). 
 * @param n Multiplicity parameter, which determines the frequency of the 
 *          potential. 
 * @returns Potential value. 
 */
template <typename T>
T dihedralHarmonic(const T phi, const T K, const int d, const int n)
{
    return K * (1 + d * cos(n * phi)); 
}

/**
 * Generate an empirical CDF corresponding to the FENE potential.
 *
 * This function generates a CDF corresponding to the probability distribution
 * that arises upon assigning to each bond length r the weight, 
 *
 * r^2 \exp{\{ -E / kT \}}, 
 * 
 * where E is the corresponding FENE energy plus the corresponding Weeks-
 * Chandler-Andersen energy. 
 *
 * @param eps Lennard-Jones (Weeks-Chandler-Andersen) energy parameter. 
 * @param sigma Lennard-Jones (Weeks-Chandler-Andersen) length-scale parameter. 
 * @param K FENE energy parameter. 
 * @param R0 Maximum bond length.
 * @param kT Boltzmann's constant times temperature (in the appropriate units).
 * @param n_bins Number of bins.
 * @param delta Bins range from delta to R0 - delta. 
 */
template <typename T>
Matrix<T, Dynamic, 2> getFeneCDF(const T eps, const T sigma, const T K, 
                                 const T R0, const T kT, const int n_bins,
                                 const T delta = 1e-6)
{
    // Evaluate the FENE density at each point along the range 
    Matrix<T, Dynamic, 1> bins = Matrix<T, Dynamic, 1>::LinSpaced(
        n_bins + 1, delta, R0 - delta
    );
    Matrix<T, Dynamic, 1> density = Matrix<T, Dynamic, 1>::Zero(n_bins + 1);
    Matrix<T, Dynamic, 1> log_density = Matrix<T, Dynamic, 1>::Zero(n_bins + 1); 
    for (int i = 0; i < n_bins + 1; ++i)
    {
        // The density is proportional to r^2 * \exp{\{ -E / kT \}}, where 
        // E is the total energy (FENE and WCA)
        //
        // Compute the log-density to avoid overflow
        T energy = lj<T>(bins(i), eps, sigma, true) + bondFene<T>(bins(i), K, R0);
        log_density(i) = 2 * log(bins(i)) - energy / kT; 
    }

    // Shift then exponentiate 
    T max_log_density = log_density.maxCoeff(); 
    density = (log_density.array() - max_log_density).exp().matrix();

    // Normalize the density
    T integral = 0; 
    for (int i = 1; i < n_bins + 1; ++i)
        integral += ((bins(i) - bins(i - 1)) * density(i));
    density /= integral;  

    // Get the CDF 
    Matrix<T, Dynamic, 2> cdf = Matrix<T, Dynamic, 2>::Zero(n_bins + 1, 2);
    cdf(0, 0) = bins(0);  
    for (int i = 1; i < n_bins + 1; ++i)
    {
        cdf(i, 0) = bins(i); 
        cdf(i, 1) = cdf(i - 1, 1) + (bins(i) - bins(i - 1)) * density(i); 
    }

    return cdf;  
}

/**
 * Identify, via binary search, the index of the nearest value to the given
 * query, x, in the given array.
 *
 * The values are assumed to be distinct and sorted in ascending order.
 *
 * @param values Input array.
 * @param x Query value. 
 * @returns Index of nearest value to x.  
 */
template <typename T>
int nearestValue(const Ref<const Matrix<T, Dynamic, 1> >& values, const T x)
{
    // Quickly check if x is less than the first value or greater than the
    // last value
    if (x <= values(0))
        return 0; 
    else if (x >= values(values.size() - 1))
        return values.size() - 1;  

    // Otherwise, do binary search 
    int low = 0; 
    int high = values.size() - 1;
    int nearest_idx = 0;
    while (low <= high)
    {
        int mid = (low + high) / 2;

        // If x falls between values(mid) and values(mid + 1), set mid 
        // as the nearest index
        if (values(mid) <= x && x < values(mid + 1))
        {
            nearest_idx = mid; 
            break;
        }
        // If x is greater than values(mid + 1), then increase low 
        else if (x >= values(mid + 1))
        {
            low = mid + 1;
        }
        // If x is less than values(mid), then decrease high  
        else
        {
            high = mid - 1;
        }
    }

    // Note that this loop cannot have exited due to low > high
    if (low > high)
        throw std::runtime_error("Unexpected error during binary search");

    // Check which of the two endpoints of the interval is nearest
    if (x - values(nearest_idx) < values(nearest_idx + 1) - x)
        return nearest_idx; 
    else 
        return nearest_idx + 1;  
}

/**
 * Sample a bond length according to the FENE potential.
 *
 * This function samples from the probability distribution that arises upon
 * assigning to each bond length r the weight
 *
 * r^2 \exp{\{ -E / kT \}}, 
 * 
 * where E is the corresponding FENE energy plus the corresponding Weeks-
 * Chandler-Andersen energy.
 *
 * This function uses a pre-computed CDF for this distribution.  
 *
 * @param rng Random number generator.
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @param cdf Pre-computed CDF for this distribution. 
 * @returns Sampled bond length. 
 */
template <typename T>
T sampleFene(boost::random::mt19937& rng, boost::random::uniform_01<>& uniform_dist,
             const Ref<const Matrix<T, Dynamic, 2> >& cdf)
{
    // Sample a value from [0, 1], then get the pre-image of that value via
    // the CDF
    //
    // Column 0 contains the bin edges (bond lengths), column 1 contains the CDF
    T u = uniform_dist(rng); 
    int idx = nearestValue<T>(cdf.col(1), u);

    // If the chosen input value is exactly u, return the chosen CDF value
    if (cdf(idx, 1) == u)
    {
        return cdf(idx, 0); 
    }
    // If the chosen input value is smaller than u ... 
    else if (cdf(idx, 1) < u)
    {
        // If the chosen value is the very last value in the array, then 
        // return it 
        if (idx == cdf.rows() - 1)
        {
            return cdf(idx, 0);
        } 
        // Otherwise, interpolate between the chosen value and the one above it
        else
        {
            // Map from CDF value to bond length  
            T x1 = cdf(idx, 1); 
            T x2 = cdf(idx + 1, 1); 
            T y1 = cdf(idx, 0); 
            T y2 = cdf(idx + 1, 0);
            return y1 + (y2 - y1) * (u - x1) / (x2 - x1); 
        } 
    }
    else    // Otherwise ...  
    {
        // If the chosen value is the very first value in the array, then
        // return it 
        if (idx == 0)
        {
            return cdf(idx, 0); 
        }
        // Otherwise, interpolate between the chosen value and the one below it
        else 
        {
            // Map from CDF value to bond length  
            T x1 = cdf(idx - 1, 1); 
            T x2 = cdf(idx, 1); 
            T y1 = cdf(idx - 1, 0); 
            T y2 = cdf(idx, 0); 
            return y1 + (y2 - y1) * (u - x1) / (x2 - x1); 
        } 
    } 
}

/**
 * Sample a bond length according to the FENE potential. 
 *
 * This function samples from the probability distribution that arises upon
 * assigning to each bond length r the weight
 *
 * r^2 \exp{\{ -E / kT \}}, 
 * 
 * where E is the corresponding FENE energy plus the corresponding Weeks-
 * Chandler-Andersen energy.
 *
 * This function uses a Metropolis-Hastings-like strategy to perform this
 * sampling. 
 *
 * @param eps Lennard-Jones (Weeks-Chandler-Andersen) energy parameter. 
 * @param sigma Lennard-Jones (Weeks-Chandler-Andersen) length-scale parameter. 
 * @param K FENE energy parameter. 
 * @param R0 Maximum bond length.
 * @param kT Boltzmann's constant times temperature (in the appropriate units). 
 * @param rng Random number generator.
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @param n_burnin Number of burn-in samples. 
 * @returns Sampled bond length. 
 */
template <typename T>
T sampleFene(const T eps, const T sigma, const T K, const T R0, const T kT,
             boost::random::mt19937& rng, boost::random::uniform_01<>& uniform_dist,
             const int n_burnin = 50)
{
    // Define the effective FENE energy 
    auto energy = [&eps, &sigma, &K, &R0, &kT](const T r) -> T
    {
        return lj<T>(r, eps, sigma, true) + bondFene<T>(r, K, R0) - 2 * kT * log(r); 
    };

    // Run the Monte Carlo procedure ...
    T length = 0.9 * R0; 
    T stdev = 0.01 * R0;    // Start at a small initial value 
    int iter = 0; 
    while (iter < n_burnin)
    {
        // Try sampling the next value 
        T length_new = length + stdev * standardNormal<T>(rng, uniform_dist);
        
        // If the length falls outside the interval, reflect accordingly
        int n_reflections = 0;  
        while (length_new < 0 || length_new >= R0)
        {
            if (length_new < 0)
                length_new *= -1;
            else    // length_new >= R0 
                length_new = 2 * R0 - length_new;

            // Keep track of the number of reflections 
            n_reflections++; 
            if (n_reflections > 10) 
                break; 
        }

        // Check if the reflection has failed 
        if (length_new <= 0 || length_new >= R0)
            continue; 

        // Compute the energy difference
        T energy_diff = energy(length_new) - energy(length); 

        // Calculate the Metropolis acceptance probability
        T prob_accept = min(1.0, exp(-energy_diff / kT));

        // Adapt the standard deviation of the sampling distribution 
        if (prob_accept < 0.2)
            stdev *= 0.8; 
        else if (prob_accept > 0.8)
            stdev *= 1.25;

        // Accept the next value 
        if (uniform_dist(rng) < prob_accept)
            length = length_new; 
        iter++;  
    }

    return length;  
}

/**
 * Sample a bond angle according to the cosine potential.
 *
 * This function samples from the probability distribution that arises upon
 * assigning to each bond angle \theta the weight,
 *
 * \sin(\theta) \exp{\{ -E / kT \}},
 *
 * where E is the corresponding cosine potential value. 
 *
 * @param K Energy parameter.
 * @param theta0 Equilibrium angle. 
 * @param kT Boltzmann's constant times temperature (in the appropriate units).
 * @param rng Random number generator.
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @returns Sampled bond angle. 
 */
template <typename T>
T sampleAngleCosine(const T K, const T theta0, const T kT,
                    boost::random::mt19937& rng,
                    boost::random::uniform_01<>& uniform_dist)
{
    // If K == 0, then simply sample with probability proportional to sin(theta)
    //
    // This can be done by sampling from [-1, 1] and taking arccos of that value
    if (K == 0)
    {
        T u = -1 + 2 * uniform_dist(rng);
        return acosSafe<T>(u);  
    }
    // Otherwise, run the rejection sampling procedure ...
    else
    { 
        T theta = theta0; 
        T kappa = K / kT;
        bool accept = false; 
        while (!accept)
        {
            // Try sampling the next value from the Boltzmann distribution
            // (minus the Jacobian) 
            theta = vonMises<T>(theta0, kappa, rng, uniform_dist);

            // Accept with probability sin(abs(theta))
            accept = (uniform_dist(rng) < sin(abs(theta))); 
        }
        return abs(theta);  
    }
}

/**
 * Sample a bond angle according to a two-component Gaussian mixture potential.
 *
 * This function samples from the probability distribution that arises upon
 * assigning to each bond angle \theta the weight,
 *
 * \sin(\theta) \exp{\{ -E / kT \}},
 *
 * where E is the corresponding potential value. 
 *
 * @param A1, A2 Weights of the two Gaussian components.
 * @param w1, w2 Twice the standard deviations of the two Gaussian components. 
 * @param theta1, theta2 Means of the two Gaussian components. 
 * @param kT Boltzmann's constant times temperature (in the appropriate units). 
 * @param rng Random number generator.
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @returns Sampled bond angle. 
 */
template <typename T>
T sampleAngleDualGaussianMixture(const T A1, const T A2, const T w1, const T w2,
                                 const T theta1, const T theta2, const T kT,
                                 boost::random::mt19937& rng,
                                 boost::random::uniform_01<>& uniform_dist)
{
    // Set up a four-component von Mises mixture, reflected about 0
    T mu1 = theta1; 
    T mu2 = theta2;
    T mu3 = -theta1; 
    T mu4 = -theta2;
    T stdev1 = w1 / 2;     // w1 and w2 are double the standard deviations 
    T stdev2 = w2 / 2;  
    T kappa1 = 1 / (stdev1 * stdev1);    // Concentration = 1 / variance  
    T kappa2 = 1 / (stdev2 * stdev2);
    T kappa3 = kappa1; 
    T kappa4 = kappa2; 
    T weight1 = A1 / 2; 
    T weight2 = A2 / 2; 
    T weight3 = A1 / 2; 
    T weight4 = A2 / 2;
    boost::random::discrete_distribution<> component_dist({weight1, weight2, weight3, weight4});

    // Run the rejection sampling procedure ... 
    T theta = theta1;
    bool accept = false; 
    while (!accept) 
    {
        // Try sampling the next value
        int choice = component_dist(rng); 
        if (choice == 0)
            theta = vonMises<T>(mu1, kappa1, rng, uniform_dist); 
        else if (choice == 1) 
            theta = vonMises<T>(mu2, kappa2, rng, uniform_dist);
        else if (choice == 2) 
            theta = vonMises<T>(mu3, kappa3, rng, uniform_dist); 
        else    // choice == 3
            theta = vonMises<T>(mu4, kappa4, rng, uniform_dist);

        // Accept with probability sin(abs(theta))
        accept = (uniform_dist(rng) < sin(abs(theta))); 
    }

    return abs(theta);  
}

/**
 * Sample a dihedral angle according to the harmonic potential.
 *
 * The phase parameter, d, is set to 1 (so that the potential is maximized 
 * at 0 (cis) and minimized at \pi (trans)). 
 *
 * The multiplicity parameter, n, is set to 1 (so that the potential has a
 * unique minimum at \pi == -\pi).  
 *
 * @param K Energy parameter.
 * @param kT Boltzmann's constant times temperature (in the appropriate units).
 * @param rng Random number generator.
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @returns Sampled dihedral angle. 
 */
template <typename T>
T sampleDihedralHarmonic(const T K, const T kT, boost::random::mt19937& rng,
                         boost::random::uniform_01<>& uniform_dist)
{
    // Note that the Boltzmann distribution is proportional to 
    //
    // e^{-U / kT} = e^{-(K(1 + \cos{\phi})) / kT}
    // = e^{(-K - K \cos{\phi}) / kT}
    // \propto e^{-K \cos{\phi} / kT}
    // = e^{K \cos{(\phi - \pi)} / kT},
    //
    // which is proportional to the von Mises probability with mean \pi and 
    // concentration K / kT
    //
    // If K is zero, then simply return a uniformly distributed value between
    // -\pi and \pi
    if (K == 0)
    {
        return -boost::math::constants::pi<T>() + boost::math::constants::two_pi<T>() * uniform_dist(rng); 
    }
    else 
    {
        T kappa = K / kT;
        return vonMises<T>(boost::math::constants::pi<T>(), kappa, rng, uniform_dist);
    } 
}

/**
 * Given an array of atomic coordinates and a distance threshold, return all
 * pairs of atoms that are within that distance. 
 *
 * @param r Array of atomic coordinates. 
 * @param neighbor_threshold Distance threshold. 
 * @returns Array of neighboring pairs of atoms (indices and the distance
 *          between them). 
 */
template <typename T>
Matrix<T, Dynamic, 3> getNeighbors(const Ref<const Matrix<T, Dynamic, 3> >& r, 
                                   const T neighbor_threshold)
{
    const int n = r.rows();
    int n_neighbors = 0; 
    Matrix<T, Dynamic, 3> neighbors(n_neighbors, 3);  
    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            T dij = (r.row(i) - r.row(j)).norm(); 
            if (dij < neighbor_threshold)
            {
                n_neighbors++; 
                neighbors.conservativeResize(n_neighbors, 3); 
                neighbors(n_neighbors - 1, 0) = i; 
                neighbors(n_neighbors - 1, 1) = j; 
                neighbors(n_neighbors - 1, 2) = dij;  
            } 
        }
    } 

    return neighbors; 
} 

/**
 * Given two arrays of atomic coordinates and a distance threshold, return all
 * pairs of atoms, with one atom in the first array and the other atom in the
 * other array, that are within that distance. 
 *
 * @param r1 First array of atomic coordinates.
 * @param r2 Second array of atomic coordinates.  
 * @param neighbor_threshold Distance threshold. 
 * @returns Array of neighboring pairs of atoms (indices and the distance
 *          between them). 
 */
template <typename T>
Matrix<T, Dynamic, 3> getNeighbors(const Ref<const Matrix<T, Dynamic, 3> >& r1, 
                                   const Ref<const Matrix<T, Dynamic, 3> >& r2, 
                                   const T neighbor_threshold)
{
    const int n1 = r1.rows();
    const int n2 = r2.rows(); 
    int n_neighbors = 0; 
    Matrix<T, Dynamic, 3> neighbors(n_neighbors, 3);  
    for (int i = 0; i < n1; ++i)
    {
        for (int j = 0; j < n2; ++j)
        {
            T dij = (r1.row(i) - r2.row(j)).norm(); 
            if (dij < neighbor_threshold)
            {
                n_neighbors++; 
                neighbors.conservativeResize(n_neighbors, 3); 
                neighbors(n_neighbors - 1, 0) = i; 
                neighbors(n_neighbors - 1, 1) = j;
                neighbors(n_neighbors - 1, 2) = dij;  
            } 
        }
    } 

    return neighbors; 
} 

/**
 * Parse a JSON file specifying simulation parameters.
 *
 * @param filename Input JSON configurations file.
 * @returns `boost::json::value` instance containing the JSON data.  
 */
boost::json::value parseConfigFile(const std::string filename)
{
    std::string line;
    std::ifstream infile(filename);
    boost::json::stream_parser p; 
    boost::json::error_code ec;  
    while (std::getline(infile, line))
    {
        p.write(line, ec); 
        if (ec)
            return nullptr; 
    }
    p.finish(ec); 
    if (ec)
        return nullptr;
    
    return p.release(); 
}

#endif
