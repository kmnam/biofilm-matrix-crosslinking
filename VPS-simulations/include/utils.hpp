/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     1/30/2026
 */

#ifndef POLYMER_UTILS_HPP 
#define POLYMER_UTILS_HPP 

#include <stdexcept>
#include <cmath>
#include <string>
#include <limits>
#include <unordered_map>
#include <functional>
#include <Eigen/Dense>
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
using std::isinf; 

using namespace Eigen;

enum Units
{
    MICRO,
    NANO
};

enum AngleMode
{
    COSINE,
    GAUSSIAN
};

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
 * @param w1, w2 Standard deviations of the two Gaussian components. 
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
    
    return -kT * prob;  
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
 * Sample a bond length according to the FENE potential. 
 *
 * This function samples from the probability distribution that arises upon
 * assigning to each bond length r the weight r^2 \exp{\{ -E / kT \}}, 
 * where E is the corresponding FENE energy plus the corresponding 
 * Weeks-Chandler-Andersen energy. 
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
    for (int i = 0; i < n_burnin + 1; ++i)
    {
        // Try sampling the next value 
        T length_new = length + stdev * standardNormal<T>(rng, uniform_dist);
        
        // If the length falls outside the interval, reflect accordingly 
        while (length_new <= 0 || length_new >= R0)
        {
            if (length_new <= 0)
                length_new *= -1;
            else    // length_new >= R0 
                length_new = 2 * R0 - length_new; 
        } 

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
 * @param n_burnin Number of burn-in samples. 
 * @returns Sampled bond angle. 
 */
template <typename T>
T sampleAngleCosine(const T K, const T theta0, const T kT,
                    boost::random::mt19937& rng,
                    boost::random::uniform_01<>& uniform_dist,
                    const int n_burnin = 50)
{
    // Run the Monte Carlo procedure ...
    T theta = theta0; 
    T kappa = K / kT;
    for (int i = 0; i < n_burnin + 1; ++i)
    {
        // Try sampling the next value from the Boltzmann distribution
        // (minus the Jacobian) 
        T theta_new = vonMises<T>(theta0, kappa, rng, uniform_dist); 
        if (theta_new < 0)
            theta_new *= -1;

        // Calculate the Metropolis acceptance probability
        T prob_accept = min(1.0, sin(theta_new) / sin(theta)); 

        // Accept the next value 
        if (uniform_dist(rng) < prob_accept)
            theta = theta_new;  
    }

    return theta;  
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
 * @param w1, w2 Standard deviations of the two Gaussian components. 
 * @param theta1, theta2 Means of the two Gaussian components. 
 * @param kT Boltzmann's constant times temperature (in the appropriate units). 
 * @param rng Random number generator.
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @param n_burnin Number of burn-in samples. 
 * @returns Sampled bond angle. 
 */
template <typename T>
T sampleAngleDualGaussianMixture(const T A1, const T A2, const T w1, const T w2,
                                 const T theta1, const T theta2, const T kT,
                                 boost::random::mt19937& rng,
                                 boost::random::uniform_01<>& uniform_dist,
                                 const int n_burnin = 50)
{
    // Set up a four-component von Mises mixture, reflected about 0
    T mu1 = theta1; 
    T mu2 = theta2;
    T mu3 = -theta1; 
    T mu4 = -theta2; 
    T kappa1 = 1 / (w1 * w1); 
    T kappa2 = 1 / (w2 * w2);
    T kappa3 = kappa1; 
    T kappa4 = kappa2; 
    T weight1 = A1 / 2; 
    T weight2 = A2 / 2; 
    T weight3 = A1 / 2; 
    T weight4 = A2 / 2;
    boost::random::discrete_distribution<> component_dist({weight1, weight2, weight3, weight4}); 

    // Run the Monte Carlo procedure ...
    T theta = theta1;
    for (int i = 0; i < n_burnin + 1; ++i)
    {
        // Try sampling the next value
        int choice = component_dist(rng); 
        T theta_new; 
        if (choice == 0)
            theta_new = vonMises<T>(mu1, kappa1, rng, uniform_dist); 
        else if (choice == 1) 
            theta_new = vonMises<T>(mu2, kappa2, rng, uniform_dist);
        else if (choice == 2) 
            theta_new = vonMises<T>(mu3, kappa3, rng, uniform_dist); 
        else    // choice == 3
            theta_new = vonMises<T>(mu4, kappa4, rng, uniform_dist);

        // Fold the distribution back into [0, \pi)
        if (theta_new < 0)
            theta_new *= -1;
        
        // Calculate the Metropolis acceptance probability
        T prob_accept = min(1.0, sin(theta_new) / sin(theta)); 

        // Accept the next value 
        if (uniform_dist(rng) < prob_accept)
            theta = theta_new;  
    }

    return theta;  
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
    T kappa = K / kT;
    return vonMises<T>(boost::math::constants::pi<T>(), kappa, rng, uniform_dist); 
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
 * A class for storing and manipulating linear polymer configurations. 
 */
template <typename T>
class PolymerConfiguration 
{
    private:
        int length; 
        Matrix<T, Dynamic, 3> r;

        /**
         * Get the energy arising from all interactions between the given
         * segment and the atoms along the polymer with indices [0, ..., idx - 1]
         * and [idx + n, ..., N - 1], where n is the segment length and N is
         * the polymer length. 
         *
         * @param segment Input segment. 
         * @param idx Index demarcating the polymer atoms to consider. 
         * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
         * @param neighbor_threshold Distance threshold for identifying
         *                           neighboring (non-bonded) atoms. 
         * @param fene_params FENE parameters. 
         * @param angle_mode Angle potential type.  
         * @param angle_params Angle potential parameters. Must include the 
         *                     cosine potential parameters (K and theta0) or
         *                     the dual Gaussian mixture potential parameters
         *                     (A1, A2, w1, w2, theta1, theta2). 
         * @param dihedral_params Dihedral angle potential parameters. 
         * @returns Interaction energy between segment and polymer.  
         */
        T getSegmentInteractionEnergy(const Ref<const Matrix<T, Dynamic, 3> >& segment,
                                      const int idx,
                                      std::unordered_map<std::string, T>& lj_params,  
                                      const T neighbor_threshold, 
                                      std::unordered_map<std::string, T>& fene_params,
                                      const AngleMode angle_mode, 
                                      std::unordered_map<std::string, T>& angle_params,
                                      std::unordered_map<std::string, T>& dihedral_params)
        {
            // Check that the specified polymer indices to slice out of the 
            // polymer are valid
            const int n = segment.rows(); 
            if (idx < 0 || idx + n - 1 >= this->length)
                throw std::runtime_error(
                    "Specified segment cannot be inserted into polymer at specified index"
                ); 

            // ----------------------------------------------------------- //
            // Get the non-bonded interaction energy
            // ----------------------------------------------------------- //
            // First identify all pairs of neighboring atoms (p, q), where p
            // lies within the entire polymer and q lies within the segment
            const int n1 = idx; 
            const int n2 = this->length - idx - n; 
            Matrix<T, Dynamic, 3> r_sub(n1 + n2, 3);
            r_sub(Eigen::seqN(0, n1), Eigen::all) = this->r(Eigen::seqN(0, n1), Eigen::all); 
            r_sub(Eigen::seqN(n1, n2), Eigen::all) = this->r(Eigen::seqN(idx + n, n2), Eigen::all); 
            Matrix<T, Dynamic, 3> neighbors = getNeighbors<T>(
                r_sub, segment, neighbor_threshold
            );

            // Calculate the non-bonded interaction energy 
            T energy_curr = 0; 
            for (int i = 0; i < neighbors.rows(); ++i)
                energy_curr += lj<T>(
                    neighbors(i, 2), lj_params["eps"], lj_params["sigma"], true
                );

            // ----------------------------------------------------------- //
            // Get the bonded interaction energy 
            // ----------------------------------------------------------- //
            // Calculate the bond energy, including the bonds on either side
            // of the segment as well
            for (int i = 0; i < n - 1; ++i)
                energy_curr += bondFene<T>(
                    (segment.row(i + 1) - segment.row(i)).norm(),
                    fene_params["K"], fene_params["R0"]
                );

            // Identify adjacent atoms to the segment along the polymer
            int n_adj_bonds = 0; 
            Matrix<T, Dynamic, 3> adj_bonds(n_adj_bonds, 3);
            if (idx > 0)
            {
                n_adj_bonds++; 
                adj_bonds.conservativeResize(n_adj_bonds, 3); 
                adj_bonds.row(n_adj_bonds - 1) = segment.row(0) - this->r.row(idx - 1); 
            }
            if (idx + n < this->length)
            {
                n_adj_bonds++; 
                adj_bonds.conservativeResize(n_adj_bonds, 3); 
                adj_bonds.row(n_adj_bonds - 1) = this->r.row(idx + n) - segment.row(n - 1); 
            }
            for (int i = 0; i < n_adj_bonds; ++i) 
                energy_curr += bondFene<T>(
                    adj_bonds.row(i).norm(), fene_params["K"], fene_params["R0"]
                );

            // ----------------------------------------------------------- //
            // Get the bond angle energy
            // ----------------------------------------------------------- //
            // Define angle potential function, depending on the parameters
            std::function<T(const T)> potential;  
            if (angle_mode == AngleMode::GAUSSIAN)
            {
                potential = [this, &angle_params](const T theta) -> T
                {
                    return angleDualGaussianMixture<T>(
                        theta, angle_params["A1"], angle_params["A2"],
                        angle_params["w1"], angle_params["w2"],
                        angle_params["theta1"], angle_params["theta2"],
                        this->kT
                    ); 
                }; 
            }
            else if (angle_mode == AngleMode::COSINE)
            {
                potential = [&angle_params](const T theta) -> T
                {
                    return angleCosine<T>(
                        theta, angle_params["K"], angle_params["theta0"]
                    );
                }; 
            }
            else 
            {
                throw std::runtime_error("Invalid angle potential mode specified"); 
            }

            // Add the bond angle energies along the segment
            Matrix<T, 3, 1> u, v;  
            for (int i = 0; i < n - 2; ++i)
            {
                u = segment.row(i) - segment.row(i + 1); 
                v = segment.row(i + 2) - segment.row(i + 1); 
                energy_curr += potential(acosSafe<T>(u.dot(v) / (u.norm() * v.norm()))); 
            }

            // Identify bond angles adjacent to the segment (maximum 4)
            //
            // Here, it is assumed that the segment length is >= 2
            //
            // There are then four possible bond angles: P-P-S, P-S-S, S-S-P, S-P-P,
            // depending on the placement of the segment along the polymer 
            int n_adj_angles = 0;
            Matrix<T, Dynamic, 6> adj_angles(n_adj_angles, 6);
            if (idx > 1)
            {
                n_adj_angles++; 
                adj_angles.conservativeResize(n_adj_angles, 6);
                u = this->r.row(idx - 2) - this->r.row(idx - 1); 
                v = segment.row(0) - this->r.row(idx - 1);  
                adj_angles(n_adj_angles - 1, Eigen::seqN(0, 3)) = u;
                adj_angles(n_adj_angles - 1, Eigen::seqN(3, 3)) = v;  
            }
            if (idx > 0)
            {
                n_adj_angles++; 
                adj_angles.conservativeResize(n_adj_angles, 6);
                u = this->r.row(idx - 1) - segment.row(0); 
                v = segment.row(1) - segment.row(0);
                adj_angles(n_adj_angles - 1, Eigen::seqN(0, 3)) = u;
                adj_angles(n_adj_angles - 1, Eigen::seqN(3, 3)) = v;  
            }
            if (idx + n < this->length)
            {
                n_adj_angles++; 
                adj_angles.conservativeResize(n_adj_angles, 6);
                u = segment.row(n - 2) - segment.row(n - 1); 
                v = this->r.row(idx + n) - segment.row(n - 1); 
                adj_angles(n_adj_angles - 1, Eigen::seqN(0, 3)) = u;
                adj_angles(n_adj_angles - 1, Eigen::seqN(3, 3)) = v;  
            }
            if (idx + n + 1 < this->length)
            {
                n_adj_angles++; 
                adj_angles.conservativeResize(n_adj_angles, 6);
                u = segment.row(n - 1) - this->r.row(idx + n); 
                v = this->r.row(idx + n + 1) - this->r.row(idx + n); 
                adj_angles(n_adj_angles - 1, Eigen::seqN(0, 3)) = u;
                adj_angles(n_adj_angles - 1, Eigen::seqN(3, 3)) = v;  
            }
            for (int i = 0; i < n_adj_angles; ++i)
            {
                Matrix<T, 3, 1> u = adj_angles(i, Eigen::seqN(0, 3)); 
                Matrix<T, 3, 1> v = adj_angles(i, Eigen::seqN(3, 3)); 
                T theta = acosSafe<T>(u.dot(v) / (u.norm() * v.norm())); 
                energy_curr += potential(theta); 
            }

            // ----------------------------------------------------------- //
            // Get the dihedral angle energy
            // ----------------------------------------------------------- //
            // Start with the dihedral angles along the segment
            T phi;  
            for (int i = 0; i < n - 3; ++i)
            {
                phi = getDihedral<T>(
                    segment.row(i), segment.row(i + 1), segment.row(i + 2), 
                    segment.row(i + 3)
                );
                energy_curr += dihedralHarmonic<T>(
                    phi, dihedral_params["K"], static_cast<int>(dihedral_params["d"]),
                    static_cast<int>(dihedral_params["n"])
                ); 
            }

            // Identify bond angles adjacent to the segment (maximum 4)
            //
            // Here, it is assumed that the segment length is >= 2
            int n_adj_dihedrals = 0; 
            Matrix<T, Dynamic, 12> adj_dihedrals(n_adj_dihedrals, 12);

            // If the segment length is 2, then there are five possible dihedrals:
            // P-P-P-S, P-P-S-S, P-S-S-P, S-S-P-P, S-P-P-P, depending on the 
            // placement of the segment along the polymer
            if (n == 2)
            {
                if (idx > 2)    // P-P-P-S
                {
                    n_adj_dihedrals++; 
                    adj_dihedrals.conservativeResize(n_adj_dihedrals, 12);
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(0, 3)) = this->r.row(idx - 3); 
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(3, 3)) = this->r.row(idx - 2);
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(6, 3)) = this->r.row(idx - 1);
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(9, 3)) = segment.row(0); 
                }
                if (idx > 1)    // P-P-S-S
                {
                    n_adj_dihedrals++; 
                    adj_dihedrals.conservativeResize(n_adj_dihedrals, 12);
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(0, 3)) = this->r.row(idx - 2); 
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(3, 3)) = this->r.row(idx - 1);
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(6, 3)) = segment.row(0);
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(9, 3)) = segment.row(1); 
                }
                if (idx > 0)    // P-S-S-P
                {
                    n_adj_dihedrals++; 
                    adj_dihedrals.conservativeResize(n_adj_dihedrals, 12);
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(0, 3)) = this->r.row(idx - 1); 
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(3, 3)) = segment.row(0); 
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(6, 3)) = segment.row(1);
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(9, 3)) = this->r.row(idx + 2); 
                }
                if (idx + n < this->length - 1)      // S-S-P-P
                {
                    n_adj_dihedrals++; 
                    adj_dihedrals.conservativeResize(n_adj_dihedrals, 12);
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(0, 3)) = segment.row(0); 
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(3, 3)) = segment.row(1); 
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(6, 3)) = this->r.row(idx + 2); 
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(9, 3)) = this->r.row(idx + 3); 
                }
                if (idx + n < this->length - 2)      // S-P-P-P
                {
                    n_adj_dihedrals++; 
                    adj_dihedrals.conservativeResize(n_adj_dihedrals, 12);
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(0, 3)) = segment.row(1); 
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(3, 3)) = this->r.row(idx + 2); 
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(6, 3)) = this->r.row(idx + 3); 
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(9, 3)) = this->r.row(idx + 4); 
                }
            }
            // If the segment length is > 2, then there are six possible dihedrals:
            // P-P-P-S, P-P-S-S, P-S-S-S, S-S-S-P, S-S-P-P, S-P-P-P, depending
            // on the placement of the segment along the polymer
            else 
            {
                if (idx > 2)    // P-P-P-S
                {
                    n_adj_dihedrals++; 
                    adj_dihedrals.conservativeResize(n_adj_dihedrals, 12);
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(0, 3)) = this->r.row(idx - 3); 
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(3, 3)) = this->r.row(idx - 2);
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(6, 3)) = this->r.row(idx - 1);
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(9, 3)) = segment.row(0); 
                }
                if (idx > 1)    // P-P-S-S
                {
                    n_adj_dihedrals++; 
                    adj_dihedrals.conservativeResize(n_adj_dihedrals, 12);
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(0, 3)) = this->r.row(idx - 2); 
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(3, 3)) = this->r.row(idx - 1);
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(6, 3)) = segment.row(0);
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(9, 3)) = segment.row(1); 
                }
                if (idx > 0)    // P-S-S-S
                {
                    n_adj_dihedrals++; 
                    adj_dihedrals.conservativeResize(n_adj_dihedrals, 12);
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(0, 3)) = this->r.row(idx - 1); 
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(3, 3)) = segment.row(0); 
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(6, 3)) = segment.row(1);
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(9, 3)) = segment.row(2); 
                }
                if (idx + n < this->length)       // S-S-S-P
                {
                    n_adj_dihedrals++; 
                    adj_dihedrals.conservativeResize(n_adj_dihedrals, 12);
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(0, 3)) = segment.row(n - 3); 
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(3, 3)) = segment.row(n - 2); 
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(6, 3)) = segment.row(n - 1); 
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(9, 3)) = this->r.row(idx + n); 
                }
                if (idx + n < this->length - 1)   // S-S-P-P
                {
                    n_adj_dihedrals++; 
                    adj_dihedrals.conservativeResize(n_adj_dihedrals, 12);
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(0, 3)) = segment.row(n - 2); 
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(3, 3)) = segment.row(n - 1); 
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(6, 3)) = this->r.row(idx + n); 
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(9, 3)) = this->r.row(idx + n + 1); 
                }
                if (idx + n < this->length - 2)   // S-P-P-P
                {
                    n_adj_dihedrals++; 
                    adj_dihedrals.conservativeResize(n_adj_dihedrals, 12);
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(0, 3)) = segment.row(n - 1); 
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(3, 3)) = this->r.row(idx + n); 
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(6, 3)) = this->r.row(idx + n + 1); 
                    adj_dihedrals(n_adj_dihedrals - 1, Eigen::seqN(9, 3)) = this->r.row(idx + n + 2); 
                }
            }
            for (int i = 0; i < n_adj_dihedrals; ++i)
            {
                phi = getDihedral<T>(
                    adj_dihedrals(i, Eigen::seqN(0, 3)), 
                    adj_dihedrals(i, Eigen::seqN(3, 3)), 
                    adj_dihedrals(i, Eigen::seqN(6, 3)), 
                    adj_dihedrals(i, Eigen::seqN(9, 3))
                );
                energy_curr += dihedralHarmonic<T>(
                    phi, dihedral_params["K"], static_cast<int>(dihedral_params["d"]),
                    static_cast<int>(dihedral_params["n"])
                );
            }

            return energy_curr; 
        } 

    public:
        T kT;     // Boltzmann's constant times temperature

        /**
         * Default constructor.
         *
         * @param r Atomic coordinates. 
         * @param units Units for keeping track of Boltzmann's constant. 
         * @param temp Temperature (in Kelvin). 
         */
        PolymerConfiguration(const Ref<const Matrix<T, Dynamic, 3> >& r, 
                             const Units units, const T temp)
        {
            this->length = r.rows(); 
            this->r = r; 

            // Set kT according to the given choice of units 
            if (units == Units::NANO)
                this->kT = static_cast<T>(1.380649e-2) * temp;
            else if (units == Units::MICRO)
                this->kT = static_cast<T>(1.380649e-8) * temp; 
        }

        /**
         * Trivial destructor. 
         */
        ~PolymerConfiguration()
        {
        } 

        /**
         * Return the bond lengths. 
         *
         * @returns Vector of bond lengths. 
         */
        Matrix<T, Dynamic, 1> bondLengths() const
        {
            Matrix<T, Dynamic, 1> bond_lengths(this->length - 1);
            for (int i = 0; i < this->length - 1; ++i)
                bond_lengths(i) = (this->r.row(i + 1) - this->r.row(i)).norm(); 

            return bond_lengths;  
        }

        /**
         * Return the length of the polymer.
         *
         * @returns Polymer length.  
         */
        int getLength() const 
        {
            return this->length; 
        }

        /**
         * Return the bond angles.
         *
         * This method defines each bond angle as the angle between the unit
         * vectors (1) from atom i to atom i - 1 and (2) from atom i to i + 1.
         * Therefore, three collinear atoms have a bond angle of 180 degrees 
         * at the central atom. 
         *
         * @returns Vector of bond angles. 
         */
        Matrix<T, Dynamic, 1> bondAngles() const
        {
            Matrix<T, Dynamic, 1> bond_angles(this->length - 2); 
            for (int i = 0; i < this->length - 2; ++i)
            {
                Matrix<T, 3, 1> u = this->r.row(i) - this->r.row(i + 1); 
                u /= u.norm(); 
                Matrix<T, 3, 1> v = this->r.row(i + 2) - this->r.row(i + 1); 
                v /= v.norm(); 
                bond_angles(i) = u.dot(-v);  
            }

            return bond_angles; 
        }

        /**
         * Return the dihedral angles. 
         *
         * @returns Vector of dihedral angles. 
         */
        Matrix<T, Dynamic, 1> dihedralAngles() const 
        {
            Matrix<T, Dynamic, 1> dihedrals(this->length - 3); 
            for (int i = 0; i < this->length - 3; ++i)
                dihedrals(i) = getDihedral<T>(
                    this->r.row(i), this->r.row(i + 1), this->r.row(i + 2), 
                    this->r.row(i + 3)
                );

            return dihedrals;  
        }

        /**
         * Get the atomic coordinates of the segment from `idx` to `idx + n`.
         *
         * @param idx Index of first atom in the segment. 
         * @param n Segment length.
         * @returns Atom coordinates of the segment.  
         */
        Matrix<T, Dynamic, 3> getSegment(const int idx, const int n) const 
        {
            // Check that the specified polymer indices are valid
            if (idx + n - 1 >= this->length)
                throw std::runtime_error(
                    "Specified segment does not exist in polymer"
                ); 

            return this->r(Eigen::seqN(idx, n), Eigen::all); 
        }

        /**
         * Get the minimum distance between the given atom and the polymer. 
         *
         * @param p Input atomic coordinates. 
         * @returns Minimum distance between the atom and the polymer. 
         */
        T getMinDist(const Ref<const Matrix<T, 3, 1> >& p) const 
        {
            return (this->r.rowwise() - p.transpose()).rowwise().norm().minCoeff(); 
        }

        /**
         * Replace the segment starting from atom `idx` within the polymer
         * with the given segment. 
         *
         * @param segment Array of atom coordinates for the new segment.
         * @param idx Index of first atom to replace. 
         */
        void replaceSegment(const Ref<const Matrix<T, Dynamic, 3> >& segment,
                            const int idx)
        {
            // Check that the specified polymer indices are valid
            const int n = segment.rows();  
            if (idx + n - 1 >= this->length)
                throw std::runtime_error(
                    "Specified segment cannot be inserted into polymer at specified index"
                ); 

            // Replace atoms idx, ..., idx + n - 1, where n is the segment length
            for (int i = 0; i < n; ++i)
                this->r.row(idx + i) = segment.row(i); 
        }

        /**
         * Append the given atom onto the tail of the polymer. 
         *
         * @param r Atom coordinates for the new atom. 
         */
        void appendAtomToTail(const Ref<const Matrix<T, 3, 1> >& r)
        {
            this->length++; 
            this->r.conservativeResize(this->length, 3); 
            this->r.row(this->length - 1) = r; 
        }

        /**
         * Append the given atom onto the head of the polymer. 
         *
         * @param r Atom coordinates for the new atom. 
         */
        void appendAtomToHead(const Ref<const Matrix<T, 3, 1> >& r)
        {
            this->length++; 
            this->r.conservativeResize(this->length, 3);
            this->r(Eigen::seqN(1, this->length - 1))
                = this->r(Eigen::seqN(0, this->length - 1)).eval();  
            this->r.row(0) = r; 
        }

        /**
         * Append the given segment onto the tail of the polymer. 
         *
         * @param segment Array of atom coordinates for the new segment. 
         */
        void appendSegmentToTail(const Ref<const Matrix<T, Dynamic, 3> >& segment)
        {
            const int n = segment.rows(); 
            this->length += n;
            this->r.conservativeResize(this->length, 3);  
            for (int i = 0; i < n; ++i)
                this->r.row(this->length - n + i) = segment.row(i);
        }

        /**
         * Append the given segment onto the head of the polymer.
         *
         * @param segment Array of atom coordinates for the new segment.  
         */
        void appendSegmentToHead(const Ref<const Matrix<T, Dynamic, 3> >& segment)
        {
            const int n = segment.rows(); 
            this->length += n; 
            this->r.conservativeResize(this->length, 3); 

            // Copy over the current polymer coordinates
            this->r(Eigen::seqN(n, this->length - n), Eigen::all)
                = this->r(Eigen::seqN(0, this->length - n), Eigen::all).eval();

            // Append the segment onto the head of the polymer 
            for (int i = 0; i < n; ++i)
                this->r.row(i) = segment.row(i); 
        }

        /**
         * Append the given atom onto the tail of the polymer. 
         *
         * @param r Atom coordinates for the new atom. 
         */
        void popAtomFromTail()
        {
            this->r.conservativeResize(this->length - 1, 3);
            this->length--; 
        }

        /**
         * Append the given atom onto the head of the polymer. 
         *
         * @param r Atom coordinates for the new atom. 
         */
        void popAtomFromHead()
        {
            this->r(Eigen::seqN(1, this->length - 1))
                = this->r(Eigen::seqN(0, this->length - 1)).eval(); 
            this->r.conservativeResize(this->length, 3);
            this->length--;
        }

        /**
         * Pop the given segment from the tail of the polymer. 
         *
         * @param idx Index of first atom to remove from the polymer. 
         */
        void popSegmentFromTail(const int idx)
        {
            // Disallow removal of all atoms 
            if (idx < 1 || idx >= this->length)
                throw std::runtime_error("Invalid index for first atom to be removed"); 

            this->r.conservativeResize(idx, 3);
            this->length = idx;  
        }

        /**
         * Pop the given segment from the head of the polymer. 
         *
         * @param idx Index of last atom to remove from the polymer. 
         */
        void popSegmentFromHead(const int idx)
        {
            // Disallow removal of all atoms 
            if (idx < 0 || idx >= this->length - 1)
                throw std::runtime_error("Invalid index for last atom to be removed"); 
            const int n = this->length - idx - 1; 

            // Copy over the current polymer coordinates 
            this->r(Eigen::seqN(0, n), Eigen::all)
                = this->r(Eigen::seqN(idx + 1, n), Eigen::all).eval();

            // Remove the remaining rows 
            this->r.conservativeResize(n, 3); 
            this->length = n; 
        }

        /**
         * Change the polymer according to a reptation move towards the tail, 
         * i.e., remove the 0-th atom and add the given atom to the other end.
         * 
         * @param r New atom to be added to the tail. 
         */
        void reptateTowardsTail(const Ref<const Matrix<T, 3, 1> >& r)
        {
            // Copy over the current polymer coordinates 
            this->r(Eigen::seqN(0, this->length - 1), Eigen::all)
                = this->r(Eigen::seqN(1, this->length - 1), Eigen::all).eval();

            // Add the new atom  
            this->r.row(this->length - 1) = r.transpose(); 
        }

        /**
         * Change the polymer according to a reptation move towards the head,
         * i.e., remove the last atom and add the given atom to the other end.
         * 
         * @param r New atom to be added to the head. 
         */
        void reptateTowardsHead(const Ref<const Matrix<T, 3, 1> >& r)
        {
            // Copy over the current polymer coordinates 
            this->r(Eigen::seqN(1, this->length - 1), Eigen::all)
                = this->r(Eigen::seqN(0, this->length - 1), Eigen::all).eval();

            // Add the new atom  
            this->r.row(0) = r.transpose(); 
        }

        /**
         * Get the energetic contributions of the non-bonded (repulsive)
         * interactions between all atoms to the energy of the current polymer
         * configuration.
         *
         * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
         * @param neighbor_threshold Distance threshold for identifying
         *                           neighboring (non-bonded) atoms. 
         * @returns Non-bonded interaction energy. 
         */
        T getNonbondedEnergy(std::unordered_map<std::string, T>& lj_params, 
                             const T neighbor_threshold) const
        {
            T energy = 0.0; 

            // Identify all pairs of neighboring atoms 
            Matrix<T, Dynamic, 3> neighbors = getNeighbors<T>(this->r, neighbor_threshold);

            // Calculate all non-bonded interaction energies 
            for (int i = 0; i < neighbors.rows(); ++i)
                energy += lj<T>(
                    neighbors(i, 2), lj_params["eps"], lj_params["sigma"], true
                );

            return energy; 
        } 

        /**
         * Get the energetic contributions of the bonded interactions between
         * consecutive atoms to the energy of the current polymer configuration.
         *
         * This function ignores the energetic contributions of non-bonded 
         * interactions between consecutive atoms.  
         *
         * @param fene_params FENE parameters. 
         * @returns Bonded interaction energy. 
         */
        T getBondEnergy(std::unordered_map<std::string, T>& fene_params) const
        {
            T energy = 0.0; 

            // Calculate all bond energies 
            for (int i = 0; i < this->length - 1; ++i)
            {
                Matrix<T, 3, 1> u = this->r.row(i + 1) - this->r.row(i);
                T bond_energy = bondFene<T>(u.norm(), fene_params["K"], fene_params["R0"]);

                // If the i-th bond energy is infinite, just return infinity 
                if (isinf(bond_energy))
                    return std::numeric_limits<T>::infinity(); 
                else 
                    energy += bond_energy;  
            }

            return energy; 
        }

        /**
         * Get the energetic contributions of the bond angles to the energy 
         * of the current polymer configuration.
         *
         * @param angle_mode Angle potential type.  
         * @param angle_params Angle potential parameters. Must include the 
         *                     cosine potential parameters (K and theta0) or
         *                     the dual Gaussian mixture potential parameters
         *                     (A1, A2, w1, w2, theta1, theta2). 
         * @returns Bond angle energy. 
         */
        T getBondAngleEnergy(const AngleMode angle_mode, 
                             std::unordered_map<std::string, T>& angle_params) const
        {
            T energy = 0.0;

            // Define angle potential function, depending on the parameters
            std::function<T(const T)> potential;  
            if (angle_mode == AngleMode::GAUSSIAN)
            {
                potential = [this, &angle_params](const T theta) -> T
                {
                    return angleDualGaussianMixture<T>(
                        theta, angle_params["A1"], angle_params["A2"],
                        angle_params["w1"], angle_params["w2"],
                        angle_params["theta1"], angle_params["theta2"], this->kT
                    ); 
                }; 
            }
            else if (angle_mode == AngleMode::COSINE)
            {
                potential = [&angle_params](const T theta) -> T
                {
                    return angleCosine<T>(theta, angle_params["K"], angle_params["theta0"]);
                }; 
            }
            else 
            {
                throw std::runtime_error("Invalid angle potential mode specified"); 
            }

            // Calculate all bond angle energies 
            for (int i = 0; i < this->length - 2; ++i)
            {
                Matrix<T, 3, 1> u = this->r.row(i) - this->r.row(i + 1);
                Matrix<T, 3, 1> v = this->r.row(i + 2) - this->r.row(i + 1);
                energy += potential(acosSafe<T>(u.dot(v) / (u.norm() * v.norm())));  
            }

            return energy; 
        }

        /**
         * Get the energetic contributions of the dihedral angles along the 
         * polymer to the energy of the current polymer configuration.
         *
         * @param dihedral_params Dihedral angle potential parameters. 
         * @returns Dihedral angle energy. 
         */
        T getDihedralAngleEnergy(std::unordered_map<std::string, T>& dihedral_params) const
        {
            T energy = 0.0; 

            // Calculate all dihedral angle energies 
            for (int i = 0; i < this->length - 3; ++i)
            {
                T phi = getDihedral<T>(
                    this->r.row(i), this->r.row(i + 1), this->r.row(i + 2),
                    this->r.row(i + 3)
                );
                energy += dihedralHarmonic<T>(
                    phi, dihedral_params["K"], static_cast<int>(dihedral_params["d"]),
                    static_cast<int>(dihedral_params["n"])
                ); 
            }

            return energy; 
        }

        /**
         * Get the energy difference between the current polymer configuration 
         * and the configuration that would arise from replacing the current
         * segment at the given index with the given segment. 
         *
         * @param segment Input segment. 
         * @param idx Index demarcating the polymer atoms to consider. 
         * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
         * @param neighbor_threshold Distance threshold for identifying
         *                           neighboring (non-bonded) atoms. 
         * @param fene_params FENE parameters. 
         * @param angle_mode Angle potential type.  
         * @param angle_params Angle potential parameters. Must include the 
         *                     cosine potential parameters (K and theta0) or
         *                     the dual Gaussian mixture potential parameters
         *                     (A1, A2, w1, w2, theta1, theta2). 
         * @param dihedral_params Dihedral angle potential parameters. 
         * @returns Energy difference due to segment replacement. 
         */
        T getSegmentReplacementEnergyDifference(const Ref<const Matrix<T, Dynamic, 3> >& segment,
                                                const int idx,
                                                std::unordered_map<std::string, T>& lj_params,  
                                                const T neighbor_threshold, 
                                                std::unordered_map<std::string, T>& fene_params,
                                                const AngleMode angle_mode,  
                                                std::unordered_map<std::string, T>& angle_params,
                                                std::unordered_map<std::string, T>& dihedral_params)
        {
            // Get the current segment 
            const int n = segment.rows(); 
            Matrix<T, Dynamic, 3> segment_curr = this->getSegment(idx, n);

            // Get the energy difference 
            T energy_curr = this->getSegmentInteractionEnergy(
                segment_curr, idx, lj_params, neighbor_threshold, fene_params, 
                angle_mode, angle_params, dihedral_params
            ); 
            T energy_new = this->getSegmentInteractionEnergy(
                segment, idx, lj_params, neighbor_threshold, fene_params, 
                angle_mode, angle_params, dihedral_params
            ); 
            return energy_new - energy_curr; 
        } 

        /**
         * Get the Metropolis-Hastings acceptance probability of switching
         * in the given segment of atoms into the polymer at the given index.  
         *
         * @param segment Input segment. 
         * @param idx Index demarcating the polymer atoms to consider. 
         * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
         * @param neighbor_threshold Distance threshold for identifying
         *                           neighboring (non-bonded) atoms. 
         * @param fene_params FENE parameters. 
         * @param angle_mode Angle potential type.  
         * @param angle_params Angle potential parameters. Must include the 
         *                     cosine potential parameters (K and theta0) or
         *                     the dual Gaussian mixture potential parameters
         *                     (A1, A2, w1, w2, theta1, theta2). 
         * @param dihedral_params Dihedral angle potential parameters. 
         * @returns Metropolis-Hastings acceptance probability of switching
         *          in the given segment into the polymer.  
         */
        T getMetropolisAcceptanceProb(const Ref<const Matrix<T, Dynamic, 3> >& segment,
                                      const int idx,
                                      std::unordered_map<std::string, T>& lj_params,  
                                      const T neighbor_threshold, 
                                      std::unordered_map<std::string, T>& fene_params,
                                      const AngleMode angle_mode, 
                                      std::unordered_map<std::string, T>& angle_params,
                                      std::unordered_map<std::string, T>& dihedral_params) const
        {
            // Get the energy of the current polymer configuration 
            const int n = segment.rows(); 
            Matrix<T, Dynamic, 3> segment_curr = this->r(Eigen::seqN(idx, n), Eigen::all);
            const T energy_curr = this->getSegmentInteractionEnergy(
                segment_curr, idx, lj_params, neighbor_threshold, fene_params, 
                angle_mode, angle_params, dihedral_params
            ); 

            // Get the energy of the proposed polymer configuration 
            const T energy_new = this->getSegmentInteractionEnergy(
                segment, idx, lj_params, neighbor_threshold, fene_params, 
                angle_mode, angle_params, dihedral_params
            );

            // Calculate the Metropolis acceptance probability
            return min(1, exp(-(energy_new - energy_curr) / this->kT));  
        }
};

/**
 * Generate a random K-mer in which the inter-atom distances, bond lengths, 
 * bond angles, and dihedral angles follow the given potentials. 
 *
 * The inter-atom distances also obey a minimum distance criterion.
 *
 * This procedure extends the K-mer one atom at a time from a given position
 * for the 0-th atom (r0), by sampling bond lengths, angles, and dihedrals 
 * and testing whether the new atom is not too close to the existing atoms. 
 * If this sampling fails to generate a new atomic position within a given 
 * number of attempts (max_tries_per_atom), the procedure backtracks by 
 * deleting the previous atom and generating a new position for that atom.
 * The procedure terminates prematurely if it exceeds the maximum number of 
 * backtracks (max_n_backtracks).  
 *
 * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
 * @param neighbor_threshold Distance threshold for identifying
 *                           neighboring (non-bonded) atoms. 
 * @param fene_params FENE parameters. 
 * @param angle_mode Angle potential type.  
 * @param angle_params Angle potential parameters. Must include the 
 *                     cosine potential parameters (K and theta0) or
 *                     the dual Gaussian mixture potential parameters
 *                     (A1, A2, w1, w2, theta1, theta2). 
 * @param dihedral_params Dihedral angle potential parameters. 
 * @param r0 Position of 0-th atom. 
 * @param collision_threshold Distance threshold for identifying atoms that 
 *                            are too close to each other. 
 * @param max_tries_per_atom Maximum number of attempts to place each atom
 *                           before backtracking. 
 * @param max_n_backtracks Maximum number of backtracks. 
 * @param rng Random number generator. 
 * @param uniform_dist Pre-defined instance of standard uniform distribution. 
 * @param units Units for keeping track of Boltzmann's constant. 
 * @param temp Temperature (in Kelvin). 
 * @returns Resulting polymer configuration.  
 */
template <typename T, size_t K>
PolymerConfiguration<T> generateKMer(std::unordered_map<std::string, T>& lj_params,
                                     std::unordered_map<std::string, T>& fene_params,
                                     const AngleMode angle_mode,  
                                     std::unordered_map<std::string, T>& angle_params, 
                                     std::unordered_map<std::string, T>& dihedral_params,
                                     const Ref<const Matrix<T, 3, 1> >& r0,
                                     const T collision_threshold, 
                                     const int max_tries_per_atom,
                                     const int max_n_backtracks,  
                                     boost::random::mt19937& rng,
                                     boost::random::uniform_01<>& uniform_dist,
                                     const Units units = Units::NANO,
                                     const T temp = 300)
{
    const T kT = (
        units == Units::MICRO ? static_cast<T>(1.380649e-8) * temp : 
        static_cast<T>(1.380649e-2) * temp
    ); 

    // Define the angle sampling function  
    std::function<T(boost::random::mt19937&)> sample_angle;
    if (angle_mode == AngleMode::COSINE)
    {
        sample_angle = [&angle_params, &uniform_dist, &kT](boost::random::mt19937& rng_) -> T
        {
            return sampleAngleCosine<T>(
                angle_params["K"], angle_params["theta0"], kT, rng_, 
                uniform_dist, 50
            );
        };
    } 
    else if (angle_mode == AngleMode::GAUSSIAN)
    {
        sample_angle = [&angle_params, &uniform_dist, &kT](boost::random::mt19937& rng_) -> T
        {
            return sampleAngleDualGaussianMixture<T>(
                angle_params["A1"], angle_params["A2"], angle_params["w1"],
                angle_params["w2"], angle_params["theta1"], angle_params["theta2"],
                kT, rng_, uniform_dist, 50
            );
        };
    }
    else 
    {
        throw std::runtime_error("Invalid angle potential mode specified"); 
    }

    // Sample an initial bond length 
    T length = sampleFene<T>(
        lj_params["eps"], lj_params["sigma"], fene_params["K"],
        fene_params["R0"], kT, rng, uniform_dist, 50 
    );
    T angle, dihedral;  

    // Generate a PolymerConfiguration<T> instance with the first 2 atoms 
    Matrix<T, Dynamic, 3> coords(K, 3); 
    coords.row(0) = r0; 
    coords(1, 0) = length; 
    coords(1, 1) = 0; 
    coords(1, 2) = 0;
    PolymerConfiguration<T> config(coords(Eigen::seqN(0, 2), Eigen::all), units, temp); 

    // Define a collision function 
    auto collision = [&collision_threshold](PolymerConfiguration<T>& config, const Ref<const Matrix<T, 3, 1> >& r) -> bool
    {
        return (config.getMinDist(r) < collision_threshold);
    };  

    // Add a 3rd atom ...
    //
    // Keep generating a new atom until no collision is detected 
    Matrix<T, 3, 1> new_atom;
    bool found_collision = true;  
    while (found_collision)
    {
        length = sampleFene<T>(
            lj_params["eps"], lj_params["sigma"], fene_params["K"],
            fene_params["R0"], config.kT, rng, uniform_dist, 50 
        );
        angle = sample_angle(rng); 
        new_atom = generateNextAtom<T>(
            coords.row(0), coords.row(1), length, angle, rng, uniform_dist
        );
        found_collision = collision(config, new_atom);  
    }
    coords.row(2) = new_atom; 
    config.appendAtomToTail(new_atom); 

    // Add the remaining atoms ...
    int curr_idx = 3;
    int n_backtracks = 0;  
    while (curr_idx < K)
    {
        Matrix<T, 3, 1> r1 = coords.row(curr_idx - 3); 
        Matrix<T, 3, 1> r2 = coords.row(curr_idx - 2); 
        Matrix<T, 3, 1> r3 = coords.row(curr_idx - 1); 

        // Keep generating a new atom until no collision is detected or 
        // the maximum number of iterations is reached  
        int n_tries = 0;
        found_collision = true; 
        while (found_collision && n_tries < max_tries_per_atom)
        { 
            length = sampleFene<T>(
                lj_params["eps"], lj_params["sigma"], fene_params["K"],
                fene_params["R0"], config.kT, rng, uniform_dist, 50 
            );
            angle = sample_angle(rng);
            dihedral = sampleDihedralHarmonic<T>(
                dihedral_params["K"], config.kT, rng, uniform_dist
            );
            new_atom = generateNextAtomDihedral<T>(
                r1, r2, r3, length, angle, dihedral, rng, uniform_dist 
            );
            found_collision = collision(config, new_atom); 
            n_tries++; 
        }

        // If the maximum number of iterations has been reached, move onto
        // the next atom 
        if (!found_collision)
        {
            coords.row(curr_idx) = new_atom; 
            config.appendAtomToTail(new_atom);
            curr_idx++;
        } 
        // Otherwise, backtrack to the previous atom unless doing so
        // encroaches into the first 3 atoms 
        else if (curr_idx > 3) 
        {
            config.popAtomFromTail(); 
            curr_idx--; 
            n_backtracks++;  
        }
        else
        {
            throw std::runtime_error(
                "Sampling procedure backtracked into first 3 atoms; try "
                "sampling more positions per atom"
            ); 
        }

        // If we have exceeded the maximum number of backtracks, raise 
        // an exception 
        if (n_backtracks > max_n_backtracks)
        {
            throw std::runtime_error(
                "Sampling procedure exceeded maximum number of backtracks; try "
                "sampling more positions per atom"
            );
        } 
    }

    return config;  
}

#endif
