/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     1/30/2026
 */

#include <iostream>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/random.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/utils.hpp"

using namespace Eigen;

/**
 * Tests for getDihedral(), generateNextAtom(), and generateNextAtomDihedral().  
 */
TEST_CASE(
    "Tests for dihedral angle calculation and next-atom generation",
    "[getDihedral(), generateNextAtom(), generateNextAtomDihedral()]"
)
{
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;
    const double tol = 1e-5;  

    // Generate an 8-atom segment with no fixed dihedral angles  
    const double length = 1.5; 
    Array<double, 6, 1> angles;
    angles << 140 * boost::math::constants::pi<double>() / 180,
              170 * boost::math::constants::pi<double>() / 180,  
              160 * boost::math::constants::pi<double>() / 180,
              boost::math::constants::half_pi<double>(), 
              150 * boost::math::constants::pi<double>() / 180,
              130 * boost::math::constants::pi<double>() / 180;
    Matrix<double, 8, 3> coords; 
    coords.row(0) = Matrix<double, 3, 1>::Zero();
    coords(1, 0) = length; 
    coords(1, 1) = 0; 
    coords(1, 2) = 0;
    for (int i = 2; i < 8; ++i) 
        coords.row(i) = generateNextAtom<double>(
            coords.row(i - 2), coords.row(i - 1), length, angles(i - 2), rng,
            uniform_dist
        );

    // Check the bond lengths 
    for (int i = 0; i < coords.rows() - 1; ++i)
        REQUIRE_THAT(
            (coords.row(i + 1) - coords.row(i)).norm(),
            Catch::Matchers::WithinAbs(length, tol)
        );

    // Check the bond angles 
    for (int i = 0; i < coords.rows() - 2; ++i)
    {
        Matrix<double, 3, 1> u = coords.row(i + 2) - coords.row(i + 1); 
        Matrix<double, 3, 1> v = coords.row(i) - coords.row(i + 1); 
        REQUIRE_THAT(
            u.dot(v) / (u.norm() * v.norm()), 
            Catch::Matchers::WithinAbs(cos(angles(i)), tol)
        );
    }

    // Generate an 8-atom segment with fixed dihedral angles
    Array<double, 5, 1> dihedrals;
    dihedrals << 60 * boost::math::constants::pi<double>() / 180, 
                 -60 * boost::math::constants::pi<double>() / 180,
                 boost::math::constants::pi<double>(),  
                 120 * boost::math::constants::pi<double>() / 180, 
                 60 * boost::math::constants::pi<double>() / 180;
    coords.row(0) = Matrix<double, 3, 1>::Zero();
    coords(1, 0) = length; 
    coords(1, 1) = 0; 
    coords(1, 2) = 0;
    coords.row(2) = generateNextAtom<double>(
        coords.row(0), coords.row(1), length, angles(0), rng, uniform_dist
    ); 
    for (int i = 3; i < 8; ++i) 
    {
        double theta = angles(i - 2); 
        double phi = dihedrals(i - 3); 
        coords.row(i) = generateNextAtomDihedral<double>(
            coords.row(i - 3), coords.row(i - 2), coords.row(i - 1), length,
            theta, phi, rng, uniform_dist, (phi > 0 ? 1 : -1)
        );
    }

    // Check the bond lengths 
    for (int i = 0; i < coords.rows() - 1; ++i)
        REQUIRE_THAT(
            (coords.row(i + 1) - coords.row(i)).norm(),
            Catch::Matchers::WithinAbs(length, tol)
        );

    // Check the bond angles 
    for (int i = 0; i < coords.rows() - 2; ++i)
    {
        Matrix<double, 3, 1> u = coords.row(i + 2) - coords.row(i + 1); 
        Matrix<double, 3, 1> v = coords.row(i) - coords.row(i + 1); 
        REQUIRE_THAT(
            u.dot(v) / (u.norm() * v.norm()), 
            Catch::Matchers::WithinAbs(cos(angles(i)), tol)
        );
    }

    // Check the dihedral angles
    for (int i = 0; i < coords.rows() - 3; ++i)
    {
        double dihedral = getDihedral<double>(
            coords.row(i), coords.row(i + 1), coords.row(i + 2), coords.row(i + 3)
        );
        REQUIRE_THAT(abs(dihedral), Catch::Matchers::WithinAbs(abs(dihedrals(i)), tol));  
    }
}

/**
 * Tests for sampleFene(), sampleAngleCosine(), sampleAngleDualGaussianMixture(),
 * and sampleDihedralHarmonic(). 
 *
 * These tests involve a mix of assertions and statistical comparisons between 
 * empirical and theoretically expected values.  
 */
TEST_CASE(
    "Tests for bond length, bond angle, and dihedral angle sampling", 
    "[sampleFene(), sampleAngleCosine(), sampleAngleDualGaussianMixture(), sampleDihedralHarmonic()]"
)
{
    // Sample bond lengths ... 
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;
    const int n = 2000; 
    const double kT = 1.380649e-2 * 300; 
    const double eps = kT;
    const double sigma = 0.9;  
    const double R0 = 1.5;
    const double K = 10.0 * kT; 
    Array<double, Dynamic, 1> sample_lengths(n); 
    for (int i = 0; i < n; ++i) 
        sample_lengths(i) = sampleFene<double>(eps, sigma, K, R0, kT, rng, uniform_dist, 50);

    // Check that the bond lengths are within (0, R0)
    REQUIRE((sample_lengths > 0).all());
    REQUIRE((sample_lengths < R0).all()); 

    // Sample bond angles according to the cosine potential ... 
    const double K_angle = 20 * kT;  
    const double theta0 = 140 * boost::math::constants::pi<double>() / 180;
    Array<double, Dynamic, 1> sample_angles_cosine(n); 
    for (int i = 0; i < n; ++i) 
        sample_angles_cosine(i) = sampleAngleCosine<double>(
            K_angle, theta0, kT, rng, uniform_dist, 50
        );

    // Check that the bond angles are within [0, \pi)
    REQUIRE((sample_angles_cosine >= 0).all()); 
    REQUIRE((sample_angles_cosine < boost::math::constants::pi<double>()).all()); 

    // Estimate the mean of the underlying von Mises distribution
    double mean_vonmises = 0; 
    double denom = 0; 
    for (int i = 0; i < sample_angles_cosine.size(); ++i)
    {
        // All angles should be in [0, \pi), so sin(theta) is positive
        double theta = sample_angles_cosine(i);
        mean_vonmises += theta / sin(theta); 
        denom += 1.0 / sin(theta); 
    }
    mean_vonmises /= denom; 
    std::cout << "Empirical vs. theoretical means from angles (cosine potential): "
              << mean_vonmises << ", " << theta0 << std::endl;

    // Estimate the variance of the underlying von Mises distribution
    // (= 1 / concentration)
    //
    // First calculate sin(theta) for each angle theta
    Matrix<double, Dynamic, 1> weights(sample_angles_cosine.size()); 
    for (int i = 0; i < sample_angles_cosine.size(); ++i)
    {
        // All angles should be in [0, \pi), so sin(theta) is positive
        double theta = sample_angles_cosine(i);
        weights(i) = 1.0 / sin(theta); 
    }

    // Normalize the weights and calculate the effective sample size 
    double total_weight = weights.sum();
    weights /= total_weight; 
    double effective_size = 1.0 / weights.dot(weights);

    // Then estimate the variance 
    double var_vonmises = 0;  
    for (int i = 0; i < sample_angles_cosine.size(); ++i)
    {
        double theta = sample_angles_cosine(i);
        double diff = theta - mean_vonmises; 
        var_vonmises += weights(i) * diff * diff; 
    }
    var_vonmises *= (effective_size / (effective_size - 1));
    std::cout << "Empirical vs. theoretical variances from angles (cosine potential): "
              << var_vonmises << ", " << kT / K_angle << std::endl;  

    // Sample bond angles according to the dual Gaussian mixture potential ... 
    double A1 = 0.7; 
    double A2 = 0.3; 
    const double w1 = sqrt(1 / (10 * kT));
    const double w2 = sqrt(1 / (8 * kT));
    const double theta1 = 160 * boost::math::constants::pi<double>() / 180;
    const double theta2 = boost::math::constants::half_pi<double>(); 
    Array<double, Dynamic, 1> sample_angles_gaussian(n); 
    for (int i = 0; i < n; ++i) 
        sample_angles_gaussian(i) = sampleAngleDualGaussianMixture<double>(
            A1, A2, w1, w2, theta1, theta2, kT, rng, uniform_dist, 50
        );

    // Check that the bond angles are within [0, \pi)
    REQUIRE((sample_angles_gaussian >= 0).all()); 
    REQUIRE((sample_angles_gaussian < boost::math::constants::pi<double>()).all()); 

    // Sample from just one Gaussian component
    A1 = 0.0; 
    A2 = 1.0; 
    for (int i = 0; i < n; ++i) 
        sample_angles_gaussian(i) = sampleAngleDualGaussianMixture<double>(
            A1, A2, w1, w2, theta1, theta2, kT, rng, uniform_dist, 50
        );

    // Divide each sampled angle by sin(theta) and check the mean 
    double mean_gaussian = 0;
    denom = 0; 
    for (int i = 0; i < sample_angles_gaussian.size(); ++i)
    {
        double theta = sample_angles_gaussian(i); 
        mean_gaussian += theta / sin(theta); 
        denom += 1.0 / sin(theta); 
    }
    mean_gaussian /= denom; 
    std::cout << "Empirical vs. theoretical means from angles (Gaussian potential): "
              << mean_gaussian << ", " << theta2 << std::endl;

    // Estimate the variance of the underlying von Mises distribution
    // (= 1 / concentration)
    //
    // First calculate sin(theta) for each angle theta
    for (int i = 0; i < sample_angles_gaussian.size(); ++i)
    {
        // All angles should be in [0, \pi), so sin(theta) is positive
        double theta = sample_angles_gaussian(i);
        weights(i) = 1.0 / sin(theta); 
    }

    // Normalize the weights and calculate the effective sample size 
    total_weight = weights.sum();
    weights /= total_weight; 
    effective_size = 1.0 / weights.dot(weights);

    // Then estimate the variance
    double var_gaussian = 0;  
    for (int i = 0; i < sample_angles_gaussian.size(); ++i)
    {
        double theta = sample_angles_gaussian(i);
        double diff = theta - mean_gaussian; 
        var_gaussian += weights(i) * diff * diff; 
    }
    var_gaussian *= (effective_size / (effective_size - 1));
    std::cout << "Empirical vs. theoretical variances from angles (Gaussian potential): "
              << var_gaussian << ", " << w2 * w2 << std::endl; 
   
    // Sample dihedral angles ... 
    const double K_dihedral = 10 * kT; 
    Array<double, Dynamic, 1> sample_dihedrals(n); 
    for (int i = 0; i < n; ++i) 
        sample_dihedrals(i) = sampleDihedralHarmonic<double>(
            K_dihedral, kT, rng, uniform_dist
        );

    // Check that the dihedral angles are within [-\pi, \pi)
    REQUIRE((sample_dihedrals >= -boost::math::constants::pi<double>()).all()); 
    REQUIRE((sample_dihedrals < boost::math::constants::pi<double>()).all()); 

    // Estimate the mean of the underlying von Mises distribution
    double mean_dihedral = 0; 
    for (int i = 0; i < sample_dihedrals.size(); ++i)
    {
        // If the angle is between -\pi and 0, add 2*\pi plus the angle
        if (sample_dihedrals(i) < 0)
            mean_dihedral += boost::math::constants::two_pi<double>() + sample_dihedrals(i); 
        else 
            mean_dihedral += sample_dihedrals(i); 
    }
    mean_dihedral /= sample_dihedrals.size(); 
    std::cout << "Empirical vs. theoretical means from dihedrals: " 
              << mean_dihedral << ", " << boost::math::constants::pi<double>() << std::endl;

    // Estimate the variance of the underlying von Mises distribution
    // (= 1 / concentration)
    double var_dihedral = 0; 
    for (int i = 0; i < sample_dihedrals.size(); ++i)
    {
        // If the angle is between -\pi and 0, add 2*\pi plus the angle
        double theta; 
        if (sample_dihedrals(i) < 0)
            theta = boost::math::constants::two_pi<double>() + sample_dihedrals(i); 
        else 
            theta = sample_dihedrals(i);
        double diff = theta - mean_dihedral; 
        var_dihedral += (diff * diff);  
    }
    var_dihedral /= (sample_dihedrals.size() - 1); 
    std::cout << "Empirical vs. theoretical variances from dihedrals: "
              << var_dihedral << ", " << kT / K_dihedral << std::endl;  
}

