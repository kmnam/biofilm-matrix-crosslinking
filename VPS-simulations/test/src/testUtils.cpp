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
#include <boost/math/special_functions/bessel.hpp>
#include <boost/random.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/utils.hpp"

using namespace Eigen;

/**
 * Construct an empirical PDF of the form, 
 *
 * \sin{\theta} * P(\theta), 
 *
 * where P is a von Mises distribution of the given mean and concentration. 
 */
Matrix<double, Dynamic, 2> vonMisesTransformedPDF(const double mu,
                                                  const double kappa, 
                                                  const int n_mesh)
{
    Matrix<double, Dynamic, 2> pdf = Matrix<double, Dynamic, 2>::Zero(n_mesh, 2); 
    pdf.col(0) = Matrix<double, Dynamic, 1>::LinSpaced(
        n_mesh, -boost::math::constants::pi<double>(),
        boost::math::constants::pi<double>()
    );  
    for (int i = 0; i < n_mesh; ++i)
    {
        // No need to calculate the denominator, as we will normalize anyway
        pdf(i, 1) = sin(pdf(i, 0)) * exp(kappa * cos(pdf(i, 0) - mu)); 
    }

    // Now integrate over the mesh, using the midpoint rule
    double integral = 0;
    double dx = pdf(1, 0) - pdf(0, 0);    // Mesh is uniform  
    for (int i = 1; i < n_mesh; ++i)
        integral += dx * (pdf(i - 1, 1) + pdf(i, 1)) / 2; 

    // Normalize by the integral and return 
    pdf.col(1) /= integral;

    return pdf;  
}

/**
 * Construct an empirical PDF of the form, 
 *
 * \sin{\theta} * (A1 * P1(\theta) + A2 * P2(\theta)), 
 *
 * where P1 and P2 are von Mises distributions with the given means and
 * concentrations, and A1 and A2 are weights. 
 */
Matrix<double, Dynamic, 2> dualVonMisesMixtureTransformedPDF(const double mu1,
                                                             const double kappa1,
                                                             const double A1, 
                                                             const double mu2, 
                                                             const double kappa2, 
                                                             const double A2,  
                                                             const int n_mesh)
{
    Matrix<double, Dynamic, 2> pdf = Matrix<double, Dynamic, 2>::Zero(n_mesh, 2); 
    pdf.col(0) = Matrix<double, Dynamic, 1>::LinSpaced(
        n_mesh, -boost::math::constants::pi<double>(),
        boost::math::constants::pi<double>()
    );  
    for (int i = 0; i < n_mesh; ++i)
    {
        // No need to calculate the denominator, as we will normalize anyway
        pdf(i, 1) = sin(pdf(i, 0)) * (
            A1 * exp(kappa1 * cos(pdf(i, 0) - mu1)) +
            A2 * exp(kappa2 * cos(pdf(i, 0) - mu2))
        );
    }

    // Now integrate over the mesh, using the midpoint rule
    double integral = 0;
    double dx = pdf(1, 0) - pdf(0, 0);    // Mesh is uniform  
    for (int i = 1; i < n_mesh; ++i)
        integral += dx * (pdf(i - 1, 1) + pdf(i, 1)) / 2; 

    // Normalize by the integral and return 
    pdf.col(1) /= integral;

    return pdf;  
}

/**
 * Calculate the circular mean of an empirical circular distribution.
 *
 * The mesh over which the empirical distribution is defined is assumed to be
 * uniform.  
 */
double getCircularMeanFromPDF(const Ref<const Matrix<double, Dynamic, 2> >& pdf)
{
    double dx = pdf(1, 0) - pdf(0, 0); 
    double c = (pdf.col(0).array().cos() * pdf.col(1).array()).sum() * dx;
    double s = (pdf.col(0).array().sin() * pdf.col(1).array()).sum() * dx; 
    return atan2(s, c);  
}

/**
 * Calculate the circular variance of an empirical circular distribution.
 *
 * The mesh over which the empirical distribution is defined is assumed to be
 * uniform.  
 */
double getCircularVarianceFromPDF(const Ref<const Matrix<double, Dynamic, 2> >& pdf)
{
    double dx = pdf(1, 0) - pdf(0, 0);
    double c = (pdf.col(0).array().cos() * pdf.col(1).array()).sum() * dx;
    double s = (pdf.col(0).array().sin() * pdf.col(1).array()).sum() * dx; 
    return 1.0 - sqrt(c * c + s * s); 
}

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
            theta, phi
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
        REQUIRE_THAT(dihedral, Catch::Matchers::WithinAbs(dihedrals(i), tol)); 
    }
}

/**
 * Tests for sampleFene(), sampleAngleCosine(), sampleAngleDualGaussianMixture(),
 * sampleDihedralHarmonic(), and sampleDihedralFourierSingleComponent().  
 *
 * These tests involve a mix of assertions and statistical comparisons between 
 * empirical and theoretically expected values.  
 */
TEST_CASE(
    "Tests for bond length, bond angle, and dihedral angle sampling", 
    "[sampleFene(), sampleAngleCosine(), sampleAngleDualGaussianMixture(), sampleDihedralHarmonic(), sampleDihedralFourierSingleComponent()]"
)
{
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;
    const int n = 1000000; 

    // ----------------------------------------------------------------- //
    // Tests for sampleFene()
    // ----------------------------------------------------------------- //
    // Sample bond lengths ...
    const double kT = 1.380649e-2 * 300;     // Use nano units 
    const double eps = kT;
    const double sigma = 0.9;  
    const double R0 = 1.5;
    const double K = 10.0 * kT;
    Matrix<double, Dynamic, 2> bond_length_cdf = getFeneCDF<double>(
        eps, sigma, K, R0, kT, 10000, 1e-6
    ); 
    Array<double, Dynamic, 1> sample_lengths(n); 
    for (int i = 0; i < n; ++i) 
        sample_lengths(i) = sampleFene<double>(rng, uniform_dist, bond_length_cdf);

    // Check that the bond lengths are within (0, R0)
    REQUIRE((sample_lengths > 0).all());
    REQUIRE((sample_lengths < R0).all());

    // Compare against the FENE CDF
    double fene_mean_theoretical = 0;  
    for (int i = 0; i < bond_length_cdf.rows() - 1; ++i)
    {
        double dx = bond_length_cdf(i + 1, 0) - bond_length_cdf(i, 0); 
        fene_mean_theoretical += dx * (1 - bond_length_cdf(i, 1)); 
    }
    std::cout << "Empirical vs. theoretical means from angles (FENE potential): "
              << sample_lengths.mean() << ", " << fene_mean_theoretical << std::endl; 

    // ----------------------------------------------------------------- //
    // Tests for sampleAngleCosine()
    // ----------------------------------------------------------------- //
    // Sample bond angles according to the cosine potential ... 
    const double K_angle = 20 * kT;  
    const double theta0 = 140 * boost::math::constants::pi<double>() / 180;
    Array<double, Dynamic, 1> sample_angles_cosine(n); 
    for (int i = 0; i < n; ++i) 
        sample_angles_cosine(i) = sampleAngleCosine<double>(
            K_angle, theta0, kT, rng, uniform_dist
        );

    // Check that the bond angles are within [0, \pi)
    REQUIRE((sample_angles_cosine >= 0).all()); 
    REQUIRE((sample_angles_cosine < boost::math::constants::pi<double>()).all()); 

    // Estimate the empirical circular mean of the bond angles 
    double mean_cosine_empirical = getCircularMean<double>(sample_angles_cosine);

    // Get the theoretical circular mean by first defining the underlying 
    // distribution
    Matrix<double, Dynamic, 2> cosine_pdf = vonMisesTransformedPDF(
        theta0, K_angle / kT, 10000
    );
    double mean_cosine_theoretical = getCircularMeanFromPDF(cosine_pdf);  
    std::cout << "Empirical vs. theoretical means from angles (cosine potential): "
              << mean_cosine_empirical << ", " << mean_cosine_theoretical
              << std::endl; 

    // Estimate the empirical circular variance of the bond angles
    double var_cosine_empirical = getCircularVariance<double>(sample_angles_cosine);

    // Get the theoretical circular variance 
    double var_cosine_theoretical = getCircularVarianceFromPDF(cosine_pdf); 
    std::cout << "Empirical vs. theoretical variances from angles (cosine potential): "
              << var_cosine_empirical << ", " << var_cosine_theoretical << std::endl; 

    // ----------------------------------------------------------------- //
    // Tests for sampleAngleDualGaussianMixture()
    // ----------------------------------------------------------------- //
    // Sample bond angles according to the dual Gaussian mixture potential ...
    //
    // Here, use a calibrated set of weights and concentrations, to account 
    // for the geometric prefactor in the sampling distribution 
    double w1 = 2 / sqrt(20);     // Concentration = 20 ==> standard deviation = sqrt(1/20)
    double w2 = 2 / sqrt(20);  
    double A1 = 0.962103305;      // This should ideally yield a 90%/10% mixture
    double A2 = 1 - A1; 
    const double theta1 = 160 * boost::math::constants::pi<double>() / 180;
    const double theta2 = boost::math::constants::half_pi<double>(); 
    Array<double, Dynamic, 1> sample_angles_gaussian(n); 
    for (int i = 0; i < n; ++i)
        sample_angles_gaussian(i) = sampleAngleDualGaussianMixture<double>(
            A1, A2, w1, w2, theta1, theta2, kT, rng, uniform_dist
        );

    // Check that the bond angles are within [0, \pi)
    REQUIRE((sample_angles_gaussian >= 0).all()); 
    REQUIRE((sample_angles_gaussian < boost::math::constants::pi<double>()).all());

    // Estimate the empirical circular mean of the bond angles 
    double mean_gaussian_empirical = getCircularMean<double>(sample_angles_gaussian);

    // Compare this against the theoretical circular mean
    Matrix<double, Dynamic, 2> gaussian_pdf = dualVonMisesMixtureTransformedPDF(
        theta1, 4 / (w1 * w1), A1, theta2, 4 / (w2 * w2), A2, 10000
    );
    double mean_gaussian_theoretical = getCircularMeanFromPDF(gaussian_pdf); 
    std::cout << "Empirical vs. theoretical means from angles (Gaussian potential): "
              << mean_gaussian_empirical << ", " << mean_gaussian_theoretical
              << std::endl;

    // Estimate the empirical circular variance of the bond angles
    double var_gaussian_empirical = getCircularVariance<double>(sample_angles_gaussian);

    // Get the theoretical circular variance 
    double var_gaussian_theoretical = getCircularVarianceFromPDF(gaussian_pdf); 
    std::cout << "Empirical vs. theoretical variances from angles (Gaussian potential): "
              << var_gaussian_empirical << ", " << var_gaussian_theoretical << std::endl; 

    // Sample from just one Gaussian component (mean = theta2 = \pi/2)
    A1 = 0.0; 
    A2 = 1.0; 
    for (int i = 0; i < n; ++i)
        sample_angles_gaussian(i) = sampleAngleDualGaussianMixture<double>(
            A1, A2, w1, w2, theta1, theta2, kT, rng, uniform_dist
        );

    // Check that the bond angles are within [0, \pi)
    REQUIRE((sample_angles_gaussian >= 0).all()); 
    REQUIRE((sample_angles_gaussian < boost::math::constants::pi<double>()).all());

    // Estimate the empirical circular mean of the bond angles 
    mean_gaussian_empirical = getCircularMean<double>(sample_angles_gaussian);

    // Compare this against the theoretical circular mean
    gaussian_pdf = dualVonMisesMixtureTransformedPDF(
        theta1, 4 / (w1 * w1), A1, theta2, 4 / (w2 * w2), A2, 10000
    );
    mean_gaussian_theoretical = getCircularMeanFromPDF(gaussian_pdf); 
    std::cout << "Empirical vs. theoretical means from angles (Gaussian potential with 1 component): "
              << mean_gaussian_empirical << ", " << mean_gaussian_theoretical
              << std::endl;

    // Estimate the empirical circular variance of the bond angles
    var_gaussian_empirical = getCircularVariance<double>(sample_angles_gaussian);

    // Get the theoretical circular variance 
    var_gaussian_theoretical = getCircularVarianceFromPDF(gaussian_pdf); 
    std::cout << "Empirical vs. theoretical variances from angles (Gaussian potential with 1 component): "
              << var_gaussian_empirical << ", " << var_gaussian_theoretical << std::endl; 
 
    // ----------------------------------------------------------------- //
    // Tests for sampleDihedralHarmonic()
    // ----------------------------------------------------------------- //
    // Sample dihedral angles ... 
    const double K_dihedral = 0.7 * kT; 
    Array<double, Dynamic, 1> sample_dihedrals(n); 
    for (int i = 0; i < n; ++i) 
        sample_dihedrals(i) = sampleDihedralHarmonic<double>(
            K_dihedral, kT, rng, uniform_dist
        );

    // Check that the dihedral angles are within [-\pi, \pi)
    REQUIRE((sample_dihedrals >= -boost::math::constants::pi<double>()).all()); 
    REQUIRE((sample_dihedrals <= boost::math::constants::pi<double>()).all()); 

    // Estimate the empirical circular mean of the dihedral angles 
    double mean_dihedral_empirical = getCircularMean<double>(sample_dihedrals); 

    // Compare this against the theoretical circular mean
    std::cout << "Empirical vs. theoretical means from dihedrals (harmonic potential): "
              << mean_dihedral_empirical << ", " << boost::math::constants::pi<double>()
              << std::endl;

    // Estimate the empirical circular variance of the dihedral angles 
    double var_dihedral_empirical = getCircularVariance<double>(sample_dihedrals); 

    // Compare this against the theoretical circular variance
    double i1k = boost::math::cyl_bessel_i(1, K_dihedral / kT);
    double i0k = boost::math::cyl_bessel_i(0, K_dihedral / kT);  
    double var_dihedral_theoretical = 1 - i1k / i0k; 
    std::cout << "Empirical vs. theoretical variances from dihedrals (harmonic potential): "
              << var_dihedral_empirical << ", " << var_dihedral_theoretical
              << std::endl;  

    // ----------------------------------------------------------------- //
    // Tests for sampleDihedralFourierSingleComponent()
    // ----------------------------------------------------------------- //
    // Sample dihedral angles with an equilibrium value of 150 degrees ... 
    sample_dihedrals = Array<double, Dynamic, 1>::Zero(n); 
    for (int i = 0; i < n; ++i) 
        sample_dihedrals(i) = sampleDihedralFourierSingleComponent<double>(
            K_dihedral, -30 * boost::math::constants::pi<double>() / 180.0, kT,
            rng, uniform_dist 
        );

    // Check that the dihedral angles are within [-\pi, \pi)
    REQUIRE((sample_dihedrals >= -boost::math::constants::pi<double>()).all()); 
    REQUIRE((sample_dihedrals <= boost::math::constants::pi<double>()).all()); 

    // Estimate the empirical circular mean of the dihedral angles 
    mean_dihedral_empirical = getCircularMean<double>(sample_dihedrals);

    // Compare this against the theoretical circular mean
    std::cout << "Empirical vs. theoretical means from dihedrals (Fourier potential): "
              << mean_dihedral_empirical << ", "
              << 150 * boost::math::constants::pi<double>() / 180.0
              << std::endl;

    // Estimate the empirical circular variance of the dihedral angles 
    var_dihedral_empirical = getCircularVariance<double>(sample_dihedrals); 

    // Compare this against the theoretical circular variance
    i1k = boost::math::cyl_bessel_i(1, K_dihedral / kT);
    i0k = boost::math::cyl_bessel_i(0, K_dihedral / kT);  
    var_dihedral_theoretical = 1 - i1k / i0k; 
    std::cout << "Empirical vs. theoretical variances from dihedrals (harmonic potential): "
              << var_dihedral_empirical << ", " << var_dihedral_theoretical
              << std::endl;  
}

