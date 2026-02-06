/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     2/6/2026
 */

#include <iostream>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/random.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/cbmc.hpp"

using namespace Eigen;

/**
 * Tests for getJacobian(). 
 */
TEST_CASE("Tests for Jacobian calculation", "[getJacobian()]")
{
    const double dx = 1e-8; 
    const double tol = 1e-8; 

    // Define a multivariable function that converts spherical coordinates to 
    // Cartesian coordinates
    DynamicVectorValuedFunction<double> F = [](const Ref<const Matrix<double, Dynamic, 1> >& x)
        -> Matrix<double, Dynamic, 1>
    {
        Matrix<double, Dynamic, 1> f(3);
        f << x(0) * sin(x(1)) * cos(x(2)),     // x = [r, \theta, \phi]
             x(0) * sin(x(1)) * sin(x(2)),
             x(0) * cos(x(1));

        return f;  
    };

    // Try calculating the Jacobian at multiple values of r, \theta, \phi
    Matrix<double, Dynamic, 1> r(5), theta(5), phi(5); 
    r << 0.1, 0.2, 0.3, 0.4, 0.5; 
    theta << boost::math::constants::sixth_pi<double>(), 
             boost::math::constants::third_pi<double>(), 
             boost::math::constants::half_pi<double>(), 
             boost::math::constants::two_thirds_pi<double>(), 
             boost::math::constants::pi<double>();
    phi << 1.2 * boost::math::constants::sixth_pi<double>(), 
           1.1 * boost::math::constants::third_pi<double>(), 
           0.9 * boost::math::constants::half_pi<double>(), 
           0.8 * boost::math::constants::two_thirds_pi<double>(), 
           0.6 * boost::math::constants::pi<double>();
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            for (int k = 0; k < 5; ++k)
            { 
                Matrix<double, Dynamic, 1> x0(3);
                x0 << r(i), theta(j), phi(k);  
                Matrix<double, Dynamic, Dynamic> J = getJacobian<double>(F, x0, dx);
                REQUIRE(J.rows() == 3); 
                REQUIRE(J.cols() == 3); 
                REQUIRE_THAT(
                    J(0, 0),
                    Catch::Matchers::WithinAbs(sin(theta(j)) * cos(phi(k)), tol)
                );
                REQUIRE_THAT(
                    J(0, 1),
                    Catch::Matchers::WithinAbs(r(i) * cos(theta(j)) * cos(phi(k)), tol)
                ); 
                REQUIRE_THAT(
                    J(0, 2),
                    Catch::Matchers::WithinAbs(-r(i) * sin(theta(j)) * sin(phi(k)), tol)
                ); 
                REQUIRE_THAT(
                    J(1, 0),
                    Catch::Matchers::WithinAbs(sin(theta(j)) * sin(phi(k)), tol)
                ); 
                REQUIRE_THAT(
                    J(1, 1),
                    Catch::Matchers::WithinAbs(r(i) * cos(theta(j)) * sin(phi(k)), tol)
                ); 
                REQUIRE_THAT(
                    J(1, 2),
                    Catch::Matchers::WithinAbs(r(i) * sin(theta(j)) * cos(phi(k)), tol)
                ); 
                REQUIRE_THAT(
                    J(2, 0),
                    Catch::Matchers::WithinAbs(cos(theta(j)), tol)
                ); 
                REQUIRE_THAT(
                    J(2, 1),
                    Catch::Matchers::WithinAbs(-r(i) * sin(theta(j)), tol)
                ); 
                REQUIRE_THAT(J(2, 2), Catch::Matchers::WithinAbs(0, tol)); 
            }
        }
    }
}

/**
 * Tests for getTangentAndOrthogonalSpaceBases(). 
 */
TEST_CASE(
    "Tests for basis calculations for the Jacobian tangent space and orthogonal complement",
    "[getTangentAndOrthogonalSpaceBases()]"
)
{
    const double dx = 1e-8; 
    const double tol = 1e-8; 

    // Define a multivariable function that simply projects onto the last two 
    // coordinates
    DynamicVectorValuedFunction<double> F = [](const Ref<const Matrix<double, Dynamic, 1> >& x)
        -> Matrix<double, Dynamic, 1>
    {
        return x.tail(2); 
    };

    // Try calculating the Jacobian at multiple values of r, \theta, \phi
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist; 
    for (int i = 0; i < 100; ++i)
    {
        Matrix<double, Dynamic, 1> x0(6); 
        for (int j = 0; j < 6; ++j)
            x0(j) = uniform_dist(rng);

        // Calculate the Jacobian matrix and the corresponding bases 
        auto bases = getTangentAndOrthogonalSpaceBases<double>(F, x0, dx);
        Matrix<double, Dynamic, Dynamic> Qt = bases.first; 
        Matrix<double, Dynamic, Dynamic> Qp = bases.second; 
        
        // Check that the tangent space has dimension 6 - 2 = 4 
        REQUIRE(Qt.cols() == 4); 
        REQUIRE(Qp.cols() == 2);

        // Re-compute the SVD of the Jacobian matrix and check that the two
        // returned singular values are nonzero
        Matrix<double, Dynamic, Dynamic> J = getJacobian<double>(F, x0, dx); 
        auto svd = J.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
        Matrix<double, Dynamic, Dynamic> U = svd.matrixU();  
        Matrix<double, Dynamic, 1> singvals = svd.singularValues();
        Matrix<double, Dynamic, Dynamic> V = svd.matrixV(); 
        REQUIRE(U.rows() == 2); 
        REQUIRE(U.cols() == 2);
        REQUIRE(V.rows() == 6); 
        REQUIRE(V.cols() == 6);  
        REQUIRE(singvals.size() == 2); 
        REQUIRE(singvals(0) > tol); 
        REQUIRE(singvals(1) > tol);

        // Check that the basis for the tangent space spans the kernel of
        // the Jacobian
        for (int j = 0; j < 4; ++j)
        {
            REQUIRE_THAT(
                (V.col(j + 2) - Qt.col(j)).norm(),
                Catch::Matchers::WithinAbs(0, tol)
            );
            REQUIRE_THAT(
                (J * Qt.col(j)).norm(), Catch::Matchers::WithinAbs(0, tol)
            );
        }

        // Check that the basis vectors in the orthogonal complement are 
        // indeed orthogonal to the tangent space 
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < 4; ++k)
            {
                REQUIRE_THAT(
                    Qp.col(j).dot(Qt.col(k)),
                    Catch::Matchers::WithinAbs(0, tol)
                ); 
            }
        }
    } 
}

