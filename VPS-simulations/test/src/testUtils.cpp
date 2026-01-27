/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     1/27/2026
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
    "Tests for chain generation",
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
 */
TEST_CASE(
    "Tests for bond length, bond angle, and dihedral angle sampling", 
    "[sampleFene(), sampleAngleCosine(), sampleAngleDualGaussianMixture(), sampleDihedralHarmonic()]"
)
{
    // Sample bond lengths ... 
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;
    const int n = 100; 
    const double tol = 1e-5;

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
    const double theta0 = 160 * boost::math::constants::pi<double>() / 180;
    Array<double, Dynamic, 1> sample_angles_cosine(n); 
    for (int i = 0; i < n; ++i) 
        sample_angles_cosine(i) = sampleAngleCosine<double>(
            K_angle, theta0, kT, rng, uniform_dist, 50
        );

    // Check that the bond angles are within [0, \pi)
    REQUIRE((sample_angles_cosine >= 0).all()); 
    REQUIRE((sample_angles_cosine < boost::math::constants::pi<double>()).all());  

    // Sample bond angles according to the dual Gaussian mixture potential ... 
    const double A1 = 0.7; 
    const double A2 = 0.3; 
    const double w1 = sqrt(1 / (10 * kT));
    const double w2 = sqrt(1 / (8 * kT));
    const double theta1 = 160 * boost::math::constants::pi<double>() / 180;
    const double theta2 = boost::math::constants::half_pi<double>(); 
    Array<double, Dynamic, 1> sample_angles_dual(n); 
    for (int i = 0; i < n; ++i) 
        sample_angles_dual(i) = sampleAngleDualGaussianMixture<double>(
            A1, A2, w1, w2, theta1, theta2, kT, rng, uniform_dist, 50
        );

    // Check that the bond angles are within [0, \pi)
    REQUIRE((sample_angles_dual >= 0).all()); 
    REQUIRE((sample_angles_dual < boost::math::constants::pi<double>()).all());  

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
}

/**
 * Tests for getting and replacing segments in the PolymerConfiguration class. 
 */
TEST_CASE(
    "Tests for getting and replacing segments", "[getSegment(), replaceSegment()]"
)
{
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;
    const double tol = 1e-5; 

    // Generate an 8-atom segment
    const double length = 1.5; 
    Array<double, 6, 1> angles;
    angles << 140 * boost::math::constants::pi<double>() / 180,
              170 * boost::math::constants::pi<double>() / 180,  
              160 * boost::math::constants::pi<double>() / 180,
              boost::math::constants::half_pi<double>(), 
              150 * boost::math::constants::pi<double>() / 180,
              130 * boost::math::constants::pi<double>() / 180;
    Array<double, 5, 1> dihedrals;
    dihedrals << 60 * boost::math::constants::pi<double>() / 180, 
                 -60 * boost::math::constants::pi<double>() / 180,
                 boost::math::constants::pi<double>(),  
                 120 * boost::math::constants::pi<double>() / 180, 
                 60 * boost::math::constants::pi<double>() / 180;
    Matrix<double, 8, 3> coords; 
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

    // Generate the corresponding polymer
    PolymerConfiguration<double> config(coords, Units::NANO, 300); 

    // Check the polymer length and coordinates 
    REQUIRE(config.getLength() == 8);
    Matrix<double, Dynamic, 3> config_coords = config.getSegment(0, 8);
    REQUIRE(config_coords.rows() == 8); 
    for (int i = 0; i < 8; ++i)
        REQUIRE_THAT(
            (config_coords.row(i) - coords.row(i)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );

    // Replace atoms 2, 3, 4
    double theta1 = 132 * boost::math::constants::pi<double>() / 180; 
    double theta2 = 153 * boost::math::constants::pi<double>() / 180;
    double theta3 = 165 * boost::math::constants::pi<double>() / 180; 
    double phi2 = 54 * boost::math::constants::pi<double>() / 180;
    double phi3 = -72 * boost::math::constants::pi<double>() / 180;
    Matrix<double, Dynamic, 3> segment(3, 3); 
    segment.row(0) = generateNextAtom<double>(
        coords.row(0), coords.row(1), length, theta1, rng, uniform_dist
    );
    segment.row(1) = generateNextAtomDihedral<double>(
        config_coords.row(0), config_coords.row(1), segment.row(0), length,
        theta2, phi2, rng, uniform_dist, (phi2 > 0 ? 1 : -1)
    ); 
    segment.row(2) = generateNextAtomDihedral<double>(
        config_coords.row(1), segment.row(0), segment.row(1), length,
        theta3, phi3, rng, uniform_dist, (phi3 > 0 ? 1 : -1)
    );
    config.replaceSegment(segment, 2);

    // Check the polymer length and coordinates
    REQUIRE(config.getLength() == 8);
    config_coords = config.getSegment(0, 8); 
    REQUIRE(config_coords.rows() == 8); 
    for (int i = 0; i < 2; ++i)    // Upstream of replaced segment
        REQUIRE_THAT(
            (config_coords.row(i) - coords.row(i)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    for (int i = 2; i < 5; ++i)    // Replaced segment
        REQUIRE_THAT(
            (config_coords.row(i) - segment.row(i - 2)).norm(), 
            Catch::Matchers::WithinAbs(0, tol)
        ); 
    for (int i = 5; i < 8; ++i)    // Downstream of replaced segment 
        REQUIRE_THAT(
            (config_coords.row(i) - coords.row(i)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
}

/**
 * Tests for reptation, pop, and append methods in the PolymerConfiguration
 * class. 
 */
TEST_CASE(
    "Tests for reptation, pop, and append methods",
    "[appendSegmentToTail(), appendSegmentToHead(), popSegmentFromTail(), popSegmentFromHead(), reptateTowardsTail(), reptateTowardsHead()]"
)
{
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;
    const double tol = 1e-5; 

    // Generate an 8-atom segment
    const double length = 1.5; 
    Array<double, 6, 1> angles;
    angles << 140 * boost::math::constants::pi<double>() / 180,
              170 * boost::math::constants::pi<double>() / 180,  
              160 * boost::math::constants::pi<double>() / 180,
              boost::math::constants::half_pi<double>(), 
              150 * boost::math::constants::pi<double>() / 180,
              130 * boost::math::constants::pi<double>() / 180;
    Array<double, 5, 1> dihedrals;
    dihedrals << 60 * boost::math::constants::pi<double>() / 180, 
                 -60 * boost::math::constants::pi<double>() / 180,
                 boost::math::constants::pi<double>(),  
                 120 * boost::math::constants::pi<double>() / 180, 
                 60 * boost::math::constants::pi<double>() / 180;
    Matrix<double, 8, 3> coords; 
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

    // Generate the corresponding polymer
    PolymerConfiguration<double> config(coords, Units::NANO, 300); 

    // Append a 2-atom segment to the tail
    double theta1 = 173 * boost::math::constants::pi<double>() / 180; 
    double theta2 = 159 * boost::math::constants::pi<double>() / 180; 
    double phi1 = 58 * boost::math::constants::pi<double>() / 180; 
    double phi2 = -134 * boost::math::constants::pi<double>() / 180;
    Matrix<double, 3, 1> tail1 = generateNextAtomDihedral<double>(
        coords.row(5), coords.row(6), coords.row(7), length, theta1, phi1, 
        rng, uniform_dist, (phi1 > 0 ? 1 : -1)
    ); 
    Matrix<double, 3, 1> tail2 = generateNextAtomDihedral<double>(
        coords.row(6), coords.row(7), tail1, length, theta2, phi2,  
        rng, uniform_dist, (phi2 > 0 ? 1 : -1)
    ); 

    // Update the polymer configuration
    Matrix<double, Dynamic, 3> tail(2, 3); 
    tail << tail1(0), tail1(1), tail1(2), 
            tail2(0), tail2(1), tail2(2);  
    config.appendSegmentToTail(tail);

    // Check the polymer length and coordinates 
    REQUIRE(config.getLength() == 10);
    Matrix<double, Dynamic, 3> config_coords = config.getSegment(0, 10);
    REQUIRE(config_coords.rows() == 10); 
    for (int i = 0; i < 8; ++i)
        REQUIRE_THAT(
            (config_coords.row(i) - coords.row(i)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    REQUIRE_THAT(
        (config_coords.row(8) - tail1.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    ); 
    REQUIRE_THAT(
        (config_coords.row(9) - tail2.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );

    // Remove 3-atom segment from the head
    config.popSegmentFromHead(2);     // Last index to remove is 2
    
    // Check the polymer length and coordinates
    REQUIRE(config.getLength() == 7);
    config_coords = config.getSegment(0, 7); 
    REQUIRE(config_coords.rows() == 7); 
    for (int i = 0; i < 5; ++i)
        REQUIRE_THAT(
            (config_coords.row(i) - coords.row(i + 3)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    REQUIRE_THAT(
        (config_coords.row(5) - tail1.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    ); 
    REQUIRE_THAT(
        (config_coords.row(6) - tail2.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );

    // Append a 4-atom segment to the head 
    double theta3 = 132 * boost::math::constants::pi<double>() / 180; 
    double theta4 = 156 * boost::math::constants::pi<double>() / 180;
    double theta5 = 98 * boost::math::constants::pi<double>() / 180; 
    double theta6 = 111 * boost::math::constants::pi<double>() / 180; 
    double phi3 = 88 * boost::math::constants::pi<double>() / 180; 
    double phi4 = 145 * boost::math::constants::pi<double>() / 180;
    double phi5 = -20 * boost::math::constants::pi<double>() / 180; 
    double phi6 = 63 * boost::math::constants::pi<double>() / 180; 
    Matrix<double, 3, 1> head1 = generateNextAtomDihedral<double>(
        config_coords.row(2), config_coords.row(1), config_coords.row(0),
        length, theta3, phi3, rng, uniform_dist, (phi3 > 0 ? 1 : -1)
    ); 
    Matrix<double, 3, 1> head2 = generateNextAtomDihedral<double>(
        config_coords.row(1), config_coords.row(0), head1, length, theta4,
        phi4, rng, uniform_dist, (phi4 > 0 ? 1 : -1)
    ); 
    Matrix<double, 3, 1> head3 = generateNextAtomDihedral<double>(
        config_coords.row(0), head1, head2, length, theta5, phi5, rng,
        uniform_dist, (phi5 > 0 ? 1 : -1)
    ); 
    Matrix<double, 3, 1> head4 = generateNextAtomDihedral<double>(
        head1, head2, head3, length, theta6, phi6, rng, uniform_dist,
        (phi6 > 0 ? 1 : -1)
    );

    // Update the polymer configuration
    Matrix<double, Dynamic, 3> head(4, 3); 
    head << head4(0), head4(1), head4(2), 
            head3(0), head3(1), head3(2),
            head2(0), head2(1), head2(2),
            head1(0), head1(1), head1(2); 
    config.appendSegmentToHead(head); 

    // Check the polymer length and coordinates 
    REQUIRE(config.getLength() == 11);
    config_coords = config.getSegment(0, 11);
    REQUIRE(config_coords.rows() == 11);
    REQUIRE_THAT(
        (config_coords.row(0) - head4.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );
    REQUIRE_THAT(
        (config_coords.row(1) - head3.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    ); 
    REQUIRE_THAT(
        (config_coords.row(2) - head2.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    ); 
    REQUIRE_THAT(
        (config_coords.row(3) - head1.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    ); 
    for (int i = 4; i < 9; ++i)
    {
        // Atom 4 in the new polymer configuration is atom 3 in the original
        // array, since 3 atoms were popped and 4 were appended
        REQUIRE_THAT(
            (config_coords.row(i) - coords.row(i - 1)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    }
    REQUIRE_THAT(
        (config_coords.row(9) - tail1.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    ); 
    REQUIRE_THAT(
        (config_coords.row(10) - tail2.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );

    // Remove 3-atom segment from the head
    config.popSegmentFromTail(11 - 3);    // Last index to remove

    // Check the polymer length and coordinates
    REQUIRE(config.getLength() == 8);
    config_coords = config.getSegment(0, 8); 
    REQUIRE(config_coords.rows() == 8);
    REQUIRE_THAT(
        (config_coords.row(0) - head4.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );
    REQUIRE_THAT(
        (config_coords.row(1) - head3.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    ); 
    REQUIRE_THAT(
        (config_coords.row(2) - head2.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    ); 
    REQUIRE_THAT(
        (config_coords.row(3) - head1.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    ); 
    for (int i = 4; i < 8; ++i)
    {
        // Atom 4 in the new polymer configuration is atom 3 in the original
        // array, since 3 atoms were popped and 4 were appended from the head
        REQUIRE_THAT(
            (config_coords.row(i) - coords.row(i - 1)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    }

    // Reptate towards the tail 
    double theta7 = 37 * boost::math::constants::pi<double>() / 180; 
    double phi7 = -99 * boost::math::constants::pi<double>() / 180;
    Matrix<double, 3, 1> tail3 = generateNextAtomDihedral<double>(
        config_coords.row(5), config_coords.row(6), config_coords.row(7),
        length, theta7, phi7, rng, uniform_dist, (phi7 > 0 ? 1 : -1)
    );
    config.reptateTowardsTail(tail3);

    // Check the polymer length and coordinates
    REQUIRE(config.getLength() == 8);
    config_coords = config.getSegment(0, 8); 
    REQUIRE(config_coords.rows() == 8);
    REQUIRE_THAT(
        (config_coords.row(0) - head3.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );
    REQUIRE_THAT(
        (config_coords.row(1) - head2.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    ); 
    REQUIRE_THAT(
        (config_coords.row(2) - head1.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    ); 
    for (int i = 3; i < 7; ++i)
    {
        // Atom 3 in the new polymer configuration is now atom 3 in the
        // original array
        REQUIRE_THAT(
            (config_coords.row(i) - coords.row(i)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    }
    REQUIRE_THAT(
        (config_coords.row(7) - tail3.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );

    // Reptate towards the head
    double theta8 = 126 * boost::math::constants::pi<double>() / 180; 
    double phi8 = -2 * boost::math::constants::pi<double>() / 180;
    Matrix<double, 3, 1> head5 = generateNextAtomDihedral<double>(
        config_coords.row(2), config_coords.row(1), config_coords.row(0),
        length, theta8, phi8, rng, uniform_dist, (phi8 > 0 ? 1 : -1)
    );
    config.reptateTowardsHead(head5);

    // Check the polymer length and coordinates
    REQUIRE(config.getLength() == 8);
    config_coords = config.getSegment(0, 8); 
    REQUIRE(config_coords.rows() == 8);
    REQUIRE_THAT(
        (config_coords.row(0) - head5.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );
    REQUIRE_THAT(
        (config_coords.row(1) - head3.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );
    REQUIRE_THAT(
        (config_coords.row(2) - head2.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    ); 
    REQUIRE_THAT(
        (config_coords.row(3) - head1.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    ); 
    for (int i = 4; i < 8; ++i)
    {
        // Atom 4 in the new polymer configuration is now atom 3 in the
        // original array (popped 3, appended 4, reptated towards tail, and
        // now reptated towards head)
        REQUIRE_THAT(
            (config_coords.row(i) - coords.row(i - 1)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    }
}
