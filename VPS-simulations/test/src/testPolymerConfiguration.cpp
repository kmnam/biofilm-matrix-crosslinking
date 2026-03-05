/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     3/4/2026
 */

#include <iostream>
#include <cstdlib>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/random.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/utils.hpp"
#include "../../include/polymerConfiguration.hpp"

using namespace Eigen;

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
    Matrix<double, Dynamic, 3> new_coords = config.getSegment(0, 10);
    REQUIRE(new_coords.rows() == 10); 
    for (int i = 0; i < 8; ++i)
        REQUIRE_THAT(
            (new_coords.row(i) - coords.row(i)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    REQUIRE_THAT(
        (new_coords.row(8) - tail1.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    ); 
    REQUIRE_THAT(
        (new_coords.row(9) - tail2.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );

    // Remove 3-atom segment from the head
    config.popSegmentFromHead(2);     // Last index to remove is 2
    
    // Check the polymer length and coordinates
    REQUIRE(config.getLength() == 7);
    Matrix<double, Dynamic, 3> old_coords(new_coords); 
    new_coords = config.getSegment(0, 7); 
    REQUIRE(new_coords.rows() == 7); 
    for (int i = 0; i < 7; ++i)
        REQUIRE_THAT(
            (new_coords.row(i) - old_coords.row(i + 3)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );

    // Append a 4-atom segment to the head
    old_coords = new_coords; 
    double theta3 = 132 * boost::math::constants::pi<double>() / 180; 
    double theta4 = 156 * boost::math::constants::pi<double>() / 180;
    double theta5 = 98 * boost::math::constants::pi<double>() / 180; 
    double theta6 = 111 * boost::math::constants::pi<double>() / 180; 
    double phi3 = 88 * boost::math::constants::pi<double>() / 180; 
    double phi4 = 145 * boost::math::constants::pi<double>() / 180;
    double phi5 = -20 * boost::math::constants::pi<double>() / 180; 
    double phi6 = 63 * boost::math::constants::pi<double>() / 180; 
    Matrix<double, 3, 1> head1 = generateNextAtomDihedral<double>(
        old_coords.row(2), old_coords.row(1), old_coords.row(0),
        length, theta3, phi3, rng, uniform_dist, (phi3 > 0 ? 1 : -1)
    ); 
    Matrix<double, 3, 1> head2 = generateNextAtomDihedral<double>(
        old_coords.row(1), old_coords.row(0), head1, length, theta4,
        phi4, rng, uniform_dist, (phi4 > 0 ? 1 : -1)
    ); 
    Matrix<double, 3, 1> head3 = generateNextAtomDihedral<double>(
        old_coords.row(0), head1, head2, length, theta5, phi5, rng,
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
    new_coords = config.getSegment(0, 11);
    REQUIRE(new_coords.rows() == 11);
    REQUIRE_THAT(
        (new_coords.row(0) - head4.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );
    REQUIRE_THAT(
        (new_coords.row(1) - head3.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    ); 
    REQUIRE_THAT(
        (new_coords.row(2) - head2.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    ); 
    REQUIRE_THAT(
        (new_coords.row(3) - head1.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    ); 
    for (int i = 4; i < 11; ++i)
    {
        REQUIRE_THAT(
            (new_coords.row(i) - old_coords.row(i - 4)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    }

    // Remove 3-atom segment from the tail 
    config.popSegmentFromTail(11 - 3);    // Last index to remove

    // Check the polymer length and coordinates
    REQUIRE(config.getLength() == 8);
    old_coords = new_coords; 
    new_coords = config.getSegment(0, 8); 
    REQUIRE(new_coords.rows() == 8);
    for (int i = 0; i < 8; ++i)
    {
        REQUIRE_THAT(
            (new_coords.row(i) - old_coords.row(i)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    }

    // Reptate towards the tail 
    old_coords = new_coords; 
    double theta7 = 37 * boost::math::constants::pi<double>() / 180; 
    double phi7 = -99 * boost::math::constants::pi<double>() / 180;
    Matrix<double, 3, 1> tail3 = generateNextAtomDihedral<double>(
        old_coords.row(5), old_coords.row(6), old_coords.row(7),
        length, theta7, phi7, rng, uniform_dist, (phi7 > 0 ? 1 : -1)
    );
    config.reptateTowardsTail(tail3);

    // Check the polymer length and coordinates
    REQUIRE(config.getLength() == 8);
    new_coords = config.getSegment(0, 8); 
    REQUIRE(new_coords.rows() == 8);
    for (int i = 0; i < 7; ++i)
    {
        REQUIRE_THAT(
            (new_coords.row(i) - old_coords.row(i + 1)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    }
    REQUIRE_THAT(
        (new_coords.row(7) - tail3.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );

    // Reptate towards the head
    old_coords = new_coords; 
    double theta8 = 126 * boost::math::constants::pi<double>() / 180; 
    double phi8 = -2 * boost::math::constants::pi<double>() / 180;
    Matrix<double, 3, 1> head5 = generateNextAtomDihedral<double>(
        old_coords.row(2), old_coords.row(1), old_coords.row(0),
        length, theta8, phi8, rng, uniform_dist, (phi8 > 0 ? 1 : -1)
    );
    config.reptateTowardsHead(head5);

    // Check the polymer length and coordinates
    REQUIRE(config.getLength() == 8);
    new_coords = config.getSegment(0, 8); 
    REQUIRE(new_coords.rows() == 8);
    REQUIRE_THAT(
        (new_coords.row(0) - head5.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );
    for (int i = 1; i < 8; ++i)
    {
        REQUIRE_THAT(
            (new_coords.row(i) - old_coords.row(i - 1)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    }

    // Generate 3 new atoms at the tail and reptate
    old_coords = new_coords; 
    double theta9 = 140 * boost::math::constants::pi<double>() / 180; 
    double phi9 = -45 * boost::math::constants::pi<double>() / 180; 
    double theta10 = 94 * boost::math::constants::pi<double>() / 180; 
    double phi10 = 82 * boost::math::constants::pi<double>() / 180; 
    double theta11 = 56 * boost::math::constants::pi<double>() / 180; 
    double phi11 = -132 * boost::math::constants::pi<double>() / 180; 
    Matrix<double, Dynamic, 3> tail_segment(3, 3); 
    tail_segment.row(0) = generateNextAtomDihedral<double>(
        old_coords.row(5), old_coords.row(6), old_coords.row(7), 
        length, theta9, phi9, rng, uniform_dist, (phi9 > 0 ? 1 : -1)
    ); 
    tail_segment.row(1) = generateNextAtomDihedral<double>(
        old_coords.row(6), old_coords.row(7), tail_segment.row(0), 
        length, theta10, phi10, rng, uniform_dist, (phi10 > 0 ? 1 : -1)
    ); 
    tail_segment.row(2) = generateNextAtomDihedral<double>(
        old_coords.row(7), tail_segment.row(0), tail_segment.row(1), 
        length, theta11, phi11, rng, uniform_dist, (phi11 > 0 ? 1 : -1)
    );
    config.reptateTowardsTailMultimer(tail_segment);

    // Check the polymer length and coordinates
    REQUIRE(config.getLength() == 8);
    new_coords = config.getSegment(0, 8); 
    REQUIRE(new_coords.rows() == 8);
    for (int i = 0; i < 5; ++i)
    {
        REQUIRE_THAT(
            (new_coords.row(i) - old_coords.row(i + 3)).norm(), 
            Catch::Matchers::WithinAbs(0, tol)
        );
    }
    REQUIRE_THAT(
        (new_coords.row(5) - tail_segment.row(0)).norm(), 
        Catch::Matchers::WithinAbs(0, tol)
    );
    REQUIRE_THAT(
        (new_coords.row(6) - tail_segment.row(1)).norm(), 
        Catch::Matchers::WithinAbs(0, tol)
    ); 
    REQUIRE_THAT(
        (new_coords.row(7) - tail_segment.row(2)).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );

    // Generate 4 new atoms at the tail and reptate
    old_coords = new_coords; 
    double theta12 = 9 * boost::math::constants::pi<double>() / 180; 
    double phi12 = 178 * boost::math::constants::pi<double>() / 180; 
    double theta13 = 52 * boost::math::constants::pi<double>() / 180; 
    double phi13 = -13 * boost::math::constants::pi<double>() / 180; 
    double theta14 = 171 * boost::math::constants::pi<double>() / 180; 
    double phi14 = 16 * boost::math::constants::pi<double>() / 180; 
    double theta15 = 72 * boost::math::constants::pi<double>() / 180; 
    double phi15 = 49 * boost::math::constants::pi<double>() / 180; 
    Matrix<double, Dynamic, 3> head_segment(4, 3); 
    head_segment.row(3) = generateNextAtomDihedral<double>(
        old_coords.row(2), old_coords.row(1), old_coords.row(0), 
        length, theta12, phi12, rng, uniform_dist, (phi12 > 0 ? 1 : -1)
    ); 
    head_segment.row(2) = generateNextAtomDihedral<double>(
        old_coords.row(1), old_coords.row(0), head_segment.row(3), 
        length, theta13, phi13, rng, uniform_dist, (phi13 > 0 ? 1 : -1)
    ); 
    head_segment.row(1) = generateNextAtomDihedral<double>(
        old_coords.row(0), head_segment.row(3), head_segment.row(2), 
        length, theta14, phi14, rng, uniform_dist, (phi14 > 0 ? 1 : -1)
    );
    head_segment.row(0) = generateNextAtomDihedral<double>(
        head_segment.row(3), head_segment.row(2), head_segment.row(1),
        length, theta15, phi15, rng, uniform_dist, (phi15 > 0 ? 1 : -1)
    );
    config.reptateTowardsHeadMultimer(head_segment); 

    // Check the polymer length and coordinates
    REQUIRE(config.getLength() == 8);
    new_coords = config.getSegment(0, 8); 
    REQUIRE(new_coords.rows() == 8);
    for (int i = 0; i < 4; ++i)
    {
        REQUIRE_THAT(
            (new_coords.row(i) - head_segment.row(i)).norm(), 
            Catch::Matchers::WithinAbs(0, tol)
        );
    }
    for (int i = 4; i < 8; ++i)
    {
        REQUIRE_THAT(
            (new_coords.row(i) - old_coords.row(i - 4)).norm(), 
            Catch::Matchers::WithinAbs(0, tol)
        );
    }
}

/**
 * Tests for generateKMer(). 
 */
TEST_CASE("Tests for k-mer generation", "[generateKMer()]")
{
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;

    // Set up potential and sampling parameters
    Matrix<double, 3, 1> r0 = Matrix<double, 3, 1>::Zero();
    std::unordered_map<std::string, double> lj_params, 
                                            fene_params, 
                                            cosine_params,
                                            gaussian_params,
                                            dihedral_params;
    double kT = 1.380649e-2 * 300;
    lj_params["eps"] = kT; 
    lj_params["sigma"] = 0.9;
    fene_params["K"] = 30 * kT; 
    fene_params["R0"] = 1.5;
    cosine_params["K"] = 20 * kT;
    cosine_params["theta0"] = 160 * boost::math::constants::pi<double>() / 180;
    gaussian_params["A1"] = 0.9; 
    gaussian_params["A2"] = 0.1;
    gaussian_params["w1"] = 0.4472135955;    // = 2 * 1/sqrt(20) 
    gaussian_params["w2"] = 0.4472135955; 
    gaussian_params["theta1"] = 160 * boost::math::constants::pi<double>() / 180; 
    gaussian_params["theta2"] = 90 * boost::math::constants::pi<double>() / 180; 
    dihedral_params["K"] = 10 * kT;
    dihedral_params["d"] = 1; 
    dihedral_params["n"] = 1; 
    const double collision_threshold = 0.1;
    const int max_tries_per_atom = 50;
    const int max_n_backtracks = 50;  

    // Generate a 10-mer
    PolymerConfiguration<double> config = generateKMer<double, 10>(
        lj_params, fene_params, AngleMode::COSINE, cosine_params, dihedral_params,
        r0, collision_threshold, max_tries_per_atom, max_n_backtracks, rng, 
        uniform_dist
    );
    REQUIRE(config.getLength() == 10); 

    // Check that none of the atoms are within the collision threshold
    Matrix<double, Dynamic, 3> coords = config.getSegment(0, 10);  
    for (int i = 0; i < 10; ++i)
    {
        for (int j = i + 1; j < 10; ++j)
        {
            double dij = (coords.row(i) - coords.row(j)).norm(); 
            REQUIRE(dij > collision_threshold); 
        }
    } 
     
    // Check that none of the bond lengths exceed R0 
    for (int i = 0; i < coords.rows() - 1; ++i)
    {
        double dij = (coords.row(i + 1) - coords.row(i)).norm(); 
        REQUIRE(dij < fene_params["R0"]);
    } 
}

/**
 * Tests for parseLammps() and writeLammps(). 
 */
TEST_CASE("Tests for parsing and writing functions", "[writeLammps(), parseLammps()]")
{
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;
    const double tol = 1e-8; 

    // Set up potential and sampling parameters
    Matrix<double, 3, 1> r0 = Matrix<double, 3, 1>::Zero();
    std::unordered_map<std::string, double> lj_params, 
                                            fene_params, 
                                            cosine_params,
                                            gaussian_params,
                                            dihedral_params;
    double kT = 1.380649e-2 * 300;
    lj_params["eps"] = kT; 
    lj_params["sigma"] = 0.9;
    fene_params["K"] = 30 * kT; 
    fene_params["R0"] = 1.5;
    cosine_params["K"] = 20 * kT;
    cosine_params["theta0"] = 160 * boost::math::constants::pi<double>() / 180;
    gaussian_params["A1"] = 0.9; 
    gaussian_params["A2"] = 0.1;
    gaussian_params["w1"] = 0.4472135955;    // = 1/sqrt(20) 
    gaussian_params["w2"] = 0.4472135955; 
    gaussian_params["theta1"] = 160 * boost::math::constants::pi<double>() / 180; 
    gaussian_params["theta2"] = 90 * boost::math::constants::pi<double>() / 180; 
    dihedral_params["K"] = 10 * kT;
    dihedral_params["d"] = 1; 
    dihedral_params["n"] = 1; 
    const double collision_threshold = 0.1;
    const int max_tries_per_atom = 50;
    const int max_n_backtracks = 50;  

    // Generate a 10-mer with a cosine angle potential
    PolymerConfiguration<double> config = generateKMer<double, 10>(
        lj_params, fene_params, AngleMode::COSINE, cosine_params, dihedral_params,
        r0, collision_threshold, max_tries_per_atom, max_n_backtracks, rng, 
        uniform_dist
    );
    REQUIRE(config.getLength() == 10);

    // Write the 10-mer coordinates to file 
    config.writeLammps(
        "configs/test_10mer_cosine.txt", lj_params, fene_params, AngleMode::COSINE, 
        cosine_params, dihedral_params, "Test configuration", -100, 100, 
        -100, 100, -100, 100, 1
    ); 

    // Parse the 10-mer coordinates 
    auto result = parseLammps<double>(
        "configs/test_10mer_cosine.txt", Units::NANO, 300
    ); 
    PolymerConfiguration<double> config2 = std::get<0>(result); 
    std::unordered_map<std::string, double> lj_params2 = std::get<1>(result); 
    std::unordered_map<std::string, double> fene_params2 = std::get<2>(result); 
    AngleMode angle_mode2 = std::get<3>(result); 
    std::unordered_map<std::string, double> angle_params2 = std::get<4>(result); 
    std::unordered_map<std::string, double> dihedral_params2 = std::get<5>(result); 

    // Check the atomic coordinates 
    REQUIRE(config2.getLength() == config.getLength()); 
    Matrix<double, Dynamic, 3> coords = config.getSegment(0, 10); 
    Matrix<double, Dynamic, 3> coords2 = config2.getSegment(0, 10); 
    for (int i = 0; i < 10; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            REQUIRE_THAT(coords2(i, j), Catch::Matchers::WithinAbs(coords(i, j), tol)); 
        } 
    }

    // Check the potential parameters
    REQUIRE(lj_params2.find("eps") != lj_params2.end()); 
    REQUIRE_THAT(lj_params2["eps"], Catch::Matchers::WithinAbs(lj_params["eps"], tol));
    REQUIRE(lj_params2.find("sigma") != lj_params2.end()); 
    REQUIRE_THAT(lj_params2["sigma"], Catch::Matchers::WithinAbs(lj_params["sigma"], tol));
    REQUIRE(fene_params2.find("K") != fene_params2.end()); 
    REQUIRE_THAT(fene_params2["K"], Catch::Matchers::WithinAbs(fene_params["K"], tol)); 
    REQUIRE(fene_params2.find("R0") != fene_params2.end()); 
    REQUIRE_THAT(fene_params2["R0"], Catch::Matchers::WithinAbs(fene_params["R0"], tol)); 
    REQUIRE(angle_mode2 == AngleMode::COSINE); 
    REQUIRE(angle_params2.find("K") != angle_params2.end()); 
    REQUIRE_THAT(angle_params2["K"], Catch::Matchers::WithinAbs(cosine_params["K"], tol)); 
    REQUIRE(angle_params2.find("theta0") != angle_params2.end()); 
    REQUIRE_THAT(angle_params2["theta0"], Catch::Matchers::WithinAbs(cosine_params["theta0"], tol)); 
    REQUIRE(dihedral_params2.find("K") != dihedral_params2.end()); 
    REQUIRE_THAT(dihedral_params2["K"], Catch::Matchers::WithinAbs(dihedral_params["K"], tol)); 
    REQUIRE(dihedral_params2.find("d") != dihedral_params2.end()); 
    REQUIRE_THAT(dihedral_params2["d"], Catch::Matchers::WithinAbs(1, tol)); 
    REQUIRE(dihedral_params2.find("n") != dihedral_params2.end()); 
    REQUIRE_THAT(dihedral_params2["n"], Catch::Matchers::WithinAbs(1, tol));

    // Generate a 10-mer with a dual Gaussian mixture angle potential 
    config = generateKMer<double, 10>(
        lj_params, fene_params, AngleMode::GAUSSIAN, gaussian_params,
        dihedral_params, r0, collision_threshold, max_tries_per_atom,
        max_n_backtracks, rng, uniform_dist
    );
    REQUIRE(config.getLength() == 10);

    // Write the 10-mer coordinates to file 
    config.writeLammps(
        "configs/test_10mer_gaussian.txt", lj_params, fene_params,
        AngleMode::GAUSSIAN, gaussian_params, dihedral_params, "Test configuration",
        -100, 100, -100, 100, -100, 100, 1
    ); 

    // Parse the 10-mer coordinates 
    result = parseLammps<double>(
        "configs/test_10mer_gaussian.txt", Units::NANO, 300
    ); 
    config2 = std::get<0>(result); 
    lj_params2 = std::get<1>(result); 
    fene_params2 = std::get<2>(result); 
    angle_mode2 = std::get<3>(result); 
    angle_params2 = std::get<4>(result); 
    dihedral_params2 = std::get<5>(result); 

    // Check the atomic coordinates 
    REQUIRE(config2.getLength() == config.getLength()); 
    coords = config.getSegment(0, 10); 
    coords2 = config2.getSegment(0, 10); 
    for (int i = 0; i < 10; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            REQUIRE_THAT(coords2(i, j), Catch::Matchers::WithinAbs(coords(i, j), tol)); 
        } 
    }

    // Check the potential parameters
    REQUIRE(lj_params2.find("eps") != lj_params2.end()); 
    REQUIRE_THAT(lj_params2["eps"], Catch::Matchers::WithinAbs(lj_params["eps"], tol));
    REQUIRE(lj_params2.find("sigma") != lj_params2.end()); 
    REQUIRE_THAT(lj_params2["sigma"], Catch::Matchers::WithinAbs(lj_params["sigma"], tol));
    REQUIRE(fene_params2.find("K") != fene_params2.end()); 
    REQUIRE_THAT(fene_params2["K"], Catch::Matchers::WithinAbs(fene_params["K"], tol)); 
    REQUIRE(fene_params2.find("R0") != fene_params2.end()); 
    REQUIRE_THAT(fene_params2["R0"], Catch::Matchers::WithinAbs(fene_params["R0"], tol)); 
    REQUIRE(angle_mode2 == AngleMode::GAUSSIAN); 
    REQUIRE(angle_params2.find("A1") != angle_params2.end()); 
    REQUIRE_THAT(angle_params2["A1"], Catch::Matchers::WithinAbs(gaussian_params["A1"], tol));
    REQUIRE(angle_params2.find("A2") != angle_params2.end()); 
    REQUIRE_THAT(angle_params2["A2"], Catch::Matchers::WithinAbs(gaussian_params["A2"], tol)); 
    REQUIRE(angle_params2.find("w1") != angle_params2.end()); 
    REQUIRE_THAT(angle_params2["w1"], Catch::Matchers::WithinAbs(gaussian_params["w1"], tol));
    REQUIRE(angle_params2.find("w2") != angle_params2.end()); 
    REQUIRE_THAT(angle_params2["w2"], Catch::Matchers::WithinAbs(gaussian_params["w2"], tol)); 
    REQUIRE(angle_params2.find("theta1") != angle_params2.end()); 
    REQUIRE_THAT(angle_params2["theta1"], Catch::Matchers::WithinAbs(gaussian_params["theta1"], tol));
    REQUIRE(angle_params2.find("theta2") != angle_params2.end()); 
    REQUIRE_THAT(angle_params2["theta2"], Catch::Matchers::WithinAbs(gaussian_params["theta2"], tol)); 
    REQUIRE(dihedral_params2.find("K") != dihedral_params2.end()); 
    REQUIRE_THAT(dihedral_params2["K"], Catch::Matchers::WithinAbs(dihedral_params["K"], tol)); 
    REQUIRE(dihedral_params2.find("d") != dihedral_params2.end()); 
    REQUIRE_THAT(dihedral_params2["d"], Catch::Matchers::WithinAbs(1, tol)); 
    REQUIRE(dihedral_params2.find("n") != dihedral_params2.end()); 
    REQUIRE_THAT(dihedral_params2["n"], Catch::Matchers::WithinAbs(1, tol));
}

/**
 * Tests for getNonbondedEnergy(), getBondEnergy(), getBondAngleEnergy(), 
 * and getDihedralAngleEnergy().
 *
 * The output of these functions is compared against pre-computed values 
 * obtained with LAMMPS, specifically via the following commands:
 *
 * mpirun -np 1 lmp -i get_energy_cosine.lammps -v VARS configs/test_10mer_cosine.txt
 *
 * mpirun -np 1 lmp -i get_energy_gaussian.lammps -v VARS configs/test_10mer_gaussian.txt
 *
 * which are invoked by a wrapped Python script (run_lammps_get_energy.py)
 * that is called within this module. 
 */
TEST_CASE(
    "Tests for energy calculation methods", 
    "[getNonbondedEnergy(), getBondEnergy(), getBondAngleEnergy(), getDihedralAngleEnergy()]"
)
{
    const double tol = 1e-3;

    // Parse test 10-mer coordinates with angles chosen from a cosine potential
    auto result = parseLammps<double>(
        "configs/test_10mer_cosine.txt", Units::NANO, 300
    ); 
    PolymerConfiguration<double> config = std::get<0>(result); 
    std::unordered_map<std::string, double> lj_params = std::get<1>(result); 
    std::unordered_map<std::string, double> fene_params = std::get<2>(result); 
    std::unordered_map<std::string, double> angle_params = std::get<4>(result); 
    std::unordered_map<std::string, double> dihedral_params = std::get<5>(result);

    // Calculate energies via LAMMPS
    std::string cmd = "python3 run_lammps_get_energy.py configs/test_10mer_cosine.txt"; 
    int rc = std::system(cmd.c_str()); 
    if (rc != 0)
        throw std::runtime_error("Failed to run run_lammps_get_energy.py");
    std::ifstream infile("configs/test_10mer_cosine_energy.txt");
    std::stringstream ss;
    std::string line, token;
    std::getline(infile, line);
    ss << line; 
    std::getline(ss, token, ' '); 
    double energy_lammps = std::stod(token); 
    std::getline(ss, token, ' '); 
    double nonbonded_energy_lammps = std::stod(token); 
    std::getline(ss, token, ' '); 
    double bond_energy_lammps = std::stod(token); 
    std::getline(ss, token, ' '); 
    double angle_energy_lammps = std::stod(token);
    infile.close();  

    // Calculate the non-bonded interaction energy between non-consecutive atoms
    // and compare against LAMMPS-computed value 
    double neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"]; 
    double nonbonded_energy_nc = config.getNonbondedEnergy(
        lj_params, neighbor_threshold, true
    );
    REQUIRE_THAT(
        nonbonded_energy_nc,
        Catch::Matchers::WithinAbs(nonbonded_energy_lammps, tol)
    ); 

    // Calculate the bonded interaction energy (including Lennard-Jones) and 
    // compare against LAMMPS-computed value
    double bond_energy = config.getBondEnergy(fene_params, true, lj_params);
    REQUIRE_THAT(bond_energy, Catch::Matchers::WithinAbs(bond_energy_lammps, tol)); 

    //REQUIRE_THAT(bond_energy, Catch::Matchers::WithinAbs(654.92774, tol)); 

    // Calculate the bond angle energy and compare against LAMMPS-computed value
    double angle_energy = config.getBondAngleEnergy(AngleMode::COSINE, angle_params); 
    REQUIRE_THAT(angle_energy, Catch::Matchers::WithinAbs(angle_energy_lammps, tol));

    // Calculate the dihedral angle energy and compare against LAMMPS-computed
    // value
    double dihedral_energy = config.getDihedralAngleEnergy(dihedral_params); 
    REQUIRE_THAT(
        dihedral_energy,
        Catch::Matchers::WithinAbs(
            energy_lammps - nonbonded_energy_lammps - bond_energy_lammps - angle_energy_lammps,
            tol
        ) 
    );

    // Parse test 10-mer coordinates with angles chosen from a dual Gaussian
    // mixture potential
    result = parseLammps<double>(
        "configs/test_10mer_gaussian.txt", Units::NANO, 300
    ); 
    config = std::get<0>(result); 
    lj_params = std::get<1>(result); 
    fene_params = std::get<2>(result); 
    angle_params = std::get<4>(result); 
    dihedral_params = std::get<5>(result);

    // Calculate energies via LAMMPS
    cmd = "python3 run_lammps_get_energy.py configs/test_10mer_gaussian.txt"; 
    rc = std::system(cmd.c_str()); 
    if (rc != 0)
        throw std::runtime_error("Failed to run run_lammps_get_energy.py");
    infile.open("configs/test_10mer_gaussian_energy.txt");
    ss.str(std::string()); 
    ss.clear(); 
    std::getline(infile, line);
    ss << line; 
    std::getline(ss, token, ' '); 
    energy_lammps = std::stod(token); 
    std::getline(ss, token, ' '); 
    nonbonded_energy_lammps = std::stod(token); 
    std::getline(ss, token, ' '); 
    bond_energy_lammps = std::stod(token); 
    std::getline(ss, token, ' '); 
    angle_energy_lammps = std::stod(token); 

    // Calculate the non-bonded interaction energy between non-consecutive atoms
    // and compare against LAMMPS-computed value 
    neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"]; 
    nonbonded_energy_nc = config.getNonbondedEnergy(
        lj_params, neighbor_threshold, true
    );
    REQUIRE_THAT(
        nonbonded_energy_nc,
        Catch::Matchers::WithinAbs(nonbonded_energy_lammps, tol)
    ); 

    // Calculate the bonded interaction energy (including Lennard-Jones) and 
    // compare against LAMMPS-computed value
    bond_energy = config.getBondEnergy(fene_params, true, lj_params);
    REQUIRE_THAT(bond_energy, Catch::Matchers::WithinAbs(bond_energy_lammps, tol));

    // Calculate the bond angle energy and compare against LAMMPS-computed value
    angle_energy = config.getBondAngleEnergy(AngleMode::GAUSSIAN, angle_params); 
    REQUIRE_THAT(angle_energy, Catch::Matchers::WithinAbs(angle_energy_lammps, tol));

    // Calculate the dihedral angle energy and compare against LAMMPS-computed
    // value
    dihedral_energy = config.getDihedralAngleEnergy(dihedral_params); 
    REQUIRE_THAT(
        dihedral_energy,
        Catch::Matchers::WithinAbs(
            energy_lammps - nonbonded_energy_lammps - bond_energy_lammps - angle_energy_lammps,
            tol
        )
    ); 
}

/**
 * Tests for getSegmentReplacementEnergyDifference(). 
 */
TEST_CASE(
    "Tests for segment replacement energy difference calculation",
    "[getSegmentReplacementEnergyDifference()]"
)
{
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;
    const double tol = 1e-8;

    // Parse test 10-mer coordinates with angles chosen from a cosine potential
    auto result = parseLammps<double>(
        "configs/test_10mer_cosine.txt", Units::NANO, 300
    ); 
    PolymerConfiguration<double> config = std::get<0>(result); 
    std::unordered_map<std::string, double> lj_params = std::get<1>(result); 
    std::unordered_map<std::string, double> fene_params = std::get<2>(result); 
    std::unordered_map<std::string, double> angle_params = std::get<4>(result); 
    std::unordered_map<std::string, double> dihedral_params = std::get<5>(result);

    // Randomly perturb the last 3 atoms in this configuration
    int length = config.getLength(); 
    Matrix<double, Dynamic, 3> coords = config.getSegment(0, length); 
    Matrix<double, Dynamic, 3> coords2(coords); 
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            double r = -0.01 + 0.02 * uniform_dist(rng);  
            coords2(length - i - 1, j) += r;
        }
    } 
    PolymerConfiguration<double> config2(coords2, Units::NANO, 300);

    // Calculate the segment replacement energy difference
    double neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"]; 
    double replace_energy_12 = config.getSegmentReplacementEnergyDifference(
        coords2(Eigen::seqN(length - 3, 3), Eigen::all), length - 3, 
        lj_params, neighbor_threshold, fene_params, AngleMode::COSINE, 
        angle_params, dihedral_params
    );

    // Calculate the reverse segment replacement energy difference
    double replace_energy_21 = config2.getSegmentReplacementEnergyDifference(
        coords(Eigen::seqN(length - 3, 3), Eigen::all), length - 3, 
        lj_params, neighbor_threshold, fene_params, AngleMode::COSINE, 
        angle_params, dihedral_params
    );
    REQUIRE_THAT(replace_energy_12, Catch::Matchers::WithinAbs(-replace_energy_21, tol)); 

    // Calculate the total energy of the two configurations 
    double energy1_nonbonded = config.getNonbondedEnergy(lj_params, neighbor_threshold); 
    double energy1_bond = config.getBondEnergy(fene_params); 
    double energy1_angle = config.getBondAngleEnergy(AngleMode::COSINE, angle_params); 
    double energy1_dihedral = config.getDihedralAngleEnergy(dihedral_params); 
    double energy2_nonbonded = config2.getNonbondedEnergy(lj_params, neighbor_threshold); 
    double energy2_bond = config2.getBondEnergy(fene_params); 
    double energy2_angle = config2.getBondAngleEnergy(AngleMode::COSINE, angle_params); 
    double energy2_dihedral = config2.getDihedralAngleEnergy(dihedral_params);
    double energy1_total = energy1_nonbonded + energy1_bond + energy1_angle + energy1_dihedral;
    double energy2_total = energy2_nonbonded + energy2_bond + energy2_angle + energy2_dihedral;

    // Check that the segment replacement energy difference is equal to the
    // energy difference between the two configurations 
    REQUIRE_THAT(
        replace_energy_12,
        Catch::Matchers::WithinAbs(energy2_total - energy1_total, tol)
    ); 

    // Randomly perturb atoms 2, 3, 4, 5 in the original configuration 
    Matrix<double, Dynamic, 3> coords3(coords); 
    for (int i = 2; i < 6; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            double r = -0.01 + 0.02 * uniform_dist(rng);  
            coords3(i, j) += r; 
        }
    }
    PolymerConfiguration<double> config3(coords3, Units::NANO, 300);

    // Calculate the segment replacement energy difference
    double replace_energy_13 = config.getSegmentReplacementEnergyDifference(
        coords3(Eigen::seqN(2, 4), Eigen::all), 2, 
        lj_params, neighbor_threshold, fene_params, AngleMode::COSINE, 
        angle_params, dihedral_params
    );

    // Calculate the reverse segment replacement energy difference
    double replace_energy_31 = config3.getSegmentReplacementEnergyDifference(
        coords(Eigen::seqN(2, 4), Eigen::all), 2, 
        lj_params, neighbor_threshold, fene_params, AngleMode::COSINE, 
        angle_params, dihedral_params
    );
    REQUIRE_THAT(replace_energy_13, Catch::Matchers::WithinAbs(-replace_energy_31, tol)); 

    // Calculate the total energy of the two configurations 
    double energy3_nonbonded = config3.getNonbondedEnergy(lj_params, neighbor_threshold); 
    double energy3_bond = config3.getBondEnergy(fene_params); 
    double energy3_angle = config3.getBondAngleEnergy(AngleMode::COSINE, angle_params); 
    double energy3_dihedral = config3.getDihedralAngleEnergy(dihedral_params);
    double energy3_total = energy3_nonbonded + energy3_bond + energy3_angle + energy3_dihedral;

    // Check that the segment replacement energy difference is equal to the
    // energy difference between the two configurations 
    REQUIRE_THAT(
        replace_energy_13,
        Catch::Matchers::WithinAbs(energy3_total - energy1_total, tol)
    ); 

    // Parse test 10-mer coordinates with angles chosen from a Gaussian potential
    result = parseLammps<double>(
        "configs/test_10mer_gaussian.txt", Units::NANO, 300
    ); 
    PolymerConfiguration<double> config4 = std::get<0>(result); 
    lj_params = std::get<1>(result); 
    fene_params = std::get<2>(result); 
    angle_params = std::get<4>(result); 
    dihedral_params = std::get<5>(result);

    // Randomly perturb the last 3 atoms in this configuration
    length = config.getLength(); 
    Matrix<double, Dynamic, 3> coords4 = config4.getSegment(0, length); 
    Matrix<double, Dynamic, 3> coords5(coords4); 
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            double r = -0.01 + 0.02 * uniform_dist(rng);  
            coords5(length - i - 1, j) += r;
        }
    } 
    PolymerConfiguration<double> config5(coords5, Units::NANO, 300);

    // Calculate the segment replacement energy difference
    neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"]; 
    double replace_energy_45 = config4.getSegmentReplacementEnergyDifference(
        coords5(Eigen::seqN(length - 3, 3), Eigen::all), length - 3, 
        lj_params, neighbor_threshold, fene_params, AngleMode::COSINE, 
        angle_params, dihedral_params
    );

    // Calculate the reverse segment replacement energy difference
    double replace_energy_54 = config5.getSegmentReplacementEnergyDifference(
        coords4(Eigen::seqN(length - 3, 3), Eigen::all), length - 3, 
        lj_params, neighbor_threshold, fene_params, AngleMode::COSINE, 
        angle_params, dihedral_params
    );
    REQUIRE_THAT(replace_energy_45, Catch::Matchers::WithinAbs(-replace_energy_54, tol)); 

    // Calculate the total energy of the two configurations 
    double energy4_nonbonded = config4.getNonbondedEnergy(lj_params, neighbor_threshold); 
    double energy4_bond = config4.getBondEnergy(fene_params); 
    double energy4_angle = config4.getBondAngleEnergy(AngleMode::COSINE, angle_params); 
    double energy4_dihedral = config4.getDihedralAngleEnergy(dihedral_params); 
    double energy5_nonbonded = config5.getNonbondedEnergy(lj_params, neighbor_threshold); 
    double energy5_bond = config5.getBondEnergy(fene_params); 
    double energy5_angle = config5.getBondAngleEnergy(AngleMode::COSINE, angle_params); 
    double energy5_dihedral = config5.getDihedralAngleEnergy(dihedral_params);
    double energy4_total = energy4_nonbonded + energy4_bond + energy4_angle + energy4_dihedral;
    double energy5_total = energy5_nonbonded + energy5_bond + energy5_angle + energy5_dihedral;

    // Check that the segment replacement energy difference is equal to the
    // energy difference between the two configurations 
    REQUIRE_THAT(
        replace_energy_45,
        Catch::Matchers::WithinAbs(energy5_total - energy4_total, tol)
    ); 

    // Randomly perturb atoms 2, 3, 4, 5 in the original configuration 
    Matrix<double, Dynamic, 3> coords6(coords4); 
    for (int i = 2; i < 6; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            double r = -0.01 + 0.02 * uniform_dist(rng);  
            coords6(i, j) += r; 
        }
    }
    PolymerConfiguration<double> config6(coords6, Units::NANO, 300);

    // Calculate the segment replacement energy difference
    double replace_energy_46 = config4.getSegmentReplacementEnergyDifference(
        coords6(Eigen::seqN(2, 4), Eigen::all), 2, 
        lj_params, neighbor_threshold, fene_params, AngleMode::COSINE, 
        angle_params, dihedral_params
    );

    // Calculate the reverse segment replacement energy difference
    double replace_energy_64 = config6.getSegmentReplacementEnergyDifference(
        coords4(Eigen::seqN(2, 4), Eigen::all), 2, 
        lj_params, neighbor_threshold, fene_params, AngleMode::COSINE, 
        angle_params, dihedral_params
    );
    REQUIRE_THAT(replace_energy_46, Catch::Matchers::WithinAbs(-replace_energy_64, tol)); 

    // Calculate the total energy of the two configurations 
    double energy6_nonbonded = config6.getNonbondedEnergy(lj_params, neighbor_threshold); 
    double energy6_bond = config6.getBondEnergy(fene_params); 
    double energy6_angle = config6.getBondAngleEnergy(AngleMode::COSINE, angle_params); 
    double energy6_dihedral = config6.getDihedralAngleEnergy(dihedral_params);
    double energy6_total = energy6_nonbonded + energy6_bond + energy6_angle + energy6_dihedral;

    // Check that the segment replacement energy difference is equal to the
    // energy difference between the two configurations 
    REQUIRE_THAT(
        replace_energy_46,
        Catch::Matchers::WithinAbs(energy6_total - energy4_total, tol)
    );  
}

/**
 * Tests for getReptationNonbondedEnergyDifference() and getReptationEnergyDifference(). 
 */
TEST_CASE(
    "Tests for reptation energy difference calculation",
    "[getReptationNonbondedEnergyDifference(), getReptationEnergyDifference()]"
)
{
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;
    const double tol = 1e-8;

    // Parse test 10-mer coordinates with angles chosen from a cosine potential
    auto result = parseLammps<double>(
        "configs/test_10mer_cosine.txt", Units::NANO, 300
    ); 
    PolymerConfiguration<double> config = std::get<0>(result); 
    std::unordered_map<std::string, double> lj_params = std::get<1>(result); 
    std::unordered_map<std::string, double> fene_params = std::get<2>(result); 
    std::unordered_map<std::string, double> angle_params = std::get<4>(result); 
    std::unordered_map<std::string, double> dihedral_params = std::get<5>(result);

    // Introduce a new atom at the tail
    int length = config.getLength(); 
    Matrix<double, Dynamic, 3> coords = config.getSegment(0, length);
    double bond_length = 0.85 * fene_params["R0"];
    double angle = 1.05 * angle_params["theta0"]; 
    double dihedral = 0.95 * boost::math::constants::pi<double>(); 
    Matrix<double, 3, 1> r_tail = generateNextAtomDihedral<double>(
        coords.row(length - 3), coords.row(length - 2), coords.row(length - 1),
        bond_length, angle, dihedral, rng, uniform_dist, (dihedral > 0 ? 1 : -1)
    );

    // Generate a new reptated configuration 
    PolymerConfiguration<double> config2(config); 
    config2.reptateTowardsTail(r_tail); 

    // Calculate the reptation energy difference 
    double neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"]; 
    double reptate_energy_12 = config.getReptationEnergyDifference(
        ReptationDirection::TAIL, r_tail, lj_params, neighbor_threshold,
        fene_params, AngleMode::COSINE, angle_params, dihedral_params
    );

    // Calculate the total energy of the two configurations 
    double energy1_nonbonded = config.getNonbondedEnergy(lj_params, neighbor_threshold); 
    double energy1_bond = config.getBondEnergy(fene_params); 
    double energy1_angle = config.getBondAngleEnergy(AngleMode::COSINE, angle_params); 
    double energy1_dihedral = config.getDihedralAngleEnergy(dihedral_params); 
    double energy2_nonbonded = config2.getNonbondedEnergy(lj_params, neighbor_threshold); 
    double energy2_bond = config2.getBondEnergy(fene_params); 
    double energy2_angle = config2.getBondAngleEnergy(AngleMode::COSINE, angle_params); 
    double energy2_dihedral = config2.getDihedralAngleEnergy(dihedral_params);
    double energy1_total = energy1_nonbonded + energy1_bond + energy1_angle + energy1_dihedral;
    double energy2_total = energy2_nonbonded + energy2_bond + energy2_angle + energy2_dihedral;
    
    // Check that the reptation energy difference is equal to the energy 
    // difference between the two configurations
    REQUIRE_THAT(
        reptate_energy_12,
        Catch::Matchers::WithinAbs(energy2_total - energy1_total, tol)
    );

    // Check the non-bonded reptation energy difference 
    double nonbonded_reptate_energy_12 = config.getReptationNonbondedEnergyDifference(
        ReptationDirection::TAIL, r_tail, lj_params, neighbor_threshold
    );
    double energy1_nonbonded_nc = config.getNonbondedEnergy(
        lj_params, neighbor_threshold, true
    );
    double energy2_nonbonded_nc = config2.getNonbondedEnergy(
        lj_params, neighbor_threshold, true
    );
    REQUIRE_THAT(
        nonbonded_reptate_energy_12, 
        Catch::Matchers::WithinAbs(energy2_nonbonded_nc - energy1_nonbonded_nc, tol)
    );

    // Generate the reverse reptated configuration 
    PolymerConfiguration<double> config3(config2); 
    config3.reptateTowardsHead(coords.row(0));

    // Calculate the reptation energy difference and check that it is the 
    // negative of the previous reptation energy difference
    double reptate_energy_23 = config2.getReptationEnergyDifference(
        ReptationDirection::HEAD, coords.row(0), lj_params, neighbor_threshold,
        fene_params, AngleMode::COSINE, angle_params, dihedral_params
    );
    REQUIRE_THAT(reptate_energy_23, Catch::Matchers::WithinAbs(-reptate_energy_12, tol));
    double nonbonded_reptate_energy_23 = config2.getReptationNonbondedEnergyDifference(
        ReptationDirection::HEAD, coords.row(0), lj_params, neighbor_threshold
    ); 
    REQUIRE_THAT(
        nonbonded_reptate_energy_23,
        Catch::Matchers::WithinAbs(-nonbonded_reptate_energy_12, tol)
    ); 

    // Introduce a new atom at the head
    bond_length = 0.82 * fene_params["R0"];
    angle = 1.3 * angle_params["theta0"]; 
    dihedral = 0.87 * boost::math::constants::pi<double>(); 
    Matrix<double, 3, 1> r_head = generateNextAtomDihedral<double>(
        coords.row(2), coords.row(1), coords.row(0),
        bond_length, angle, dihedral, rng, uniform_dist, (dihedral > 0 ? 1 : -1)
    );

    // Generate a new reptated configuration (from the original) 
    PolymerConfiguration<double> config4(config); 
    config4.reptateTowardsHead(r_head); 

    // Calculate the reptation energy difference  
    double reptate_energy_14 = config.getReptationEnergyDifference(
        ReptationDirection::HEAD, r_head, lj_params, neighbor_threshold,
        fene_params, AngleMode::COSINE, angle_params, dihedral_params
    );

    // Calculate the total energy of the new configuration
    double energy4_nonbonded = config4.getNonbondedEnergy(lj_params, neighbor_threshold); 
    double energy4_bond = config4.getBondEnergy(fene_params); 
    double energy4_angle = config4.getBondAngleEnergy(AngleMode::COSINE, angle_params); 
    double energy4_dihedral = config4.getDihedralAngleEnergy(dihedral_params);
    double energy4_total = energy4_nonbonded + energy4_bond + energy4_angle + energy4_dihedral;

    // Check that the reptation energy difference is equal to the energy 
    // difference between the two configurations
    REQUIRE_THAT(
        reptate_energy_14,
        Catch::Matchers::WithinAbs(energy4_total - energy1_total, tol)
    );

    // Check the non-bonded reptation energy difference 
    double nonbonded_reptate_energy_14 = config.getReptationNonbondedEnergyDifference(
        ReptationDirection::HEAD, r_head, lj_params, neighbor_threshold
    );
    double energy4_nonbonded_nc = config4.getNonbondedEnergy(
        lj_params, neighbor_threshold, true
    );
    REQUIRE_THAT(
        nonbonded_reptate_energy_14, 
        Catch::Matchers::WithinAbs(energy4_nonbonded_nc - energy1_nonbonded_nc, tol)
    );

    // Generate the reverse reptated configuration 
    PolymerConfiguration<double> config5(config4); 
    config5.reptateTowardsTail(coords.row(length - 1));

    // Calculate the reptation energy difference and check that it is the 
    // negative of the previous reptation energy difference
    double reptate_energy_45 = config4.getReptationEnergyDifference(
        ReptationDirection::TAIL, coords.row(length - 1), lj_params,
        neighbor_threshold, fene_params, AngleMode::COSINE, angle_params,
        dihedral_params
    );
    REQUIRE_THAT(reptate_energy_45, Catch::Matchers::WithinAbs(-reptate_energy_14, tol));
    double nonbonded_reptate_energy_45 = config4.getReptationNonbondedEnergyDifference(
        ReptationDirection::TAIL, coords.row(length - 1), lj_params,
        neighbor_threshold
    ); 
    REQUIRE_THAT(
        nonbonded_reptate_energy_45,
        Catch::Matchers::WithinAbs(-nonbonded_reptate_energy_14, tol)
    ); 
}

/**
 * Tests for getMultimerReptationNonbondedEnergyDifference(). 
 */
TEST_CASE(
    "Tests for multimer reptation energy difference calculation",
    "[getMultimerReptationNonbondedEnergyDifference()]"
)
{
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;
    const double tol = 1e-8;

    // Parse test 10-mer coordinates with angles chosen from a cosine potential
    auto result = parseLammps<double>(
        "configs/test_10mer_cosine.txt", Units::NANO, 300
    ); 
    PolymerConfiguration<double> config = std::get<0>(result); 
    std::unordered_map<std::string, double> lj_params = std::get<1>(result); 
    std::unordered_map<std::string, double> fene_params = std::get<2>(result); 
    std::unordered_map<std::string, double> angle_params = std::get<4>(result); 
    std::unordered_map<std::string, double> dihedral_params = std::get<5>(result);

    // Introduce 3 new atoms at the tail
    int length = config.getLength(); 
    Matrix<double, Dynamic, 3> coords = config.getSegment(0, length);
    Matrix<double, 3, 1> bond_lengths, angles, dihedrals; 
    bond_lengths << 0.85 * fene_params["R0"],
                    0.92 * fene_params["R0"],
                    0.78 * fene_params["R0"];
    angles << 1.05 * angle_params["theta0"], 
              0.76 * angle_params["theta0"], 
              0.93 * angle_params["theta0"]; 
    dihedrals << 0.95 * boost::math::constants::pi<double>(),
                 0.97 * boost::math::constants::pi<double>(), 
                 -0.86 * boost::math::constants::pi<double>();
    Matrix<double, Dynamic, 3> segment(3, 3);
    segment.row(0) = generateNextAtomDihedral<double>(
        coords.row(length - 3), coords.row(length - 2), coords.row(length - 1),
        bond_lengths(0), angles(0), dihedrals(0), rng, uniform_dist,
        (dihedrals(0) > 0 ? 1 : -1)
    );
    segment.row(1) = generateNextAtomDihedral<double>(
        coords.row(length - 2), coords.row(length - 1), segment.row(0), 
        bond_lengths(1), angles(1), dihedrals(1), rng, uniform_dist, 
        (dihedrals(1) > 0 ? 1 : -1)
    ); 
    segment.row(2) = generateNextAtomDihedral<double>(
        coords.row(length - 1), segment.row(0), segment.row(1), 
        bond_lengths(2), angles(2), dihedrals(2), rng, uniform_dist, 
        (dihedrals(2) > 0 ? 1 : -1)
    ); 

    // Generate a new reptated configuration 
    PolymerConfiguration<double> config2(config); 
    config2.reptateTowardsTailMultimer(segment);

    // Calculate the non-bonded reptation energy difference 
    double neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"]; 
    double reptate_energy_12 = config.getMultimerReptationNonbondedEnergyDifference(
        ReptationDirection::TAIL, segment, lj_params, neighbor_threshold
    ); 

    // Calculate the non-bonded energies of the two configurations, minus 
    // contributions from consecutive pairs of atoms 
    double energy1_nonbonded = config.getNonbondedEnergy(lj_params, neighbor_threshold, true); 
    double energy2_nonbonded = config2.getNonbondedEnergy(lj_params, neighbor_threshold, true); 
    REQUIRE_THAT(
        reptate_energy_12,
        Catch::Matchers::WithinAbs(energy2_nonbonded - energy1_nonbonded, tol)
    );

    // Generate the reverse reptated configuration 
    PolymerConfiguration<double> config3(config2); 
    config3.reptateTowardsHeadMultimer(coords(Eigen::seqN(0, 3), Eigen::all)); 

    // Calculate the reptation energy difference and check that it is the 
    // negative of the previous reptation energy difference
    double reptate_energy_23 = config2.getMultimerReptationNonbondedEnergyDifference(
        ReptationDirection::HEAD, coords(Eigen::seqN(0, 3), Eigen::all),
        lj_params, neighbor_threshold
    );
    REQUIRE_THAT(reptate_energy_23, Catch::Matchers::WithinAbs(-reptate_energy_12, tol));

    // Introduce 3 new atoms at the tail
    bond_lengths << 0.93 * fene_params["R0"],
                    0.86 * fene_params["R0"],
                    0.84 * fene_params["R0"];
    angles << 0.99 * angle_params["theta0"], 
              0.85 * angle_params["theta0"], 
              1.1 * angle_params["theta0"]; 
    dihedrals << 0.87 * boost::math::constants::pi<double>(),
                 0.94 * boost::math::constants::pi<double>(), 
                 0.88 * boost::math::constants::pi<double>();
    segment.row(2) = generateNextAtomDihedral<double>(
        coords.row(2), coords.row(1), coords.row(0), bond_lengths(0), angles(0),
        dihedrals(0), rng, uniform_dist, (dihedrals(0) > 0 ? 1 : -1)
    );
    segment.row(1) = generateNextAtomDihedral<double>(
        coords.row(1), coords.row(0), segment.row(2), bond_lengths(1), angles(1),
        dihedrals(1), rng, uniform_dist, (dihedrals(1) > 0 ? 1 : -1)
    ); 
    segment.row(0) = generateNextAtomDihedral<double>(
        coords.row(0), segment.row(2), segment.row(1), bond_lengths(2), angles(2),
        dihedrals(2), rng, uniform_dist, (dihedrals(2) > 0 ? 1 : -1)
    ); 

    // Generate a new reptated configuration (from the original) 
    PolymerConfiguration<double> config4(config); 
    config4.reptateTowardsHeadMultimer(segment);

    // Calculate the reptation energy difference  
    double reptate_energy_14 = config.getMultimerReptationNonbondedEnergyDifference(
        ReptationDirection::HEAD, segment, lj_params, neighbor_threshold
    );

    // Calculate the non-bonded energy of the new configuration, minus 
    // contributions from consecutive pairs of atoms 
    double energy4_nonbonded = config4.getNonbondedEnergy(lj_params, neighbor_threshold, true); 
    REQUIRE_THAT(
        reptate_energy_14,
        Catch::Matchers::WithinAbs(energy4_nonbonded - energy1_nonbonded, tol)
    );

    // Generate the reverse reptated configuration 
    PolymerConfiguration<double> config5(config4); 
    config5.reptateTowardsTailMultimer(coords(Eigen::seqN(7, 3), Eigen::all));

    // Calculate the reptation energy difference and check that it is the 
    // negative of the previous reptation energy difference
    double reptate_energy_45 = config4.getMultimerReptationNonbondedEnergyDifference(
        ReptationDirection::TAIL, coords(Eigen::seqN(7, 3), Eigen::all),
        lj_params, neighbor_threshold
    );
    REQUIRE_THAT(reptate_energy_45, Catch::Matchers::WithinAbs(-reptate_energy_14, tol));
}

/**
 * Tests for tangentVectors() and getTangentVectorAutocorrelation(). 
 */
TEST_CASE(
    "Tests for tangent vector autocorrelation calculation", 
    "[tangentVectors(), getTangentVectorAutocorrelation()]"
)
{
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;
    const double tol = 1e-8;

    // Parse test 10-mer coordinates with angles chosen from a cosine potential
    auto result = parseLammps<double>(
        "configs/test_10mer_cosine.txt", Units::NANO, 300
    ); 
    PolymerConfiguration<double> config = std::get<0>(result); 
    std::unordered_map<std::string, double> lj_params = std::get<1>(result); 
    std::unordered_map<std::string, double> fene_params = std::get<2>(result); 
    std::unordered_map<std::string, double> angle_params = std::get<4>(result); 
    std::unordered_map<std::string, double> dihedral_params = std::get<5>(result);

    // Get the tangent vectors along the polymer configuration 
    Matrix<double, Dynamic, 3> tangent_vectors = config.tangentVectors();
    REQUIRE(config.getLength() == 10); 
    REQUIRE(tangent_vectors.rows() == 9); 

    // Check that each tangent vector has norm 1
    for (int i = 0; i < 9; ++i)
        REQUIRE_THAT(
            tangent_vectors.row(i).norm(), Catch::Matchers::WithinAbs(1, tol)
        );

    // Check that each tangent vector is parallel to the corresponding bond
    Matrix<double, Dynamic, 3> coords = config.getSegment(0, 10); 
    for (int i = 0; i < 9; ++i)
    {
        Matrix<double, 3, 1> u = coords.row(i + 1) - coords.row(i);
        Matrix<double, 3, 1> v = tangent_vectors.row(i); 
        REQUIRE_THAT((u / u.norm() - v).norm(), Catch::Matchers::WithinAbs(0, tol)); 
    }

    // Generate three more slightly perturbed configurations 
    Matrix<double, Dynamic, 3> coords2(coords), coords3(coords), coords4(coords); 
    for (int i = 0; i < 10; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            coords2(i, j) += (-0.01 + 0.02 * uniform_dist(rng)); 
            coords3(i, j) += (-0.01 + 0.02 * uniform_dist(rng)); 
            coords4(i, j) += (-0.01 + 0.02 * uniform_dist(rng));
        }
    }
    PolymerConfiguration<double> config2(coords2, Units::NANO, 300), 
                                 config3(coords3, Units::NANO, 300), 
                                 config4(coords4, Units::NANO, 300);

    // Get the tangent vectors of all four configurations  
    std::vector<Matrix<double, Dynamic, 3> > tangent_vectors_ensemble;
    tangent_vectors_ensemble.push_back(tangent_vectors); 
    tangent_vectors_ensemble.push_back(config2.tangentVectors()); 
    tangent_vectors_ensemble.push_back(config3.tangentVectors()); 
    tangent_vectors_ensemble.push_back(config4.tangentVectors()); 

    // Get the tangent vector autocorrelation for k = 1
    Matrix<double, Dynamic, 1> autocorrs = getTangentVectorAutocorrelation<double>(
        tangent_vectors_ensemble, 1
    );
    
    // Verify the autocorrelation for each configuration ... 
    //
    // Get the average dot product between all pairs of consecutive tangent 
    // vectors along each configuration 
    for (int i = 0; i < 4; ++i)
    {
        double sum = 0;
        for (int j = 0; j < 8; ++j)
        {
            Matrix<double, 3, 1> u = tangent_vectors_ensemble[i].row(j); 
            Matrix<double, 3, 1> v = tangent_vectors_ensemble[i].row(j + 1); 
            sum += u.dot(v);
        }
        double mean = sum / 8;
        REQUIRE_THAT(autocorrs(i), Catch::Matchers::WithinAbs(mean, tol)); 
    }
}

