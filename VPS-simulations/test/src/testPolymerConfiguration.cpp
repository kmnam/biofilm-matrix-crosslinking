/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     5/20/2026
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
            theta, phi
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
        theta2, phi2
    ); 
    segment.row(2) = generateNextAtomDihedral<double>(
        config_coords.row(1), segment.row(0), segment.row(1), length,
        theta3, phi3
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
    "[appendSegmentToTail(), appendSegmentToHead(), popSegmentFromTail(), reptateTowardsTail(), reptateTowardsHead()]"
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
            theta, phi
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
        coords.row(5), coords.row(6), coords.row(7), length, theta1, phi1 
    ); 
    Matrix<double, 3, 1> tail2 = generateNextAtomDihedral<double>(
        coords.row(6), coords.row(7), tail1, length, theta2, phi2
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

    // Append a 4-atom segment to the head
    Matrix<double, Dynamic, 3> old_coords = new_coords; 
    double theta3 = 132 * boost::math::constants::pi<double>() / 180; 
    double theta4 = 156 * boost::math::constants::pi<double>() / 180;
    double theta5 = 98 * boost::math::constants::pi<double>() / 180; 
    double theta6 = 111 * boost::math::constants::pi<double>() / 180; 
    double phi3 = 88 * boost::math::constants::pi<double>() / 180; 
    double phi4 = 145 * boost::math::constants::pi<double>() / 180;
    double phi5 = -20 * boost::math::constants::pi<double>() / 180; 
    double phi6 = 63 * boost::math::constants::pi<double>() / 180; 
    Matrix<double, 3, 1> head1 = generateNextAtomDihedral<double>(
        old_coords.row(2), old_coords.row(1), old_coords.row(0), length,
        theta3, phi3
    ); 
    Matrix<double, 3, 1> head2 = generateNextAtomDihedral<double>(
        old_coords.row(1), old_coords.row(0), head1, length, theta4, phi4
    ); 
    Matrix<double, 3, 1> head3 = generateNextAtomDihedral<double>(
        old_coords.row(0), head1, head2, length, theta5, phi5
    ); 
    Matrix<double, 3, 1> head4 = generateNextAtomDihedral<double>(
        head1, head2, head3, length, theta6, phi6
    );

    // Update the polymer configuration
    Matrix<double, Dynamic, 3> head(4, 3); 
    head << head4(0), head4(1), head4(2), 
            head3(0), head3(1), head3(2),
            head2(0), head2(1), head2(2),
            head1(0), head1(1), head1(2); 
    config.appendSegmentToHead(head); 

    // Check the polymer length and coordinates 
    REQUIRE(config.getLength() == 14);
    new_coords = config.getSegment(0, 14);
    REQUIRE(new_coords.rows() == 14);
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
    for (int i = 4; i < 14; ++i)
    {
        REQUIRE_THAT(
            (new_coords.row(i) - old_coords.row(i - 4)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    }

    // Remove 3-atom segment from the tail 
    config.popSegmentFromTail(11);    // Remove indices 11, 12, 13

    // Check the polymer length and coordinates
    REQUIRE(config.getLength() == 11);
    old_coords = new_coords; 
    new_coords = config.getSegment(0, 11); 
    REQUIRE(new_coords.rows() == 11);
    for (int i = 0; i < 11; ++i)
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
        old_coords.row(5), old_coords.row(6), old_coords.row(7), length,
        theta7, phi7
    );
    config.reptateTowardsTail(tail3);

    // Check the polymer length and coordinates
    REQUIRE(config.getLength() == 11);
    new_coords = config.getSegment(0, 11); 
    REQUIRE(new_coords.rows() == 11);
    for (int i = 0; i < 10; ++i)
    {
        REQUIRE_THAT(
            (new_coords.row(i) - old_coords.row(i + 1)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    }
    REQUIRE_THAT(
        (new_coords.row(10) - tail3.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );

    // Reptate towards the head
    old_coords = new_coords; 
    double theta8 = 126 * boost::math::constants::pi<double>() / 180; 
    double phi8 = -2 * boost::math::constants::pi<double>() / 180;
    Matrix<double, 3, 1> head5 = generateNextAtomDihedral<double>(
        old_coords.row(2), old_coords.row(1), old_coords.row(0), length,
        theta8, phi8
    );
    config.reptateTowardsHead(head5);

    // Check the polymer length and coordinates
    REQUIRE(config.getLength() == 11);
    new_coords = config.getSegment(0, 11); 
    REQUIRE(new_coords.rows() == 11);
    REQUIRE_THAT(
        (new_coords.row(0) - head5.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );
    for (int i = 1; i < 11; ++i)
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
        old_coords.row(5), old_coords.row(6), old_coords.row(7), length,
        theta9, phi9
    ); 
    tail_segment.row(1) = generateNextAtomDihedral<double>(
        old_coords.row(6), old_coords.row(7), tail_segment.row(0), length,
        theta10, phi10
    ); 
    tail_segment.row(2) = generateNextAtomDihedral<double>(
        old_coords.row(7), tail_segment.row(0), tail_segment.row(1), length,
        theta11, phi11
    );
    config.reptateTowardsTailMultimer(tail_segment);

    // Check the polymer length and coordinates
    REQUIRE(config.getLength() == 11);
    new_coords = config.getSegment(0, 11); 
    REQUIRE(new_coords.rows() == 11);
    for (int i = 0; i < 8; ++i)
    {
        REQUIRE_THAT(
            (new_coords.row(i) - old_coords.row(i + 3)).norm(), 
            Catch::Matchers::WithinAbs(0, tol)
        );
    }
    REQUIRE_THAT(
        (new_coords.row(8) - tail_segment.row(0)).norm(), 
        Catch::Matchers::WithinAbs(0, tol)
    );
    REQUIRE_THAT(
        (new_coords.row(9) - tail_segment.row(1)).norm(), 
        Catch::Matchers::WithinAbs(0, tol)
    ); 
    REQUIRE_THAT(
        (new_coords.row(10) - tail_segment.row(2)).norm(),
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
        old_coords.row(2), old_coords.row(1), old_coords.row(0), length,
        theta12, phi12
    ); 
    head_segment.row(2) = generateNextAtomDihedral<double>(
        old_coords.row(1), old_coords.row(0), head_segment.row(3), length,
        theta13, phi13
    ); 
    head_segment.row(1) = generateNextAtomDihedral<double>(
        old_coords.row(0), head_segment.row(3), head_segment.row(2), length,
        theta14, phi14
    );
    head_segment.row(0) = generateNextAtomDihedral<double>(
        head_segment.row(3), head_segment.row(2), head_segment.row(1), length,
        theta15, phi15
    );
    config.reptateTowardsHeadMultimer(head_segment); 

    // Check the polymer length and coordinates
    REQUIRE(config.getLength() == 11);
    new_coords = config.getSegment(0, 11); 
    REQUIRE(new_coords.rows() == 11);
    for (int i = 0; i < 4; ++i)
    {
        REQUIRE_THAT(
            (new_coords.row(i) - head_segment.row(i)).norm(), 
            Catch::Matchers::WithinAbs(0, tol)
        );
    }
    for (int i = 4; i < 11; ++i)
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
    const double collision_threshold = 0.5 * pow(2., 1. / 6.) * lj_params["sigma"];
    const int max_tries_per_atom = 50;
    const int max_n_backtracks = 50; 
    Matrix<double, Dynamic, 2> bond_length_cdf = getFeneCDF<double>(
        lj_params["eps"], lj_params["sigma"], fene_params["K"], fene_params["R0"],
        kT, 10000
    );  

    // Generate a 10-mer
    PolymerConfiguration<double> config = generateKMer<double>(
        10, lj_params, fene_params, AngleMode::COSINE, cosine_params,
        dihedral_params, r0, collision_threshold, max_tries_per_atom,
        max_n_backtracks, rng, uniform_dist, bond_length_cdf
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
    const double collision_threshold = 0.5 * pow(2., 1. / 6.) * lj_params["sigma"];
    const int max_tries_per_atom = 50;
    const int max_n_backtracks = 50; 
    Matrix<double, Dynamic, 2> bond_length_cdf = getFeneCDF<double>(
        lj_params["eps"], lj_params["sigma"], fene_params["K"], fene_params["R0"],
        kT, 10000
    );  

    // Generate a 10-mer with a cosine angle potential
    PolymerConfiguration<double> config = generateKMer<double>(
        10, lj_params, fene_params, AngleMode::COSINE, cosine_params,
        dihedral_params, r0, collision_threshold, max_tries_per_atom,
        max_n_backtracks, rng, uniform_dist, bond_length_cdf
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
    config = generateKMer<double>(
        10, lj_params, fene_params, AngleMode::GAUSSIAN, gaussian_params,
        dihedral_params, r0, collision_threshold, max_tries_per_atom,
        max_n_backtracks, rng, uniform_dist, bond_length_cdf
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
 * Tests for getReptationResidualEnergy(). 
 */
TEST_CASE(
    "Tests for reptation energy difference calculation", "[getReptationResidualEnergy()]"
)
{
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;
    const double tol = 1e-3;

    // Generate a 10-mer with randomly chosen angles and dihedrals
    //
    // The chain is generated with no angle or dihedral potentials to allow 
    // for non-bonded interactions  
    const double kT = 1.380649e-2 * 300;
    std::unordered_map<std::string, double> lj_params,
                                            fene_params, 
                                            random_params, 
                                            nodihedral_params;  
    lj_params["eps"] = kT; 
    lj_params["sigma"] = 0.9;
    fene_params["K"] = 9 * kT; 
    fene_params["R0"] = 1.5;

    // Define a null cosine potential (to mimic random coils with excluded 
    // volume interactions)
    random_params["K"] = 0.0;
    random_params["theta0"] = boost::math::constants::pi<double>();

    // Define dihedral potential parameters  
    nodihedral_params["K"] = 0;

    // Define additional parameters for initialization and sampling 
    const double collision_threshold = 0.9;    // Slightly less than 2^(1/6) * sigma ~ 1.01
    const int max_tries_per_atom = 50;
    const int max_n_backtracks = 50;  
    Matrix<double, Dynamic, 2> bond_length_cdf = getFeneCDF<double>(
        lj_params["eps"], lj_params["sigma"], fene_params["K"], fene_params["R0"],
        kT, 10000
    );

    // Generate the 10-mer 
    const int length = 10;
    Matrix<double, 3, 1> r0 = Matrix<double, 3, 1>::Zero();  
    PolymerConfiguration<double> config = generateKMer<double>(
        length, lj_params, fene_params, AngleMode::COSINE, random_params, 
        nodihedral_params, r0, collision_threshold, max_tries_per_atom,
        max_n_backtracks, rng, uniform_dist, bond_length_cdf
    );
    Matrix<double, Dynamic, 3> coords = config.getSegment(0, length);  
    REQUIRE(config.getLength() == length);
    REQUIRE(coords.rows() == length);

    // Introduce a new atom at the tail
    //
    // Make it close to the existing polymer, so that we have clearly nonzero 
    // non-bonded interaction energies
    Matrix<double, 3, 1> r1;
    r1 << -collision_threshold + 2 * collision_threshold * uniform_dist(rng), 
          -collision_threshold + 2 * collision_threshold * uniform_dist(rng), 
          -collision_threshold + 2 * collision_threshold * uniform_dist(rng); 
    Matrix<double, 3, 1> r_tail = coords.row(length - 2) + r1.transpose(); 

    // Generate a new reptated configuration 
    PolymerConfiguration<double> config_reptated(config); 
    config_reptated.reptateTowardsTail(r_tail);
    Matrix<double, Dynamic, 3> coords_reptated = config_reptated.getSegment(0, length);  

    // Check the residual energy due to reptation
    //
    // This should be the non-bonded energy between atoms 1, ..., 8 in the
    // original configuration and the new atom 
    double neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"]; 
    double residual_reptate = config.getReptationResidualEnergy(
        r_tail, lj_params, neighbor_threshold
    );
    double sum = 0; 
    for (int i = 1; i < length - 1; ++i)
    {
        double dist = (r_tail - coords.row(i).transpose()).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(residual_reptate, Catch::Matchers::WithinAbs(sum, tol)); 

    // Generate the reverse reptated configuration
    Matrix<double, 3, 1> r_head_orig = coords.row(0);  
    PolymerConfiguration<double> config_reversed(config_reptated); 
    config_reversed.reptateTowardsHead(r_head_orig); 
    Matrix<double, Dynamic, 3> coords_reversed = config_reversed.getSegment(0, length); 

    // Check the residual energy due to reptation
    //
    // This should be the non-bonded energy between atoms 1, ..., 8 in the
    // reptated configuration and the new atom (atom 0 in the original
    // configuration) 
    //
    // This is equal to the non-bonded energy between atoms 2, ..., 9 in the 
    // original configuration and the new atom
    double residual_reverse = config_reptated.getReptationResidualEnergy(
        r_head_orig, lj_params, neighbor_threshold
    );
    sum = 0; 
    for (int i = 1; i < length - 1; ++i)
    {
        double dist = (r_head_orig.transpose() - coords_reptated.row(i)).norm();
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(residual_reverse, Catch::Matchers::WithinAbs(sum, tol));
    sum = 0; 
    for (int i = 2; i < length; ++i)
    {
        double dist = (r_head_orig.transpose() - coords.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(residual_reverse, Catch::Matchers::WithinAbs(sum, tol));

    // The reptation residual energy (1 -> 2) minus the reverse residual energy
    // (2 -> 3 = 1) must be equal to the total nonbonded energy difference
    // between 2 and 1
    double energy1_nonbonded_nc = config.getNonbondedEnergy(
        lj_params, neighbor_threshold, true
    );
    double energy2_nonbonded_nc = config_reptated.getNonbondedEnergy(
        lj_params, neighbor_threshold, true
    );
    double energy_nonbonded_diff = energy2_nonbonded_nc - energy1_nonbonded_nc; 
    REQUIRE_THAT(
        residual_reptate - residual_reverse,
        Catch::Matchers::WithinAbs(energy_nonbonded_diff, tol)
    ); 

    // Introduce a new atom at the head
    //
    // Make it close to the existing polymer, so that we have clearly nonzero 
    // non-bonded interaction energies
    r1 << -collision_threshold + 2 * collision_threshold * uniform_dist(rng), 
          -collision_threshold + 2 * collision_threshold * uniform_dist(rng), 
          -collision_threshold + 2 * collision_threshold * uniform_dist(rng); 
    Matrix<double, 3, 1> r_head = coords.row(1) + r1.transpose(); 

    // Generate a new reptated configuration (from the original) 
    config_reptated = config; 
    config_reptated.reptateTowardsHead(r_head);
    coords_reptated = config_reptated.getSegment(0, length);  

    // Check the residual energy due to reptation
    //
    // This should be the non-bonded energy between atoms 1, ..., 8 in the
    // original configuration and the new atom  
    residual_reptate = config.getReptationResidualEnergy(
        r_head, lj_params, neighbor_threshold
    ); 
    sum = 0; 
    for (int i = 1; i < length - 1; ++i)
    {
        double dist = (r_head - coords.row(i).transpose()).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(residual_reptate, Catch::Matchers::WithinAbs(sum, tol)); 

    // Generate the reverse reptated configuration
    Matrix<double, 3, 1> r_tail_orig = coords.row(length - 1);  
    config_reversed = config_reptated; 
    config_reversed.reptateTowardsTail(r_tail_orig);
    coords_reversed = config_reversed.getSegment(0, length);  

    // Check the residual energy due to reptation
    //
    // This should be the non-bonded energy between atoms 1, ..., 8 in the
    // new configuration and the new atom (atom 9 in the original configuration) 
    //
    // This is equal to the non-bonded energy between atoms 0, ..., 7 in the 
    // original configuration and the new atom 
    residual_reverse = config_reptated.getReptationResidualEnergy(
        r_tail_orig, lj_params, neighbor_threshold
    );  
    sum = 0;
    for (int i = 0; i < length - 2; ++i)
    {
        double dist = (r_tail_orig.transpose() - coords.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(residual_reverse, Catch::Matchers::WithinAbs(sum, tol));
    sum = 0;
    for (int i = 1; i < length - 1; ++i)
    {
        double dist = (r_tail_orig.transpose() - coords_reptated.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(residual_reverse, Catch::Matchers::WithinAbs(sum, tol));

    // The reptation residual energy (1 -> 4) minus the reverse residual energy
    // (4 -> 5 = 1) must be equal to the total nonbonded energy difference
    // between 4 and 1
    double energy4_nonbonded_nc = config_reptated.getNonbondedEnergy(
        lj_params, neighbor_threshold, true
    );
    energy_nonbonded_diff = energy4_nonbonded_nc - energy1_nonbonded_nc; 
    REQUIRE_THAT(
        residual_reptate - residual_reverse,
        Catch::Matchers::WithinAbs(energy_nonbonded_diff, tol)
    );
}

/**
 * Tests for getMultimerReptationResidualEnergy(). 
 */
TEST_CASE(
    "Tests for multimer reptation residual energy calculation",
    "[getMultimerReptationResidualEnergy()]"
)
{
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;
    const double tol = 1e-3;

    // Generate a 10-mer with randomly chosen angles and dihedrals 
    //
    // The bond lengths are chosen to be smaller than the Lennard-Jones 
    // length scale, so that we have clearly nonzero non-bonded interaction 
    // energies
    const double kT = 1.380649e-2 * 300;
    std::unordered_map<std::string, double> lj_params;
    lj_params["eps"] = kT; 
    lj_params["sigma"] = 0.9;
    const double scale = 0.9 * lj_params["sigma"];
    const int length = 10; 
    Matrix<double, Dynamic, 3> coords1(length, 3);
    coords1.row(0) = Matrix<double, 3, 1>::Zero();
    coords1(1, 0) = scale; 
    coords1(1, 1) = 0; 
    coords1(1, 2) = 0;
    for (int i = 2; i < length; ++i)
    {
        double angle = (
            -boost::math::constants::third_pi<double>() +
            boost::math::constants::two_thirds_pi<double>() * uniform_dist(rng)
        );  
        coords1.row(i) = generateNextAtom<double>(
            coords1.row(i - 2), coords1.row(i - 1), scale, angle, rng,
            uniform_dist
        );
    }
    PolymerConfiguration<double> config1(coords1, Units::NANO, 300.0);  

    // Introduce 3 new atoms at the tail
    //
    // Make them close to the existing polymer, so that we have clearly nonzero 
    // non-bonded interaction energies 
    Matrix<double, Dynamic, 3> segment(3, 3);
    Matrix<double, 3, 1> r1, r2, r3;
    r1 << -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng); 
    r2 << -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng); 
    r3 << -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng); 
    segment.row(0) = coords1.row(length - 2) + r1.transpose();
    segment.row(1) = coords1.row(length - 3) + r2.transpose(); 
    segment.row(2) = coords1.row(length - 4) + r3.transpose(); 

    // Generate a new reptated configuration 
    PolymerConfiguration<double> config2(config1); 
    config2.reptateTowardsTailMultimer(segment);
    double neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"];
    REQUIRE(config2.getNonbondedEnergy(lj_params, neighbor_threshold) > 0);

    // Calculate the reptation residual energy for each atom in the segment 
    //
    // This should be the non-bonded energy between each atom with index > 2
    // (0-indexed) in the original configuration and the new atom, minus the
    // adjacent atom  
    //
    // Calculate residual energy for i = 0 
    double reptate_residual_energy_12_0 = config1.getMultimerReptationResidualEnergy(
        ReptationDirection::TAIL, 3, 0,
        segment(Eigen::seqN(0, 0), Eigen::all), segment.row(0), lj_params,
        neighbor_threshold
    );
    double sum = 0; 
    for (int i = 3; i < 9; ++i)    // Omit the first three atoms and atom 9
    {
        double dist = (segment.row(0) - coords1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(reptate_residual_energy_12_0, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 1
    double reptate_residual_energy_12_1 = config1.getMultimerReptationResidualEnergy(
        ReptationDirection::TAIL, 3, 1,
        segment(Eigen::seqN(0, 1), Eigen::all), segment.row(1), lj_params,
        neighbor_threshold
    );
    sum = 0; 
    for (int i = 3; i < 10; ++i)    // Omit the first three atoms
    {
        double dist = (segment.row(1) - coords1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(reptate_residual_energy_12_1, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 2
    double reptate_residual_energy_12_2 = config1.getMultimerReptationResidualEnergy(
        ReptationDirection::TAIL, 3, 2,
        segment(Eigen::seqN(0, 2), Eigen::all), segment.row(2), lj_params,
        neighbor_threshold
    );
    sum = 0; 
    for (int i = 3; i < 10; ++i)    // Omit the first three atoms
    {
        double dist = (segment.row(2) - coords1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    sum += lj<double>(    // Add in atom 0 in the segment 
        (segment.row(2) - segment.row(0)).norm(), 
        lj_params["eps"], lj_params["sigma"], true
    ); 
    REQUIRE_THAT(reptate_residual_energy_12_2, Catch::Matchers::WithinAbs(sum, tol));

    // Generate the reverse reptated configuration 
    PolymerConfiguration<double> config3(config2); 
    config3.reptateTowardsHeadMultimer(coords1(Eigen::seqN(0, 3), Eigen::all));

    // Calculate the reptation residual energy for each atom in the original
    // segment 
    //
    // This should be the non-bonded energy between each atom with index < 7
    // (0-indexed) in the new configuration and the new atom, minus the
    // adjacent atom 
    //
    // Calculate residual energy for i = 0, which is atom 2 in the original 
    // configuration
    Matrix<double, Dynamic, 3> coords2 = config2.getSegment(0, 10);  
    Matrix<double, Dynamic, 3> subsegment(0, 3);  
    double reptate_residual_energy_23_0 = config2.getMultimerReptationResidualEnergy(
        ReptationDirection::HEAD, 3, 0, subsegment, coords1.row(2), lj_params,
        neighbor_threshold
    );
    sum = 0; 
    for (int i = 1; i < 7; ++i)    // Omit the last three atoms and atom 0 in the new configuration 
    {
        double dist = (coords1.row(2) - coords2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(reptate_residual_energy_23_0, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 1, which is atom 1 in the original
    // configuration
    subsegment.resize(1, 3); 
    subsegment.row(0) = coords1.row(2);  
    double reptate_residual_energy_23_1 = config2.getMultimerReptationResidualEnergy(
        ReptationDirection::HEAD, 3, 1, subsegment, coords1.row(1), lj_params,
        neighbor_threshold
    );
    sum = 0; 
    for (int i = 0; i < 7; ++i)    // Omit the last three atoms in the new configuration
    {
        double dist = (coords1.row(1) - coords2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(reptate_residual_energy_23_1, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 2, which is atom 0 in the original
    // configuration 
    subsegment.conservativeResize(2, 3);
    subsegment.row(1) = coords1.row(1);  
    double reptate_residual_energy_23_2 = config2.getMultimerReptationResidualEnergy(
        ReptationDirection::HEAD, 3, 2, subsegment, coords1.row(0), lj_params,
        neighbor_threshold
    );
    sum = 0; 
    for (int i = 0; i < 7; ++i)    // Omit the last three atoms in the new configuration
    {
        double dist = (coords1.row(0) - coords2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    sum += lj<double>(    // Add in atom 2 in the segment 
        (coords1.row(0) - coords1.row(2)).norm(), 
        lj_params["eps"], lj_params["sigma"], true
    );
    REQUIRE_THAT(reptate_residual_energy_23_2, Catch::Matchers::WithinAbs(sum, tol));

    // The total reptation residual energy (1 -> 2) minus the total reverse
    // residual energy (2 -> 3 = 1) must be equal to the total nonbonded
    // energy difference between 2 and 1
    double energy1_nonbonded_nc = config1.getNonbondedEnergy(
        lj_params, neighbor_threshold, true
    );
    double energy2_nonbonded_nc = config2.getNonbondedEnergy(
        lj_params, neighbor_threshold, true
    );
    double energy_nonbonded_diff = energy2_nonbonded_nc - energy1_nonbonded_nc;
    double total_residual_energy_12 = (
        reptate_residual_energy_12_0 + reptate_residual_energy_12_1 +
        reptate_residual_energy_12_2
    ); 
    double total_residual_energy_23 = (
        reptate_residual_energy_23_0 + reptate_residual_energy_23_1 +
        reptate_residual_energy_23_2
    );  
    REQUIRE_THAT(
        total_residual_energy_12 - total_residual_energy_23,
        Catch::Matchers::WithinAbs(energy_nonbonded_diff, tol)
    ); 
    
    // Introduce 3 new atoms at the head
    //
    // Again make them close to the existing polymer, so that we have clearly
    // nonzero non-bonded interaction energies 
    r1 << -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng); 
    r2 << -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng); 
    r3 << -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng); 
    segment.row(2) = coords1.row(1) + r1.transpose();    // New atom 2
    segment.row(1) = coords1.row(2) + r2.transpose();    // New atom 1
    segment.row(0) = coords1.row(3) + r3.transpose();    // New atom 0

    // Generate a new reptated configuration (from the original) 
    PolymerConfiguration<double> config4(config1); 
    config4.reptateTowardsHeadMultimer(segment);
    REQUIRE(config4.getNonbondedEnergy(lj_params, neighbor_threshold) > 0);

    // Calculate the reptation residual energy for each atom in the segment 
    //
    // This should be the non-bonded energy between each atom with index < 7
    // (0-indexed) in the original configuration and the new atom, minus the
    // adjacent atom 
    //
    // Calculate residual energy for i = 0, which corresponds to atom 2 in
    // the new segment
    subsegment.resize(0, 3); 
    double reptate_residual_energy_14_0 = config1.getMultimerReptationResidualEnergy(
        ReptationDirection::HEAD, 3, 0, subsegment, segment.row(2), lj_params,
        neighbor_threshold
    );
    sum = 0;
    for (int i = 1; i < 7; ++i)    // Omit the last three atoms and atom 0
    {
        double dist = (segment.row(2) - coords1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(reptate_residual_energy_14_0, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 1, which corresponds to atom 1 in
    // the new segment
    subsegment.resize(1, 3); 
    subsegment.row(0) = segment.row(2); 
    double reptate_residual_energy_14_1 = config1.getMultimerReptationResidualEnergy(
        ReptationDirection::HEAD, 3, 1, subsegment, segment.row(1), lj_params,
        neighbor_threshold
    );
    sum = 0;
    for (int i = 0; i < 7; ++i)    // Omit the last three atoms
    {
        double dist = (segment.row(1) - coords1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(reptate_residual_energy_14_1, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 2, which corresponds to atom 0 in
    // the new segment
    subsegment.conservativeResize(2, 3); 
    subsegment.row(1) = segment.row(1); 
    double reptate_residual_energy_14_2 = config1.getMultimerReptationResidualEnergy(
        ReptationDirection::HEAD, 3, 2, subsegment, segment.row(0), lj_params,
        neighbor_threshold
    );
    sum = 0;
    for (int i = 0; i < 7; ++i)    // Omit the last three atoms
    {
        double dist = (segment.row(0) - coords1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    sum += lj<double>(    // Add in atom 2 in the segment 
        (segment.row(0) - segment.row(2)).norm(), 
        lj_params["eps"], lj_params["sigma"], true
    );
    REQUIRE_THAT(reptate_residual_energy_14_2, Catch::Matchers::WithinAbs(sum, tol));

    // Generate the reverse reptated configuration 
    PolymerConfiguration<double> config5(config4); 
    config5.reptateTowardsTailMultimer(coords1(Eigen::seqN(7, 3), Eigen::all));

    // Calculate the reptation residual energy for each atom in the original
    // segment 
    //
    // This should be the non-bonded energy between each atom with index > 2
    // (0-indexed) in the new configuration and the new atom, minus the
    // adjacent atom 
    //
    // Calculate residual energy for i = 0, which is atom 7 in the original 
    // configuration
    Matrix<double, Dynamic, 3> coords4 = config4.getSegment(0, 10);  
    double reptate_residual_energy_45_0 = config4.getMultimerReptationResidualEnergy(
        ReptationDirection::TAIL, 3, 0, coords1(Eigen::seqN(7, 0), Eigen::all),
        coords1.row(7), lj_params, neighbor_threshold
    );
    sum = 0; 
    for (int i = 3; i < 9; ++i)    // Omit first three atoms and atom 9 in the new configuration 
    {
        double dist = (coords1.row(7) - coords4.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(reptate_residual_energy_45_0, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 1, which is atom 8 in the original 
    // configuration 
    double reptate_residual_energy_45_1 = config4.getMultimerReptationResidualEnergy(
        ReptationDirection::TAIL, 3, 1, coords1(Eigen::seqN(7, 1), Eigen::all),
        coords1.row(8), lj_params, neighbor_threshold
    );
    sum = 0; 
    for (int i = 3; i < 10; ++i)    // Omit the first three atoms
    {
        double dist = (coords1.row(8) - coords4.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(reptate_residual_energy_45_1, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 2, which is atom 9 in the original 
    // configuration 
    double reptate_residual_energy_45_2 = config4.getMultimerReptationResidualEnergy(
        ReptationDirection::TAIL, 3, 2, coords1(Eigen::seqN(7, 2), Eigen::all),
        coords1.row(9), lj_params, neighbor_threshold
    );
    sum = 0; 
    for (int i = 3; i < 10; ++i)    // Omit the first three atoms
    {
        double dist = (coords1.row(9) - coords4.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    sum += lj<double>(    // Add in atom 0 in the segment (atom 7 in the original configuration) 
        (coords1.row(9) - coords1.row(7)).norm(), 
        lj_params["eps"], lj_params["sigma"], true
    );
    REQUIRE_THAT(reptate_residual_energy_45_2, Catch::Matchers::WithinAbs(sum, tol));
}

/**
 * Tests for getTerminalSegmentReplacementResidualEnergy().
 */
TEST_CASE(
    "Tests for terminal segment replacement residual energy calculation",
    "[getTerminalSegmentReplacementResidualEnergy()]"
)
{
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;
    const double tol = 1e-3;

    // Generate a 10-mer with randomly chosen angles and dihedrals 
    //
    // The bond lengths are chosen to be smaller than the Lennard-Jones 
    // length scale, so that we have clearly nonzero non-bonded interaction 
    // energies
    const double kT = 1.380649e-2 * 300;
    std::unordered_map<std::string, double> lj_params;
    lj_params["eps"] = kT; 
    lj_params["sigma"] = 0.9;
    const double scale = 0.9 * lj_params["sigma"];
    const int length = 10; 
    Matrix<double, Dynamic, 3> coords1(length, 3);
    coords1.row(0) = Matrix<double, 3, 1>::Zero();
    coords1(1, 0) = scale; 
    coords1(1, 1) = 0; 
    coords1(1, 2) = 0;
    for (int i = 2; i < length; ++i)
    {
        double angle = (
            -boost::math::constants::third_pi<double>() +
            boost::math::constants::two_thirds_pi<double>() * uniform_dist(rng)
        );  
        coords1.row(i) = generateNextAtom<double>(
            coords1.row(i - 2), coords1.row(i - 1), scale, angle, rng,
            uniform_dist
        );
    }
    PolymerConfiguration<double> config1(coords1, Units::NANO, 300.0);  

    // Introduce 3 new atoms at the tail
    //
    // Make them close to atoms 4, 5, 6 in the existing polymer, so that we
    // have clearly nonzero non-bonded interaction energies 
    Matrix<double, Dynamic, 3> segment(3, 3);
    Matrix<double, 3, 1> r1, r2, r3;
    r1 << -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng); 
    r2 << -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng); 
    r3 << -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng); 
    segment.row(0) = coords1.row(length - 4) + r1.transpose();
    segment.row(1) = coords1.row(length - 5) + r2.transpose(); 
    segment.row(2) = coords1.row(length - 6) + r3.transpose(); 

    // Generate a new configuration with the terminal segment 
    PolymerConfiguration<double> config2(config1); 
    config2.replaceSegment(segment, length - 3);
    double neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"];
    REQUIRE(config2.getNonbondedEnergy(lj_params, neighbor_threshold) > 0);  

    // Calculate the terminal segment replacement residual energy for each
    // atom in the segment 
    //
    // This should be the non-bonded energy between each atom with index < 7
    // (0-indexed) in the original configuration and the new atom, minus the
    // adjacent atom  
    //
    // Calculate residual energy for i = 0 
    double terminal_residual_energy_12_0 = config1.getTerminalSegmentReplacementResidualEnergy(
        TerminalSegmentEnd::TAIL, 3, 0, 
        segment(Eigen::seqN(0, 0), Eigen::all), segment.row(0), lj_params,
        neighbor_threshold
    );
    double sum = 0; 
    for (int i = 0; i < 6; ++i)    // Omit the last four atoms (since atom 6 is bonded to the new atom)
    {
        double dist = (segment.row(0) - coords1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(terminal_residual_energy_12_0, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 1
    double terminal_residual_energy_12_1 = config1.getTerminalSegmentReplacementResidualEnergy(
        TerminalSegmentEnd::TAIL, 3, 1,
        segment(Eigen::seqN(0, 1), Eigen::all), segment.row(1), lj_params,
        neighbor_threshold
    );
    sum = 0; 
    for (int i = 0; i < 7; ++i)    // Omit the last three atoms
    {
        double dist = (segment.row(1) - coords1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(terminal_residual_energy_12_1, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 2
    double terminal_residual_energy_12_2 = config1.getTerminalSegmentReplacementResidualEnergy(
        TerminalSegmentEnd::TAIL, 3, 2, 
        segment(Eigen::seqN(0, 2), Eigen::all), segment.row(2), lj_params,
        neighbor_threshold
    );
    sum = 0; 
    for (int i = 0; i < 7; ++i)    // Omit the last three atoms
    {
        double dist = (segment.row(2) - coords1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    sum += lj<double>(    // Add in atom 0 in the segment 
        (segment.row(2) - segment.row(0)).norm(), 
        lj_params["eps"], lj_params["sigma"], true
    ); 
    REQUIRE_THAT(terminal_residual_energy_12_2, Catch::Matchers::WithinAbs(sum, tol));

    // Generate the reverse configuration
    PolymerConfiguration<double> config3(config2); 
    config3.replaceSegment(coords1(Eigen::seqN(length - 3, 3), Eigen::all), length - 3);

    // Calculate the terminal segment replacement residual energy for each
    // atom in the original segment 
    //
    // This should again be the non-bonded energy between each atom with
    // index < 7 (0-indexed) in the original configuration and the new atom,
    // minus the adjacent atom  
    //
    // Calculate residual energy for i = 0
    Matrix<double, Dynamic, 3> coords2 = config2.getSegment(0, 10);  
    Matrix<double, Dynamic, 3> subsegment(0, 3);  
    double terminal_residual_energy_23_0 = config2.getTerminalSegmentReplacementResidualEnergy(
        TerminalSegmentEnd::TAIL, 3, 0, subsegment, coords1.row(7), lj_params,
        neighbor_threshold
    );
    sum = 0; 
    for (int i = 0; i < 6; ++i)    // Omit the last four atoms 
    {
        double dist = (coords1.row(7) - coords2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(terminal_residual_energy_23_0, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 1
    subsegment.resize(1, 3); 
    subsegment.row(0) = coords1.row(7);  
    double terminal_residual_energy_23_1 = config2.getTerminalSegmentReplacementResidualEnergy(
        TerminalSegmentEnd::TAIL, 3, 1, subsegment, coords1.row(8), lj_params,
        neighbor_threshold
    );
    sum = 0; 
    for (int i = 0; i < 7; ++i)    // Omit the last three atoms
    {
        double dist = (coords1.row(8) - coords2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(terminal_residual_energy_23_1, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 2
    subsegment.conservativeResize(2, 3);
    subsegment.row(1) = coords1.row(8);  
    double terminal_residual_energy_23_2 = config2.getTerminalSegmentReplacementResidualEnergy(
        TerminalSegmentEnd::TAIL, 3, 2, subsegment, coords1.row(9), lj_params,
        neighbor_threshold
    );
    sum = 0; 
    for (int i = 0; i < 7; ++i)    // Omit the last three atoms
    {
        double dist = (coords1.row(9) - coords2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    sum += lj<double>(    // Add in atom 0 in the segment (atom 7 in the original configuration) 
        (coords1.row(9) - coords1.row(7)).norm(), 
        lj_params["eps"], lj_params["sigma"], true
    );
    REQUIRE_THAT(terminal_residual_energy_23_2, Catch::Matchers::WithinAbs(sum, tol));

    // The total terminal segment replacement residual energy (1 -> 2) minus 
    // the reverse residual energy (2 -> 3 = 1) must be equal to the total
    // nonbonded energy difference between 2 and 1
    double energy1_nonbonded_nc = config1.getNonbondedEnergy(
        lj_params, neighbor_threshold, true
    );
    double energy2_nonbonded_nc = config2.getNonbondedEnergy(
        lj_params, neighbor_threshold, true
    );
    double energy_nonbonded_diff = energy2_nonbonded_nc - energy1_nonbonded_nc;
    double total_residual_energy_12 = (
        terminal_residual_energy_12_0 + terminal_residual_energy_12_1 +
        terminal_residual_energy_12_2
    ); 
    double total_residual_energy_23 = (
        terminal_residual_energy_23_0 + terminal_residual_energy_23_1 +
        terminal_residual_energy_23_2
    );  
    REQUIRE_THAT(
        total_residual_energy_12 - total_residual_energy_23,
        Catch::Matchers::WithinAbs(energy_nonbonded_diff, tol)
    ); 

    // Introduce 3 new atoms at the head
    //
    // Again make them close to the existing polymer (here, atoms 3, 4, 5), so
    // that we have clearly nonzero non-bonded interaction energies 
    r1 << -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng); 
    r2 << -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng); 
    r3 << -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng); 
    segment.row(2) = coords1.row(3) + r1.transpose();    // New atom 2
    segment.row(1) = coords1.row(4) + r2.transpose();    // New atom 1
    segment.row(0) = coords1.row(5) + r3.transpose();    // New atom 0

    // Generate a new configuration (from the original) 
    PolymerConfiguration<double> config4(config1); 
    config4.replaceSegment(segment, 0);
    REQUIRE(config4.getNonbondedEnergy(lj_params, neighbor_threshold) > 0);  

    // Calculate the terminal segment replacement residual energy for each atom
    // in the segment 
    //
    // This should be the non-bonded energy between each atom with index > 2
    // (0-indexed) in the original configuration and the new atom, minus the
    // adjacent atom 
    //
    // Calculate residual energy for i = 0, which is atom 2 in the new segment
    subsegment.resize(0, 3);
    double terminal_residual_energy_14_0 = config1.getTerminalSegmentReplacementResidualEnergy(
        TerminalSegmentEnd::HEAD, 3, 0, subsegment, segment.row(2), lj_params,
        neighbor_threshold 
    );
    sum = 0;
    for (int i = 4; i < 10; ++i)    // Omit the first four atoms (since atom 3 is bonded to atom 2) 
    {
        double dist = (segment.row(2) - coords1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(terminal_residual_energy_14_0, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 1, which is atom 1 in the new segment
    subsegment.resize(1, 3); 
    subsegment.row(0) = segment.row(2); 
    double terminal_residual_energy_14_1 = config1.getTerminalSegmentReplacementResidualEnergy(
        TerminalSegmentEnd::HEAD, 3, 1, subsegment, segment.row(1), lj_params,
        neighbor_threshold
    );
    sum = 0;
    for (int i = 3; i < 10; ++i)    // Omit the first three atoms
    {
        double dist = (segment.row(1) - coords1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(terminal_residual_energy_14_1, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 2, which is atom 0 in the new segment 
    subsegment.conservativeResize(2, 3); 
    subsegment.row(1) = segment.row(1); 
    double terminal_residual_energy_14_2 = config1.getTerminalSegmentReplacementResidualEnergy(
        TerminalSegmentEnd::HEAD, 3, 2, subsegment, segment.row(0), lj_params,
        neighbor_threshold
    );
    sum = 0;
    for (int i = 3; i < 10; ++i)    // Omit the first three atoms
    {
        double dist = (segment.row(0) - coords1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    sum += lj<double>(    // Add in atom 2 in the segment 
        (segment.row(0) - segment.row(2)).norm(), 
        lj_params["eps"], lj_params["sigma"], true
    );
    REQUIRE_THAT(terminal_residual_energy_14_2, Catch::Matchers::WithinAbs(sum, tol));

    // Generate the reverse reptated configuration 
    PolymerConfiguration<double> config5(config4); 
    config5.replaceSegment(coords1(Eigen::seqN(0, 3), Eigen::all), 0); 

    // Calculate the terminal segment replacement residual energy for each
    // atom in the original segment 
    //
    // This should again be the non-bonded energy between each atom with
    // index > 2 (0-indexed) in the new configuration and the new atom, minus
    // the adjacent atom 
    //
    // Calculate residual energy for i = 0, which is atom 2 in the original 
    // configuration 
    Matrix<double, Dynamic, 3> coords4 = config4.getSegment(0, 10);
    subsegment.resize(0, 3);  
    double terminal_residual_energy_45_0 = config4.getTerminalSegmentReplacementResidualEnergy(
        TerminalSegmentEnd::HEAD, 3, 0, subsegment, coords1.row(2), lj_params,
        neighbor_threshold
    );
    sum = 0; 
    for (int i = 4; i < 10; ++i)    // Omit first four atoms (since atom 3 is bonded to atom 2)
    {
        double dist = (coords1.row(2) - coords4.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(terminal_residual_energy_45_0, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 1, which is atom 1 in the original 
    // configuration
    subsegment.resize(1, 3); 
    subsegment.row(0) = coords1.row(2);  
    double terminal_residual_energy_45_1 = config4.getTerminalSegmentReplacementResidualEnergy(
        TerminalSegmentEnd::HEAD, 3, 1, subsegment, coords1.row(1), lj_params,
        neighbor_threshold
    );
    sum = 0; 
    for (int i = 3; i < 10; ++i)    // Omit the first three atoms
    {
        double dist = (coords1.row(1) - coords4.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(terminal_residual_energy_45_1, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 2, which is atom 0 in the original 
    // configuration
    subsegment.conservativeResize(2, 3); 
    subsegment.row(1) = coords1.row(1);  
    double terminal_residual_energy_45_2 = config4.getTerminalSegmentReplacementResidualEnergy(
        TerminalSegmentEnd::HEAD, 3, 2, subsegment, coords1.row(0), lj_params,
        neighbor_threshold
    );
    sum = 0; 
    for (int i = 3; i < 10; ++i)    // Omit the first three atoms
    {
        double dist = (coords1.row(0) - coords4.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    sum += lj<double>(    // Add in atom 2 in the original configuration 
        (coords1.row(0) - coords1.row(2)).norm(), 
        lj_params["eps"], lj_params["sigma"], true
    );
    REQUIRE_THAT(terminal_residual_energy_45_2, Catch::Matchers::WithinAbs(sum, tol));
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

