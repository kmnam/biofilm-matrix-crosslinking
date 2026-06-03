/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     5/21/2026
 */

#include <iostream>
#include <cstdlib>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/random.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/utils.hpp"
#include "../../include/polymerMelt.hpp"

using namespace Eigen;

/**
 * Generate an example system of multiple polymers, atom by atom. 
 *
 * This function returns a vector of coordinate arrays, one for each polymer.   
 */
std::vector<Matrix<double, Dynamic, 3> > generateExampleMeltCoords(boost::random::mt19937& rng,
                                                                   boost::random::uniform_01<>& uniform_dist)
{
    // Generate three polymer chains ...
    Matrix<double, Dynamic, 3> coords1, coords2, coords3;  
    
    // Start with an 8-atom segment 
    const double length = 1.5; 
    Array<double, Dynamic, 1> angles(6);
    angles << 140 * boost::math::constants::pi<double>() / 180,
              170 * boost::math::constants::pi<double>() / 180,  
              160 * boost::math::constants::pi<double>() / 180,
              boost::math::constants::half_pi<double>(), 
              150 * boost::math::constants::pi<double>() / 180,
              130 * boost::math::constants::pi<double>() / 180;
    Array<double, Dynamic, 1> dihedrals(5);
    dihedrals << 60 * boost::math::constants::pi<double>() / 180, 
                 -60 * boost::math::constants::pi<double>() / 180,
                 boost::math::constants::pi<double>(),  
                 120 * boost::math::constants::pi<double>() / 180, 
                 60 * boost::math::constants::pi<double>() / 180;
    coords1.resize(8, 3); 
    coords1.row(0) = Matrix<double, 3, 1>::Zero();
    coords1(1, 0) = length; 
    coords1(1, 1) = 0; 
    coords1(1, 2) = 0;
    coords1.row(2) = generateNextAtom<double>(
        coords1.row(0), coords1.row(1), length, angles(0), rng, uniform_dist
    ); 
    for (int i = 3; i < 8; ++i) 
    {
        double theta = angles(i - 2); 
        double phi = dihedrals(i - 3); 
        coords1.row(i) = generateNextAtomDihedral<double>(
            coords1.row(i - 3), coords1.row(i - 2), coords1.row(i - 1), length,
            theta, phi
        );
    }

    // Then generate a 6-atom segment 
    angles = Array<double, Dynamic, 1>::Zero(4);
    angles << 130 * boost::math::constants::pi<double>() / 180, 
              170 * boost::math::constants::pi<double>() / 180, 
              150 * boost::math::constants::pi<double>() / 180, 
              160 * boost::math::constants::pi<double>() / 180;  
    dihedrals = Array<double, Dynamic, 1>::Zero(3);
    dihedrals << 72 * boost::math::constants::pi<double>() / 180, 
                 -40 * boost::math::constants::pi<double>() / 180, 
                 100 * boost::math::constants::pi<double>() / 180;
    coords2.resize(6, 3); 
    coords2(0, 0) = 4; 
    coords2(0, 1) = 5; 
    coords2(0, 2) = 6; 
    coords2(1, 0) = coords2(0, 0); 
    coords2(1, 1) = coords2(0, 1) + length;
    coords2(1, 2) = coords2(0, 2); 
    coords2.row(2) = generateNextAtom<double>(
        coords2.row(0), coords2.row(1), length, angles(0), rng, uniform_dist
    );
    for (int i = 3; i < 6; ++i) 
    {
        double theta = angles(i - 2); 
        double phi = dihedrals(i - 3); 
        coords2.row(i) = generateNextAtomDihedral<double>(
            coords2.row(i - 3), coords2.row(i - 2), coords2.row(i - 1), length,
            theta, phi
        );
    }

    // Then generate a 7-atom segment 
    angles = Array<double, Dynamic, 1>::Zero(5);
    angles << 120 * boost::math::constants::pi<double>() / 180, 
              110 * boost::math::constants::pi<double>() / 180, 
              140 * boost::math::constants::pi<double>() / 180, 
              170 * boost::math::constants::pi<double>() / 180, 
              160 * boost::math::constants::pi<double>() / 180; 
    dihedrals = Array<double, Dynamic, 1>::Zero(4);
    dihedrals << 48 * boost::math::constants::pi<double>() / 180, 
                 106 * boost::math::constants::pi<double>() / 180,  
                 -22 * boost::math::constants::pi<double>() / 180, 
                 57 * boost::math::constants::pi<double>() / 180;  
    coords3.resize(7, 3); 
    coords3(0, 0) = -5; 
    coords3(0, 1) = -4; 
    coords3(0, 2) = -3; 
    coords3(1, 0) = coords3(0, 0); 
    coords3(1, 1) = coords3(0, 1);
    coords3(1, 2) = coords3(0, 2) + length; 
    coords3.row(2) = generateNextAtom<double>(
        coords3.row(0), coords3.row(1), length, angles(0), rng, uniform_dist
    );
    for (int i = 3; i < 7; ++i) 
    {
        double theta = angles(i - 2); 
        double phi = dihedrals(i - 3); 
        coords3.row(i) = generateNextAtomDihedral<double>(
            coords3.row(i - 3), coords3.row(i - 2), coords3.row(i - 1), length,
            theta, phi
        );
    }

    // Generate the corresponding polymer melt 
    std::vector<Matrix<double, Dynamic, 3> > coords_all; 
    coords_all.push_back(coords1); 
    coords_all.push_back(coords2); 
    coords_all.push_back(coords3);
    return coords_all;  
}

/**
 * Tests for getting and replacing segments in the PolymerMeltConfiguration
 * class. 
 */
TEST_CASE(
    "Tests for getting and replacing segments", "[getSegment(), replaceSegment()]"
)
{
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;
    const double tol = 1e-5;

    // Define a large box
    const double xmin = -100.0; 
    const double xmax = 100.0; 
    const double ymin = -100.0; 
    const double ymax = 100.0;
    const double zmin = -100.0;
    const double zmax = 100.0;

    // Generate an example melt configuration
    auto coords_all = generateExampleMeltCoords(rng, uniform_dist);
    REQUIRE(coords_all.size() == 3); 
    Matrix<double, Dynamic, 3> coords1 = coords_all[0]; 
    Matrix<double, Dynamic, 3> coords2 = coords_all[1]; 
    Matrix<double, Dynamic, 3> coords3 = coords_all[2];  
    PolymerMeltConfiguration<double> melt_configs(
        3, coords_all, Units::NANO, 300, xmin, xmax, ymin, ymax, zmin, zmax
    ); 

    // Check the polymer lengths and coordinates 
    REQUIRE(melt_configs.getLength(0) == 8);
    REQUIRE(melt_configs.getLength(1) == 6); 
    REQUIRE(melt_configs.getLength(2) == 7); 
    Matrix<double, Dynamic, 3> config_coords1 = melt_configs.getSegment(0, 0, 8);
    Matrix<double, Dynamic, 3> config_coords2 = melt_configs.getSegment(1, 0, 6); 
    Matrix<double, Dynamic, 3> config_coords3 = melt_configs.getSegment(2, 0, 7); 
    REQUIRE(config_coords1.rows() == 8);
    REQUIRE(config_coords2.rows() == 6); 
    REQUIRE(config_coords3.rows() == 7); 
    for (int i = 0; i < 8; ++i)
        REQUIRE_THAT(
            (config_coords1.row(i) - coords1.row(i)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    for (int i = 0; i < 6; ++i)
        REQUIRE_THAT(
            (config_coords2.row(i) - coords2.row(i)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    for (int i = 0; i < 7; ++i)
        REQUIRE_THAT(
            (config_coords3.row(i) - coords3.row(i)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );

    // Replace atoms 2, 3, 4 in the second polymer
    const double length = 1.5; 
    double theta1 = 132 * boost::math::constants::pi<double>() / 180; 
    double theta2 = 153 * boost::math::constants::pi<double>() / 180;
    double theta3 = 165 * boost::math::constants::pi<double>() / 180; 
    double phi1 = 54 * boost::math::constants::pi<double>() / 180;
    double phi2 = -72 * boost::math::constants::pi<double>() / 180;
    Matrix<double, Dynamic, 3> new_segment(3, 3); 
    new_segment.row(0) = generateNextAtom<double>(
        coords2.row(0), coords2.row(1), length, theta1, rng, uniform_dist
    );
    new_segment.row(1) = generateNextAtomDihedral<double>(
        coords2.row(0), coords2.row(1), new_segment.row(0), length, theta2,
        phi1
    ); 
    new_segment.row(2) = generateNextAtomDihedral<double>(
        coords2.row(1), new_segment.row(0), new_segment.row(1), length,
        theta3, phi2
    );
    melt_configs.replaceSegment(1, new_segment, 2);

    // Check the new polymer length and coordinates
    REQUIRE(melt_configs.getLength(1) == 6);
    config_coords2 = melt_configs.getSegment(1, 0, 6); 
    REQUIRE(config_coords2.rows() == 6); 
    for (int i = 0; i < 2; ++i)    // Upstream of replaced segment
        REQUIRE_THAT(
            (config_coords2.row(i) - coords2.row(i)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    for (int i = 2; i < 5; ++i)    // Replaced segment
        REQUIRE_THAT(
            (config_coords2.row(i) - new_segment.row(i - 2)).norm(), 
            Catch::Matchers::WithinAbs(0, tol)
        ); 
    for (int i = 5; i < 6; ++i)    // Downstream of replaced segment 
        REQUIRE_THAT(
            (config_coords2.row(i) - coords2.row(i)).norm(),
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

    // Define a large box
    const double xmin = -100.0; 
    const double xmax = 100.0; 
    const double ymin = -100.0; 
    const double ymax = 100.0;
    const double zmin = -100.0;
    const double zmax = 100.0;

    // Generate an example melt configuration
    auto coords_all = generateExampleMeltCoords(rng, uniform_dist);
    REQUIRE(coords_all.size() == 3); 
    Matrix<double, Dynamic, 3> coords1 = coords_all[0]; 
    Matrix<double, Dynamic, 3> coords2 = coords_all[1]; 
    Matrix<double, Dynamic, 3> coords3 = coords_all[2];  
    PolymerMeltConfiguration<double> melt_configs(
        3, coords_all, Units::NANO, 300, xmin, xmax, ymin, ymax, zmin, zmax
    ); 
    REQUIRE(melt_configs.getLength(0) == 8); 
    REQUIRE(melt_configs.getLength(1) == 6); 
    REQUIRE(melt_configs.getLength(2) == 7);  

    // Append a 2-atom segment to the tail of the first polymer
    const double length = 1.5; 
    double theta1 = 173 * boost::math::constants::pi<double>() / 180; 
    double theta2 = 159 * boost::math::constants::pi<double>() / 180; 
    double phi1 = 58 * boost::math::constants::pi<double>() / 180; 
    double phi2 = -134 * boost::math::constants::pi<double>() / 180;
    Matrix<double, 3, 1> tail1 = generateNextAtomDihedral<double>(
        coords1.row(5), coords1.row(6), coords1.row(7), length, theta1, phi1 
    ); 
    Matrix<double, 3, 1> tail2 = generateNextAtomDihedral<double>(
        coords1.row(6), coords1.row(7), tail1, length, theta2, phi2
    ); 

    // Update the first polymer's configuration
    Matrix<double, Dynamic, 3> tail(2, 3); 
    tail << tail1(0), tail1(1), tail1(2), 
            tail2(0), tail2(1), tail2(2);  
    melt_configs.appendSegmentToTail(0, tail);

    // Check the polymer length and coordinates 
    REQUIRE(melt_configs.getLength(0) == 10);
    Matrix<double, Dynamic, 3> new_coords1 = melt_configs.getSegment(0, 0, 10);
    REQUIRE(new_coords1.rows() == 10); 
    for (int i = 0; i < 8; ++i)
        REQUIRE_THAT(
            (new_coords1.row(i) - coords1.row(i)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    REQUIRE_THAT(
        (new_coords1.row(8) - tail1.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    ); 
    REQUIRE_THAT(
        (new_coords1.row(9) - tail2.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );
    
    // Append a 4-atom segment to the head of the second polymer
    double theta3 = 132 * boost::math::constants::pi<double>() / 180; 
    double theta4 = 156 * boost::math::constants::pi<double>() / 180;
    double theta5 = 98 * boost::math::constants::pi<double>() / 180; 
    double theta6 = 111 * boost::math::constants::pi<double>() / 180; 
    double phi3 = 88 * boost::math::constants::pi<double>() / 180; 
    double phi4 = 145 * boost::math::constants::pi<double>() / 180;
    double phi5 = -20 * boost::math::constants::pi<double>() / 180; 
    double phi6 = 63 * boost::math::constants::pi<double>() / 180; 
    Matrix<double, 3, 1> head1 = generateNextAtomDihedral<double>(
        coords2.row(2), coords2.row(1), coords2.row(0), length, theta3, phi3
    ); 
    Matrix<double, 3, 1> head2 = generateNextAtomDihedral<double>(
        coords2.row(1), coords2.row(0), head1, length, theta4, phi4
    ); 
    Matrix<double, 3, 1> head3 = generateNextAtomDihedral<double>(
        coords2.row(0), head1, head2, length, theta5, phi5
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
    melt_configs.appendSegmentToHead(1, head); 

    // Check the polymer length and coordinates 
    REQUIRE(melt_configs.getLength(1) == 10);
    Matrix<double, Dynamic, 3> new_coords2 = melt_configs.getSegment(1, 0, 10);
    REQUIRE(new_coords2.rows() == 10);
    REQUIRE_THAT(
        (new_coords2.row(0) - head4.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );
    REQUIRE_THAT(
        (new_coords2.row(1) - head3.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    ); 
    REQUIRE_THAT(
        (new_coords2.row(2) - head2.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    ); 
    REQUIRE_THAT(
        (new_coords2.row(3) - head1.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    ); 
    for (int i = 4; i < 10; ++i)
    {
        REQUIRE_THAT(
            (new_coords2.row(i) - coords2.row(i - 4)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    }

    // Remove 3-atom segment from the tail of the first polymer 
    melt_configs.popSegmentFromTail(0, 10 - 3);    // Last index to remove

    // Check the polymer length and coordinates
    REQUIRE(melt_configs.getLength(0) == 7);
    Matrix<double, Dynamic, 3> old_coords1 = new_coords1; 
    new_coords1 = melt_configs.getSegment(0, 0, 7); 
    REQUIRE(new_coords1.rows() == 7);
    for (int i = 0; i < 7; ++i)
    {
        REQUIRE_THAT(
            (new_coords1.row(i) - old_coords1.row(i)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    }

    // Reptate towards the tail of the first polymer 
    old_coords1 = new_coords1; 
    double theta7 = 37 * boost::math::constants::pi<double>() / 180; 
    double phi7 = -99 * boost::math::constants::pi<double>() / 180;
    Matrix<double, 3, 1> tail3 = generateNextAtomDihedral<double>(
        old_coords1.row(4), old_coords1.row(5), old_coords1.row(6), length,
        theta7, phi7
    );
    melt_configs.reptateTowardsTail(0, tail3);

    // Check the polymer length and coordinates
    REQUIRE(melt_configs.getLength(0) == 7);
    new_coords1 = melt_configs.getSegment(0, 0, 7); 
    REQUIRE(new_coords1.rows() == 7);
    for (int i = 0; i < 6; ++i)
    {
        REQUIRE_THAT(
            (new_coords1.row(i) - old_coords1.row(i + 1)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    }
    REQUIRE_THAT(
        (new_coords1.row(6) - tail3.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );

    // Reptate towards the head of the third polymer
    double theta8 = 126 * boost::math::constants::pi<double>() / 180; 
    double phi8 = -2 * boost::math::constants::pi<double>() / 180;
    Matrix<double, 3, 1> head5 = generateNextAtomDihedral<double>(
        coords3.row(2), coords3.row(1), coords3.row(0), length,
        theta8, phi8
    );
    melt_configs.reptateTowardsHead(2, head5);

    // Check the polymer length and coordinates
    REQUIRE(melt_configs.getLength(2) == 7);
    Matrix<double, Dynamic, 3> new_coords3 = melt_configs.getSegment(2, 0, 7); 
    REQUIRE(new_coords3.rows() == 7);
    REQUIRE_THAT(
        (new_coords3.row(0) - head5.transpose()).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );
    for (int i = 1; i < 7; ++i)
    {
        REQUIRE_THAT(
            (new_coords3.row(i) - coords3.row(i - 1)).norm(),
            Catch::Matchers::WithinAbs(0, tol)
        );
    }

    // Generate 3 new atoms at the tail of the second polymer and reptate 
    Matrix<double, Dynamic, 3> old_coords2 = new_coords2; 
    double theta9 = 140 * boost::math::constants::pi<double>() / 180; 
    double phi9 = -45 * boost::math::constants::pi<double>() / 180; 
    double theta10 = 94 * boost::math::constants::pi<double>() / 180; 
    double phi10 = 82 * boost::math::constants::pi<double>() / 180; 
    double theta11 = 56 * boost::math::constants::pi<double>() / 180; 
    double phi11 = -132 * boost::math::constants::pi<double>() / 180; 
    Matrix<double, Dynamic, 3> tail_segment(3, 3); 
    tail_segment.row(0) = generateNextAtomDihedral<double>(
        old_coords2.row(7), old_coords2.row(8), old_coords2.row(9), length,
        theta9, phi9
    ); 
    tail_segment.row(1) = generateNextAtomDihedral<double>(
        old_coords2.row(8), old_coords2.row(9), tail_segment.row(0), length,
        theta10, phi10
    ); 
    tail_segment.row(2) = generateNextAtomDihedral<double>(
        old_coords2.row(9), tail_segment.row(0), tail_segment.row(1), length,
        theta11, phi11
    );
    melt_configs.reptateTowardsTailMultimer(1, tail_segment);

    // Check the polymer length and coordinates
    REQUIRE(melt_configs.getLength(1) == 10);
    new_coords2 = melt_configs.getSegment(1, 0, 10); 
    REQUIRE(new_coords2.rows() == 10);
    for (int i = 0; i < 7; ++i)
    {
        REQUIRE_THAT(
            (new_coords2.row(i) - old_coords2.row(i + 3)).norm(), 
            Catch::Matchers::WithinAbs(0, tol)
        );
    }
    REQUIRE_THAT(
        (new_coords2.row(7) - tail_segment.row(0)).norm(), 
        Catch::Matchers::WithinAbs(0, tol)
    );
    REQUIRE_THAT(
        (new_coords2.row(8) - tail_segment.row(1)).norm(), 
        Catch::Matchers::WithinAbs(0, tol)
    ); 
    REQUIRE_THAT(
        (new_coords2.row(9) - tail_segment.row(2)).norm(),
        Catch::Matchers::WithinAbs(0, tol)
    );

    // Generate 4 new atoms at the head of the third polymer and reptate
    Matrix<double, Dynamic, 3> old_coords3 = new_coords3; 
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
        old_coords3.row(2), old_coords3.row(1), old_coords3.row(0), length,
        theta12, phi12
    ); 
    head_segment.row(2) = generateNextAtomDihedral<double>(
        old_coords3.row(1), old_coords3.row(0), head_segment.row(3), length,
        theta13, phi13
    ); 
    head_segment.row(1) = generateNextAtomDihedral<double>(
        old_coords3.row(0), head_segment.row(3), head_segment.row(2), length,
        theta14, phi14
    );
    head_segment.row(0) = generateNextAtomDihedral<double>(
        head_segment.row(3), head_segment.row(2), head_segment.row(1), length,
        theta15, phi15
    );
    melt_configs.reptateTowardsHeadMultimer(2, head_segment); 

    // Check the polymer length and coordinates
    REQUIRE(melt_configs.getLength(2) == 7);
    new_coords3 = melt_configs.getSegment(2, 0, 7); 
    REQUIRE(new_coords3.rows() == 7);
    for (int i = 0; i < 4; ++i)
    {
        REQUIRE_THAT(
            (new_coords3.row(i) - head_segment.row(i)).norm(), 
            Catch::Matchers::WithinAbs(0, tol)
        );
    }
    for (int i = 4; i < 7; ++i)
    {
        REQUIRE_THAT(
            (new_coords3.row(i) - old_coords3.row(i - 4)).norm(), 
            Catch::Matchers::WithinAbs(0, tol)
        ); 
    }
}

/**
 * Tests for parseMeltLammps() and writeLammps(). 
 */
TEST_CASE(
    "Tests for parsing and writing functions",
    "[writeLammps(), parseMeltLammps()]"
)
{
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;
    const double tol = 1e-8; 

    // Define a large box
    const double xmin = -100.0; 
    const double xmax = 100.0; 
    const double ymin = -100.0; 
    const double ymax = 100.0;
    const double zmin = -100.0;
    const double zmax = 100.0;

    // Set up potential and sampling parameters
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

    // Generate an example melt configuration 
    auto coords_all = generateExampleMeltCoords(rng, uniform_dist);
    REQUIRE(coords_all.size() == 3); 
    Matrix<double, Dynamic, 3> coords1 = coords_all[0];
    Matrix<double, Dynamic, 3> coords2 = coords_all[1]; 
    Matrix<double, Dynamic, 3> coords3 = coords_all[2];  
    PolymerMeltConfiguration<double> melt_configs(
        3, coords_all, Units::NANO, 300, xmin, xmax, ymin, ymax, zmin, zmax
    );

    // Write the coordinates to file 
    melt_configs.writeLammps(
        "configs/test_melt_cosine.txt", lj_params, fene_params, AngleMode::COSINE, 
        cosine_params, dihedral_params, "Test configuration", 1
    );

    // Parse the coordinates 
    auto result = parseMeltLammps<double>(
        "configs/test_melt_cosine.txt", Units::NANO, 300
    ); 
    PolymerMeltConfiguration<double> parsed_configs = std::get<0>(result); 
    std::unordered_map<std::string, double> lj_params2 = std::get<1>(result); 
    std::unordered_map<std::string, double> fene_params2 = std::get<2>(result); 
    AngleMode angle_mode2 = std::get<3>(result); 
    std::unordered_map<std::string, double> angle_params2 = std::get<4>(result); 
    std::unordered_map<std::string, double> dihedral_params2 = std::get<5>(result); 

    // Check the atomic coordinates
    for (int i = 0; i < 3; ++i)
    { 
        REQUIRE(parsed_configs.getLength(i) == melt_configs.getLength(i));
        const int length = melt_configs.getLength(i);  
        Matrix<double, Dynamic, 3> coords = melt_configs.getSegment(i, 0, length); 
        Matrix<double, Dynamic, 3> coords2 = parsed_configs.getSegment(i, 0, length); 
        for (int i = 0; i < length; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                REQUIRE_THAT(coords2(i, j), Catch::Matchers::WithinAbs(coords(i, j), tol)); 
            } 
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

    // Rewrite the melt coordinates to file with the Gaussian angle potential 
    // parameters
    melt_configs.writeLammps(
        "configs/test_melt_gaussian.txt", lj_params, fene_params,
        AngleMode::GAUSSIAN, gaussian_params, dihedral_params,
        "Test configuration", 1
    ); 

    // Parse the coordinates again 
    result = parseMeltLammps<double>(
        "configs/test_melt_gaussian.txt", Units::NANO, 300
    ); 
    parsed_configs = std::get<0>(result); 
    lj_params2 = std::get<1>(result); 
    fene_params2 = std::get<2>(result); 
    angle_mode2 = std::get<3>(result); 
    angle_params2 = std::get<4>(result); 
    dihedral_params2 = std::get<5>(result); 

    // Check the atomic coordinates
    for (int i = 0; i < 3; ++i)
    { 
        REQUIRE(parsed_configs.getLength(i) == melt_configs.getLength(i));
        const int length = melt_configs.getLength(i);  
        Matrix<double, Dynamic, 3> coords = melt_configs.getSegment(i, 0, length); 
        Matrix<double, Dynamic, 3> coords2 = parsed_configs.getSegment(i, 0, length); 
        for (int i = 0; i < length; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                REQUIRE_THAT(coords2(i, j), Catch::Matchers::WithinAbs(coords(i, j), tol)); 
            } 
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
 * Tests for generateKMerMelt(). 
 */
TEST_CASE("Tests for melt generation", "[generateKMerMelt()]")
{
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;
    const double tol = 1e-3;

    // Set up potential and sampling parameters
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

    // Define collision thresholds such that some non-bonded interactions are
    // allowed 
    const double intra_collision_threshold = pow(2., 1. / 6.) * lj_params["sigma"];
    const double inter_collision_threshold = 0.5 * pow(2., 1. / 6.) * lj_params["sigma"]; 
    const int max_tries_per_atom = 50;
    const int max_tries_per_kmer = 20; 
    const int max_tries_per_seed = 20; 
    const int max_n_backtracks = 50;
    const int max_n_restarts = 5;

    // Calculate bond length CDF 
    Matrix<double, Dynamic, 2> bond_length_cdf = getFeneCDF<double>(
        lj_params["eps"], lj_params["sigma"], fene_params["K"], fene_params["R0"],
        kT, 10000
    );

    // Define a large box 
    double xmin = -100.0; 
    double xmax = 100.0; 
    double ymin = -100.0; 
    double ymax = 100.0;
    double zmin = -100.0;
    double zmax = 100.0;

    // Generate a melt of five 10-mers within the small box 
    PolymerMeltConfiguration<double> melt_configs = generateKMerMelt<double>(
        10, 5, lj_params, fene_params, AngleMode::COSINE, cosine_params,
        dihedral_params, intra_collision_threshold, inter_collision_threshold,
        max_tries_per_atom, max_tries_per_kmer, max_tries_per_seed,
        max_n_backtracks, max_n_restarts, rng, uniform_dist, xmax, ymax, 
        zmax, bond_length_cdf, Units::NANO, 300.0, true
    );
    REQUIRE(melt_configs.numChains() == 5);
    for (int i = 0; i < 5; ++i) 
        REQUIRE(melt_configs.getLength(i) == 10);

    // Check that each pair of non-bonded atoms are within the collision
    // thresholds
    for (int i = 0; i < 5; ++i)
    {
        Matrix<double, Dynamic, 3> coords = melt_configs.getSegment(i, 0, 10);

        // Compare non-bonded atoms along chain i ...  
        for (int j = 0; j < 10; ++j)
        {
            for (int k = 0; k < 10; ++k)
            {
                if (abs(j - k) > 1)
                {
                    REQUIRE((coords.row(j) - coords.row(k)).norm() > intra_collision_threshold); 
                }
            }
        }

        // Compare atoms in chain i with atoms in other chains ... 
        for (int j = i + 1; j < 5; ++j)
        {
            Matrix<double, Dynamic, 3> coords2 = melt_configs.getSegment(j, 0, 10);
            for (int k = 0; k < 10; ++k)
            {
                for (int m = 0; m < 10; ++m)
                {
                    REQUIRE((coords.row(k) - coords2.row(m)).norm() > inter_collision_threshold); 
                }
            } 
        }
    }

    // Check that each pair of bonded atoms are within the maximum bond length
    for (int i = 0; i < 5; ++i)
    {
        Matrix<double, Dynamic, 3> coords = melt_configs.getSegment(i, 0, 10); 
        for (int j = 0; j < 9; ++j)
        {
            double bond_length = (coords.row(j + 1) - coords.row(j)).norm();
            REQUIRE(bond_length < fene_params["R0"]); 
        }
    }

    // Compute the average bond angle and dihedral angle, for comparison with
    // equilibrium values 
    double mean_angle = 0; 
    int curr_idx = 0;  
    for (int i = 0; i < 5; ++i)
    {
        Matrix<double, Dynamic, 3> coords = melt_configs.getSegment(i, 0, 10); 
        for (int j = 0; j < 8; ++j)
        {
            Matrix<double, 3, 1> u = coords.row(j) - coords.row(j + 1); 
            Matrix<double, 3, 1> v = coords.row(j + 2) - coords.row(j + 1);
            double u_norm = u.norm(); 
            double v_norm = v.norm();  
            u /= u_norm; 
            v /= v_norm;
            double theta = acosSafe<double>(u.dot(v));
            mean_angle = (theta + curr_idx * mean_angle) / (curr_idx + 1);
            curr_idx++; 
        }
    } 
    std::cout << "- Mean bond angle = " << mean_angle << " (equilibrium value: "
              << cosine_params["theta0"] << ")" << std::endl; 
    double mean_dihedral = 0;
    curr_idx = 0; 
    for (int i = 0; i < 5; ++i)
    {
        Matrix<double, Dynamic, 3> coords = melt_configs.getSegment(i, 0, 10); 
        for (int j = 0; j < 7; ++j)
        {
            double phi = getDihedral<double>(
                coords.row(j), coords.row(j + 1), coords.row(j + 2),
                coords.row(j + 3)
            );
            if (phi < 0)
                phi += boost::math::constants::two_pi<double>();  
            mean_dihedral = (phi + curr_idx * mean_dihedral) / (curr_idx + 1); 
            curr_idx++;  
        }
    }
    std::cout << "- Mean dihedral = " << mean_dihedral << " (equilibrium value: "
              << boost::math::constants::pi<double>() << ")" << std::endl; 

    // Define a small box so that polymer chains tend to be closer together 
    xmin = -5.0; 
    xmax = 5.0; 
    ymin = -5.0; 
    ymax = 5.0;
    zmin = -5.0;
    zmax = 5.0;

    // Generate a melt of five 10-mers within the small box 
    melt_configs = generateKMerMelt<double>(
        10, 5, lj_params, fene_params, AngleMode::COSINE, cosine_params,
        dihedral_params, intra_collision_threshold, inter_collision_threshold,
        max_tries_per_atom, max_tries_per_kmer, max_tries_per_seed,
        max_n_backtracks, max_n_restarts, rng, uniform_dist, xmax, ymax, 
        zmax, bond_length_cdf, Units::NANO, 300.0, true
    );
    REQUIRE(melt_configs.numChains() == 5);
    for (int i = 0; i < 5; ++i) 
        REQUIRE(melt_configs.getLength(i) == 10);

    // Check that each pair of non-bonded atoms are within the collision
    // thresholds
    for (int i = 0; i < 5; ++i)
    {
        Matrix<double, Dynamic, 3> coords = melt_configs.getSegment(i, 0, 10);

        // Compare non-bonded atoms along chain i ...  
        for (int j = 0; j < 10; ++j)
        {
            for (int k = 0; k < 10; ++k)
            {
                if (abs(j - k) > 1)
                {
                    REQUIRE((coords.row(j) - coords.row(k)).norm() > intra_collision_threshold); 
                }
            }
        }

        // Compare atoms in chain i with atoms in other chains ... 
        for (int j = i + 1; j < 5; ++j)
        {
            Matrix<double, Dynamic, 3> coords2 = melt_configs.getSegment(j, 0, 10);
            for (int k = 0; k < 10; ++k)
            {
                for (int m = 0; m < 10; ++m)
                {
                    REQUIRE((coords.row(k) - coords2.row(m)).norm() > inter_collision_threshold); 
                }
            } 
        }
    }

    // Check that each pair of bonded atoms are within the maximum bond length
    for (int i = 0; i < 5; ++i)
    {
        Matrix<double, Dynamic, 3> coords = melt_configs.getSegment(i, 0, 10); 
        for (int j = 0; j < 9; ++j)
        {
            double bond_length = (coords.row(j + 1) - coords.row(j)).norm();
            REQUIRE(bond_length < fene_params["R0"]); 
        }
    }
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
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;
    const double tol = 1e-3;

    // Define a large box
    const double xmin = -100.0; 
    const double xmax = 100.0; 
    const double ymin = -100.0; 
    const double ymax = 100.0;
    const double zmin = -100.0;
    const double zmax = 100.0;

    // Set up potential and sampling parameters
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

    // Calculate bond length CDF 
    Matrix<double, Dynamic, 2> bond_length_cdf = getFeneCDF<double>(
        lj_params["eps"], lj_params["sigma"], fene_params["K"], fene_params["R0"],
        kT, 10000
    ); 

    // Generate a melt of five 10-mers
    const double intra_collision_threshold = pow(2., 1. / 6.) * lj_params["sigma"];
    const double inter_collision_threshold = pow(2., 1. / 6.) * lj_params["sigma"]; 
    const int max_tries_per_atom = 50;
    const int max_tries_per_kmer = 20; 
    const int max_tries_per_seed = 20; 
    const int max_n_backtracks = 50;
    const int max_n_restarts = 5;
    PolymerMeltConfiguration<double> melt_configs = generateKMerMelt<double>(
        10, 5, lj_params, fene_params, AngleMode::COSINE, cosine_params,
        dihedral_params, intra_collision_threshold, inter_collision_threshold,
        max_tries_per_atom, max_tries_per_kmer, max_tries_per_seed,
        max_n_backtracks, max_n_restarts, rng, uniform_dist, xmax, ymax, 
        zmax, bond_length_cdf, Units::NANO, 300.0, true
    );

    // Write the 10-mer coordinates to file
    melt_configs.writeLammps(
        "configs/test_10mer_melt_cosine.txt", lj_params, fene_params,
        AngleMode::COSINE, cosine_params, dihedral_params, "Test configuration",
        1
    );

    // Calculate energies via LAMMPS
    std::string cmd = "python3 run_lammps_get_energy.py configs/test_10mer_melt_cosine.txt"; 
    int rc = std::system(cmd.c_str()); 
    if (rc != 0)
        throw std::runtime_error("Failed to run run_lammps_get_energy.py");
    std::ifstream infile("configs/test_10mer_melt_cosine_energy.txt");
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
    double nonbonded_energy_nc = melt_configs.getTotalNonbondedEnergy(
        lj_params, neighbor_threshold, true
    );
    REQUIRE_THAT(
        nonbonded_energy_nc,
        Catch::Matchers::WithinAbs(nonbonded_energy_lammps, tol)
    ); 

    // Calculate the bonded interaction energy (including Lennard-Jones) and 
    // compare against LAMMPS-computed value
    double bond_energy = melt_configs.getTotalBondEnergy(fene_params, true, lj_params);
    REQUIRE_THAT(bond_energy, Catch::Matchers::WithinAbs(bond_energy_lammps, tol)); 

    // Calculate the bond angle energy and compare against LAMMPS-computed value
    double angle_energy = melt_configs.getTotalBondAngleEnergy(
        AngleMode::COSINE, cosine_params
    ); 
    REQUIRE_THAT(angle_energy, Catch::Matchers::WithinAbs(angle_energy_lammps, tol));

    // Calculate the dihedral angle energy and compare against LAMMPS-computed
    // value
    double dihedral_energy = melt_configs.getTotalDihedralAngleEnergy(
        dihedral_params
    ); 
    REQUIRE_THAT(
        dihedral_energy,
        Catch::Matchers::WithinAbs(
            energy_lammps - nonbonded_energy_lammps - bond_energy_lammps - angle_energy_lammps,
            tol
        ) 
    );

    // Generate a melt of five 10-mers with the Gaussian potential
    melt_configs = generateKMerMelt<double>(
        10, 5, lj_params, fene_params, AngleMode::GAUSSIAN, gaussian_params,
        dihedral_params, intra_collision_threshold, inter_collision_threshold,
        max_tries_per_atom, max_tries_per_kmer, max_tries_per_seed,
        max_n_backtracks, max_n_restarts, rng, uniform_dist, xmax, ymax, 
        zmax, bond_length_cdf, Units::NANO, 300.0, true
    );

    // Write the 10-mer coordinates to file
    melt_configs.writeLammps(
        "configs/test_10mer_melt_gaussian.txt", lj_params, fene_params,
        AngleMode::GAUSSIAN, gaussian_params, dihedral_params,
        "Test configuration", 1
    );

    // Calculate energies via LAMMPS
    cmd = "python3 run_lammps_get_energy.py configs/test_10mer_melt_gaussian.txt"; 
    rc = std::system(cmd.c_str()); 
    if (rc != 0)
        throw std::runtime_error("Failed to run run_lammps_get_energy.py");
    infile.open("configs/test_10mer_melt_gaussian_energy.txt");
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
    nonbonded_energy_nc = melt_configs.getTotalNonbondedEnergy(
        lj_params, neighbor_threshold, true
    );
    REQUIRE_THAT(
        nonbonded_energy_nc,
        Catch::Matchers::WithinAbs(nonbonded_energy_lammps, tol)
    ); 

    // Calculate the bonded interaction energy (including Lennard-Jones) and 
    // compare against LAMMPS-computed value
    bond_energy = melt_configs.getTotalBondEnergy(fene_params, true, lj_params);
    REQUIRE_THAT(bond_energy, Catch::Matchers::WithinAbs(bond_energy_lammps, tol));

    // Calculate the bond angle energy and compare against LAMMPS-computed value
    angle_energy = melt_configs.getTotalBondAngleEnergy(
        AngleMode::GAUSSIAN, gaussian_params
    ); 
    REQUIRE_THAT(angle_energy, Catch::Matchers::WithinAbs(angle_energy_lammps, tol));

    // Calculate the dihedral angle energy and compare against LAMMPS-computed
    // value
    dihedral_energy = melt_configs.getTotalDihedralAngleEnergy(dihedral_params); 
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
    "Tests for reptation residual energy calculation",
    "[getReptationResidualEnergy()]"
)
{
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;
    const double tol = 1e-8;

    // Parse the previously generated 10-mer melt 
    auto result = parseMeltLammps<double>(
        "configs/test_10mer_melt_cosine.txt", Units::NANO, 300.0
    ); 
    PolymerMeltConfiguration<double> melt_configs = std::get<0>(result);
    std::unordered_map<std::string, double> lj_params = std::get<1>(result); 
    std::unordered_map<std::string, double> fene_params = std::get<2>(result); 
    AngleMode angle_mode = std::get<3>(result); 
    std::unordered_map<std::string, double> angle_params = std::get<4>(result); 
    std::unordered_map<std::string, double> dihedral_params = std::get<5>(result);

    // Reptate the second polymer towards the tail
    //
    // Choose a new atom that is close to atom 8 along the original configuration, 
    // so that we have some non-bonded interactions 
    int l0 = melt_configs.getLength(0); 
    int l1 = melt_configs.getLength(1); 
    int l2 = melt_configs.getLength(2);
    REQUIRE(l0 == 10); 
    REQUIRE(l1 == 10); 
    REQUIRE(l2 == 10);
    const double scale = 0.9 * lj_params["sigma"];
    Matrix<double, Dynamic, 3> coords_chain0_1 = melt_configs.getSegment(0, 0, 10); 
    Matrix<double, Dynamic, 3> coords_chain1_1 = melt_configs.getSegment(1, 0, 10);
    Matrix<double, Dynamic, 3> coords_chain2_1 = melt_configs.getSegment(2, 0, 10); 
    Matrix<double, 3, 1> r; 
    r << -scale + 2 * scale * uniform_dist(rng), 
         -scale + 2 * scale * uniform_dist(rng), 
         -scale + 2 * scale * uniform_dist(rng); 
    Matrix<double, 3, 1> r_tail = coords_chain1_1.row(l1 - 2) + r.transpose();
    PolymerMeltConfiguration<double> melt_configs2(melt_configs); 
    melt_configs2.reptateTowardsTail(1, r_tail); 
   
    // Check that the coordinates are updated correctly  
    Matrix<double, Dynamic, 3> coords_chain1_2 = melt_configs2.getSegment(1, 0, 10);
    for (int i = 0; i < l1 - 1; ++i) 
        REQUIRE((coords_chain1_2.row(i) - coords_chain1_1.row(i + 1)).norm() < tol);
    REQUIRE((coords_chain1_2.row(l1 - 1) - r_tail.transpose()).norm() < tol); 

    // Calculate the reptation residual energy
    //
    // This should be the non-bonded energy between: 
    // - each atom with index > 0 (0-indexed) in the original configuration
    //   and the new atom, minus the adjacent atom, and 
    // - each atom in every other chain and the new atom  
    double neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"]; 
    double reptate_residual_energy_12 = melt_configs.getReptationResidualEnergy(
        1, r_tail, lj_params, neighbor_threshold
    );
    double sum = 0;
    for (int i = 1; i < 9; ++i)    // Omit atoms 0 and 9
    {
        double dist = (r_tail.transpose() - coords_chain1_1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    for (int i = 0; i < 10; ++i)    // Run through atoms on other chains
    {
        double dist = (r_tail.transpose() - coords_chain0_1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
        dist = (r_tail.transpose() - coords_chain2_1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(reptate_residual_energy_12, Catch::Matchers::WithinAbs(sum, tol));

    // Generate the reverse reptated configuration 
    PolymerMeltConfiguration<double> melt_configs3(melt_configs2); 
    melt_configs3.reptateTowardsHead(1, coords_chain1_1.row(0));

    // Check that the coordinates are updated correctly  
    Matrix<double, Dynamic, 3> coords_chain1_3 = melt_configs3.getSegment(1, 0, 10);
    for (int i = 0; i < l1; ++i) 
        REQUIRE((coords_chain1_3.row(i) - coords_chain1_1.row(i)).norm() < tol);

    // Calculate the reverse reptation residual energy 
    //
    // This should be the non-bonded energy between: 
    // - each atom with index < 9 (0-indexed) in the reptated configuration
    //   and the new atom, minus the adjacent atom, and
    // - each atom in every other chain and the new atom  
    Matrix<double, Dynamic, 3> coords_chain0_2 = melt_configs2.getSegment(0, 0, 10); 
    Matrix<double, Dynamic, 3> coords_chain2_2 = melt_configs2.getSegment(2, 0, 10); 
    double reptate_residual_energy_23 = melt_configs2.getReptationResidualEnergy(
        1, coords_chain1_1.row(0), lj_params, neighbor_threshold 
    );
    sum = 0;
    for (int i = 1; i < 9; ++i)    // Omit atoms 0 and 9 
    {
        double dist = (coords_chain1_1.row(0) - coords_chain1_2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    for (int i = 0; i < 10; ++i)    // Run through atoms on other chains
    {
        double dist = (coords_chain1_1.row(0) - coords_chain0_2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
        dist = (coords_chain1_1.row(0) - coords_chain2_2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(reptate_residual_energy_23, Catch::Matchers::WithinAbs(sum, tol));
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
    const double tol = 1e-8;

    // Parse the previously generated 10-mer melt 
    auto result = parseMeltLammps<double>(
        "configs/test_10mer_melt_cosine.txt", Units::NANO, 300.0
    ); 
    PolymerMeltConfiguration<double> melt_configs = std::get<0>(result);
    std::unordered_map<std::string, double> lj_params = std::get<1>(result); 
    std::unordered_map<std::string, double> fene_params = std::get<2>(result); 
    AngleMode angle_mode = std::get<3>(result); 
    std::unordered_map<std::string, double> angle_params = std::get<4>(result); 
    std::unordered_map<std::string, double> dihedral_params = std::get<5>(result);

    // Reptate the second polymer towards the tail
    //
    // Choose three new atoms that are close to atoms to atoms 8, 7, 6 along
    // the original configuration, so that we have some non-bonded interactions 
    int l0 = melt_configs.getLength(0); 
    int l1 = melt_configs.getLength(1); 
    int l2 = melt_configs.getLength(2);
    REQUIRE(l0 == 10); 
    REQUIRE(l1 == 10); 
    REQUIRE(l2 == 10);
    const double scale = 0.9 * lj_params["sigma"];
    Matrix<double, Dynamic, 3> coords_chain0_1 = melt_configs.getSegment(0, 0, 10); 
    Matrix<double, Dynamic, 3> coords_chain1_1 = melt_configs.getSegment(1, 0, 10);
    Matrix<double, Dynamic, 3> coords_chain2_1 = melt_configs.getSegment(2, 0, 10); 
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
    segment.row(0) = coords_chain1_1.row(l1 - 2) + r1.transpose();
    segment.row(1) = coords_chain1_1.row(l1 - 3) + r2.transpose(); 
    segment.row(2) = coords_chain1_1.row(l1 - 4) + r3.transpose();
    PolymerMeltConfiguration<double> melt_configs2(melt_configs); 
    melt_configs2.reptateTowardsTailMultimer(1, segment);
   
    // Check that the coordinates are updated correctly  
    Matrix<double, Dynamic, 3> coords_chain1_2 = melt_configs2.getSegment(1, 0, 10);
    for (int i = 0; i < l1 - 3; ++i) 
        REQUIRE((coords_chain1_2.row(i) - coords_chain1_1.row(i + 3)).norm() < tol);
    REQUIRE((coords_chain1_2.row(l1 - 3) - segment.row(0)).norm() < tol); 
    REQUIRE((coords_chain1_2.row(l1 - 2) - segment.row(1)).norm() < tol); 
    REQUIRE((coords_chain1_2.row(l1 - 1) - segment.row(2)).norm() < tol);

    // Calculate the reptation residual energy for each atom in the segment 
    //
    // This should be the non-bonded energy between:
    // - each atom with index > 2 (0-indexed) in the original configuration
    //   and the new atom, minus the adjacent atom, and 
    // - each atom in every other chain and the new atom 
    //
    // Calculate residual energy for i = 0
    double neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"]; 
    double reptate_residual_energy_12_0 = melt_configs.getMultimerReptationResidualEnergy(
        1, ReptationDirection::TAIL, 3, 0, 
        segment(Eigen::seqN(0, 0), Eigen::all), segment.row(0), lj_params, 
        neighbor_threshold
    );
    double sum = 0;
    for (int i = 3; i < 9; ++i)    // Omit the first three atoms and atom 9
    {
        double dist = (segment.row(0) - coords_chain1_1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    for (int i = 0; i < 10; ++i)    // Run through atoms on other chains
    {
        double dist = (segment.row(0) - coords_chain0_1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
        dist = (segment.row(0) - coords_chain2_1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(reptate_residual_energy_12_0, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 1
    double reptate_residual_energy_12_1 = melt_configs.getMultimerReptationResidualEnergy(
        1, ReptationDirection::TAIL, 3, 1,
        segment(Eigen::seqN(0, 1), Eigen::all), segment.row(1), lj_params, 
        neighbor_threshold
    ); 
    sum = 0; 
    for (int i = 3; i < 10; ++i)    // Omit the first three atoms
    {
        double dist = (segment.row(1) - coords_chain1_1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    for (int i = 0; i < 10; ++i)    // Run through atoms on other chains
    {
        double dist = (segment.row(1) - coords_chain0_1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
        dist = (segment.row(1) - coords_chain2_1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(reptate_residual_energy_12_1, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 2
    double reptate_residual_energy_12_2 = melt_configs.getMultimerReptationResidualEnergy(
        1, ReptationDirection::TAIL, 3, 2, 
        segment(Eigen::seqN(0, 2), Eigen::all), segment.row(2), lj_params, 
        neighbor_threshold
    );
    sum = 0;
    for (int i = 3; i < 10; ++i)    // Omit the first three atoms
    {
        double dist = (segment.row(2) - coords_chain1_1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    sum += lj<double>(
        (segment.row(2) - segment.row(0)).norm(),    // Add in atom 0 in the segment
        lj_params["eps"], lj_params["sigma"], true
    ); 
    for (int i = 0; i < 10; ++i)    // Run through atoms on other chains
    {
        double dist = (segment.row(2) - coords_chain0_1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
        dist = (segment.row(2) - coords_chain2_1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(reptate_residual_energy_12_2, Catch::Matchers::WithinAbs(sum, tol));

    // Generate the reverse reptated configuration 
    PolymerMeltConfiguration<double> melt_configs3(melt_configs2); 
    melt_configs3.reptateTowardsHeadMultimer(
        1, coords_chain1_1(Eigen::seqN(0, 3), Eigen::all)
    );

    // Check that the coordinates are updated correctly  
    Matrix<double, Dynamic, 3> coords_chain1_3 = melt_configs3.getSegment(1, 0, 10);
    for (int i = 0; i < l1; ++i) 
        REQUIRE((coords_chain1_3.row(i) - coords_chain1_1.row(i)).norm() < tol);

    // Calculate the reverse reptation residual energy for each atom in the
    // original configuration 
    //
    // This should be the non-bonded energy between:
    // - each atom with index < 2 (0-indexed) in the reptated configuration
    //   and the new atom, minus the adjacent atom, and
    // - each atom in every other chain and the new atom 
    //
    // Calculate residual energy for i = 0, which is atom 2 in the original 
    // configuration
    Matrix<double, Dynamic, 3> coords_chain0_2 = melt_configs2.getSegment(0, 0, 10); 
    Matrix<double, Dynamic, 3> coords_chain2_2 = melt_configs2.getSegment(2, 0, 10); 
    Matrix<double, Dynamic, 3> subsegment(0, 3); 
    double reptate_residual_energy_23_0 = melt_configs2.getMultimerReptationResidualEnergy(
        1, ReptationDirection::TAIL, 3, 0, subsegment, coords_chain1_1.row(2), 
        lj_params, neighbor_threshold 
    );
    sum = 0;
    for (int i = 1; i < 7; ++i)    // Omit the last three atoms and atom 0
    {
        double dist = (coords_chain1_1.row(2) - coords_chain1_2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    for (int i = 0; i < 10; ++i)    // Run through atoms on other chains
    {
        double dist = (coords_chain1_1.row(2) - coords_chain0_2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
        dist = (coords_chain1_1.row(2) - coords_chain2_2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(reptate_residual_energy_23_0, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 1
    subsegment.resize(1, 3); 
    subsegment.row(0) = coords_chain1_1.row(2); 
    double reptate_residual_energy_23_1 = melt_configs2.getMultimerReptationResidualEnergy(
        1, ReptationDirection::TAIL, 3, 1, subsegment, coords_chain1_1.row(1), 
        lj_params, neighbor_threshold
    ); 
    sum = 0; 
    for (int i = 0; i < 7; ++i)    // Omit the last three atoms
    {
        double dist = (coords_chain1_1.row(1) - coords_chain1_2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    for (int i = 0; i < 10; ++i)    // Run through atoms on other chains
    {
        double dist = (coords_chain1_1.row(1) - coords_chain0_2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
        dist = (coords_chain1_1.row(1) - coords_chain2_2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(reptate_residual_energy_23_1, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 2
    subsegment.conservativeResize(2, 3); 
    subsegment.row(1) = coords_chain1_1.row(1); 
    double reptate_residual_energy_23_2 = melt_configs2.getMultimerReptationResidualEnergy(
        1, ReptationDirection::TAIL, 3, 2, subsegment, coords_chain1_1.row(0),
        lj_params, neighbor_threshold
    );
    sum = 0;
    for (int i = 0; i < 7; ++i)    // Omit the last three atoms
    {
        double dist = (coords_chain1_1.row(0) - coords_chain1_2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    sum += lj<double>(
        (coords_chain1_1.row(0) - coords_chain1_1.row(2)).norm(),   // Add in atom 2 in the segment
        lj_params["eps"], lj_params["sigma"], true
    ); 
    for (int i = 0; i < 10; ++i)    // Run through atoms on other chains
    {
        double dist = (coords_chain1_1.row(0) - coords_chain0_2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
        dist = (coords_chain1_1.row(0) - coords_chain2_2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(reptate_residual_energy_23_2, Catch::Matchers::WithinAbs(sum, tol));
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
    const double tol = 1e-8;

    // Parse the previously generated 10-mer melt 
    auto result = parseMeltLammps<double>(
        "configs/test_10mer_melt_cosine.txt", Units::NANO, 300.0
    ); 
    PolymerMeltConfiguration<double> melt_configs = std::get<0>(result);
    std::unordered_map<std::string, double> lj_params = std::get<1>(result); 
    std::unordered_map<std::string, double> fene_params = std::get<2>(result); 
    AngleMode angle_mode = std::get<3>(result); 
    std::unordered_map<std::string, double> angle_params = std::get<4>(result); 
    std::unordered_map<std::string, double> dihedral_params = std::get<5>(result);

    // Move the last four atoms in the third polymer
    //
    // Choose four new atoms that are close to atoms to atoms 5, 4, 3, 2 along
    // the original configuration, so that we have some non-bonded interactions
    int l0 = melt_configs.getLength(0); 
    int l1 = melt_configs.getLength(1); 
    int l2 = melt_configs.getLength(2);
    REQUIRE(l0 == 10); 
    REQUIRE(l1 == 10); 
    REQUIRE(l2 == 10);
    const double scale = 0.9 * lj_params["sigma"];
    Matrix<double, Dynamic, 3> coords_chain0_1 = melt_configs.getSegment(0, 0, 10); 
    Matrix<double, Dynamic, 3> coords_chain1_1 = melt_configs.getSegment(1, 0, 10);
    Matrix<double, Dynamic, 3> coords_chain2_1 = melt_configs.getSegment(2, 0, 10); 
    Matrix<double, Dynamic, 3> segment(4, 3);
    Matrix<double, 3, 1> r1, r2, r3, r4;
    r1 << -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng); 
    r2 << -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng); 
    r3 << -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng);
    r4 << -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng), 
          -scale + 2 * scale * uniform_dist(rng); 
    segment.row(0) = coords_chain2_1.row(l2 - 5) + r1.transpose();
    segment.row(1) = coords_chain2_1.row(l2 - 6) + r2.transpose(); 
    segment.row(2) = coords_chain2_1.row(l2 - 7) + r3.transpose();
    segment.row(3) = coords_chain2_1.row(l2 - 8) + r4.transpose();
    PolymerMeltConfiguration<double> melt_configs2(melt_configs); 
    melt_configs2.replaceSegment(2, segment, 6); 
   
    // Check that the coordinates are updated correctly  
    Matrix<double, Dynamic, 3> coords_chain2_2 = melt_configs2.getSegment(2, 0, 10);
    for (int i = 0; i < l2 - 4; ++i) 
        REQUIRE((coords_chain2_2.row(i) - coords_chain2_1.row(i)).norm() < tol);
    REQUIRE((coords_chain2_2.row(l2 - 4) - segment.row(0)).norm() < tol); 
    REQUIRE((coords_chain2_2.row(l2 - 3) - segment.row(1)).norm() < tol); 
    REQUIRE((coords_chain2_2.row(l2 - 2) - segment.row(2)).norm() < tol);
    REQUIRE((coords_chain2_2.row(l2 - 1) - segment.row(3)).norm() < tol);

    // Calculate the terminal segment replacement residual energy for each
    // atom in the segment 
    //
    // This should be the non-bonded energy between:
    // - each atom with index < 6 (0-indexed) in the original configuration
    //   and the new atom, minus the adjacent atom, and
    // - each atom in every other chain and the new atom 
    //
    // Calculate residual energy for i = 0
    double neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"]; 
    double terminal_residual_energy_12_0 = melt_configs.getTerminalSegmentReplacementResidualEnergy(
        2, TerminalSegmentEnd::TAIL, 4, 0, 
        segment(Eigen::seqN(0, 0), Eigen::all), segment.row(0), lj_params, 
        neighbor_threshold
    );
    double sum = 0;
    for (int i = 0; i < 5; ++i)    // Omit the last five atoms, since atom 5 is bonded to the new atom
    {
        double dist = (segment.row(0) - coords_chain2_1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    for (int i = 0; i < 10; ++i)    // Run through atoms on other chains
    {
        double dist = (segment.row(0) - coords_chain0_1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
        dist = (segment.row(0) - coords_chain1_1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(terminal_residual_energy_12_0, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 1
    double terminal_residual_energy_12_1 = melt_configs.getTerminalSegmentReplacementResidualEnergy(
        2, TerminalSegmentEnd::TAIL, 4, 1,
        segment(Eigen::seqN(0, 1), Eigen::all), segment.row(1), lj_params, 
        neighbor_threshold
    ); 
    sum = 0; 
    for (int i = 0; i < 6; ++i)    // Omit the last four atoms 
    {
        double dist = (segment.row(1) - coords_chain2_1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    for (int i = 0; i < 10; ++i)    // Run through atoms on other chains
    {
        double dist = (segment.row(1) - coords_chain0_1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
        dist = (segment.row(1) - coords_chain1_1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(terminal_residual_energy_12_1, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 2
    double terminal_residual_energy_12_2 = melt_configs.getTerminalSegmentReplacementResidualEnergy(
        2, TerminalSegmentEnd::TAIL, 4, 2,
        segment(Eigen::seqN(0, 2), Eigen::all), segment.row(2), lj_params, 
        neighbor_threshold
    );
    sum = 0;
    for (int i = 0; i < 6; ++i)    // Omit the last four atoms
    {
        double dist = (segment.row(2) - coords_chain2_1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    sum += lj<double>(
        (segment.row(2) - segment.row(0)).norm(),    // Add in atom 0 in the segment
        lj_params["eps"], lj_params["sigma"], true
    ); 
    for (int i = 0; i < 10; ++i)    // Run through atoms on other chains
    {
        double dist = (segment.row(2) - coords_chain0_1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
        dist = (segment.row(2) - coords_chain1_1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(terminal_residual_energy_12_2, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 3
    double terminal_residual_energy_12_3 = melt_configs.getTerminalSegmentReplacementResidualEnergy(
        2, TerminalSegmentEnd::TAIL, 4, 3,
        segment(Eigen::seqN(0, 3), Eigen::all), segment.row(3), lj_params, 
        neighbor_threshold
    );
    sum = 0;
    for (int i = 0; i < 6; ++i)    // Omit the last four atoms
    {
        double dist = (segment.row(3) - coords_chain2_1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    sum += lj<double>(
        (segment.row(3) - segment.row(0)).norm(),    // Add in atom 0 in the segment
        lj_params["eps"], lj_params["sigma"], true
    );
    sum += lj<double>(
        (segment.row(3) - segment.row(1)).norm(),    // Add in atom 1 in the segment 
        lj_params["eps"], lj_params["sigma"], true
    ); 
    for (int i = 0; i < 10; ++i)    // Run through atoms on other chains
    {
        double dist = (segment.row(3) - coords_chain0_1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
        dist = (segment.row(3) - coords_chain1_1.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(terminal_residual_energy_12_3, Catch::Matchers::WithinAbs(sum, tol));

    // Generate the reverse configuration 
    PolymerMeltConfiguration<double> melt_configs3(melt_configs2); 
    melt_configs3.replaceSegment(
        2, coords_chain2_1(Eigen::seqN(6, 4), Eigen::all), 6
    );

    // Check that the coordinates are updated correctly  
    Matrix<double, Dynamic, 3> coords_chain2_3 = melt_configs3.getSegment(2, 0, 10);
    for (int i = 0; i < l2; ++i) 
        REQUIRE((coords_chain2_3.row(i) - coords_chain2_1.row(i)).norm() < tol);

    // Calculate the reverse residual energy for each atom in the original
    // configuration 
    //
    // This should be the non-bonded energy between:
    // - each atom with index > 6 (0-indexed) in the new configuration and
    //   the new atom (in the original configuration), minus the adjacent atom,
    //   and 
    // - each atom in every other chain with the new atom (in the original 
    //   configuration)  
    //
    // Calculate residual energy for i = 0, which is atom 6 in the original 
    // configuration
    Matrix<double, Dynamic, 3> coords_chain0_2 = melt_configs2.getSegment(0, 0, 10); 
    Matrix<double, Dynamic, 3> coords_chain1_2 = melt_configs2.getSegment(1, 0, 10); 
    Matrix<double, Dynamic, 3> subsegment(0, 3); 
    double terminal_residual_energy_23_0 = melt_configs2.getTerminalSegmentReplacementResidualEnergy(
        2, TerminalSegmentEnd::TAIL, 4, 0, subsegment, coords_chain2_1.row(6), 
        lj_params, neighbor_threshold 
    );
    sum = 0;
    for (int i = 0; i < 5; ++i)    // Omit the last five atoms, since atom 5 is bonded to the new atom
    {
        double dist = (coords_chain2_1.row(6) - coords_chain2_2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    for (int i = 0; i < 10; ++i)    // Run through atoms on other chains
    {
        double dist = (coords_chain2_1.row(6) - coords_chain0_2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
        dist = (coords_chain2_1.row(6) - coords_chain1_2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(terminal_residual_energy_23_0, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 1
    subsegment.resize(1, 3); 
    subsegment.row(0) = coords_chain2_1.row(6); 
    double terminal_residual_energy_23_1 = melt_configs2.getTerminalSegmentReplacementResidualEnergy(
        2, TerminalSegmentEnd::TAIL, 4, 1, subsegment, coords_chain2_1.row(7), 
        lj_params, neighbor_threshold
    ); 
    sum = 0; 
    for (int i = 0; i < 6; ++i)    // Omit the last four atoms 
    {
        double dist = (coords_chain2_1.row(7) - coords_chain2_2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    for (int i = 0; i < 10; ++i)    // Run through atoms on other chains
    {
        double dist = (coords_chain2_1.row(7) - coords_chain0_2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
        dist = (coords_chain2_1.row(7) - coords_chain1_2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(terminal_residual_energy_23_1, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 2
    subsegment.conservativeResize(2, 3); 
    subsegment.row(1) = coords_chain2_1.row(7); 
    double terminal_residual_energy_23_2 = melt_configs2.getTerminalSegmentReplacementResidualEnergy(
        2, TerminalSegmentEnd::TAIL, 4, 2, subsegment, coords_chain2_1.row(8),
        lj_params, neighbor_threshold
    );
    sum = 0;
    for (int i = 0; i < 6; ++i)    // Omit the last four atoms 
    {
        double dist = (coords_chain2_1.row(8) - coords_chain2_2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    sum += lj<double>(
        (coords_chain2_1.row(8) - coords_chain2_1.row(6)).norm(),   // Add in atom 6 in the original configuration 
        lj_params["eps"], lj_params["sigma"], true
    ); 
    for (int i = 0; i < 10; ++i)    // Run through atoms on other chains
    {
        double dist = (coords_chain2_1.row(8) - coords_chain0_2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
        dist = (coords_chain2_1.row(8) - coords_chain1_2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(terminal_residual_energy_23_2, Catch::Matchers::WithinAbs(sum, tol));

    // Calculate residual energy for i = 3
    subsegment.conservativeResize(3, 3); 
    subsegment.row(2) = coords_chain2_1.row(8); 
    double terminal_residual_energy_23_3 = melt_configs2.getTerminalSegmentReplacementResidualEnergy(
        2, TerminalSegmentEnd::TAIL, 4, 3, subsegment, coords_chain2_1.row(9),
        lj_params, neighbor_threshold
    );
    sum = 0;
    for (int i = 0; i < 6; ++i)    // Omit the last four atoms 
    {
        double dist = (coords_chain2_1.row(9) - coords_chain2_2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    sum += lj<double>(
        (coords_chain2_1.row(9) - coords_chain2_1.row(6)).norm(),   // Add in atom 6 in the original configuration 
        lj_params["eps"], lj_params["sigma"], true
    );
    sum += lj<double>(
        (coords_chain2_1.row(9) - coords_chain2_1.row(7)).norm(),   // Add in atom 7 in the original configuration 
        lj_params["eps"], lj_params["sigma"], true
    ); 
    for (int i = 0; i < 10; ++i)    // Run through atoms on other chains
    {
        double dist = (coords_chain2_1.row(9) - coords_chain0_2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
        dist = (coords_chain2_1.row(9) - coords_chain1_2.row(i)).norm(); 
        sum += lj<double>(dist, lj_params["eps"], lj_params["sigma"], true); 
    }
    REQUIRE_THAT(terminal_residual_energy_23_3, Catch::Matchers::WithinAbs(sum, tol));
}

