/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     1/27/2026
 */

#include <iostream>
#include <string>
#include <unordered_map>
#include <functional>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/random.hpp>
#include "../include/utils.hpp"
#include "../include/cbmc.hpp"

/**
 * Generate a random K-mer in which the inter-atom distances, bond lengths, 
 * bond angles, and dihedral angles follow the given potentials. 
 *
 * The inter-atom distances also obey a minimum distance criterion.
 *
 * @param lj_params
 * @param fene_params
 * @param angle_mode
 * @param angle_params
 * @param dihedral_params
 * @param r0
 * @param collision_threshold
 * @param max_tries_per_atom
 * @param rng
 * @param uniform_dist
 * @param units
 * @param temp
 * @returns  
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

int main(int argc, char** argv)
{
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> uniform_dist;  
    Matrix<double, 3, 1> r0 = Matrix<double, 3, 1>::Zero();
    std::unordered_map<std::string, double> lj_params, 
                                            fene_params, 
                                            angle_params, 
                                            dihedral_params;
    double kT = 1.380649e-2 * 300;
    lj_params["eps"] = kT; 
    lj_params["sigma"] = 0.9;
    fene_params["K"] = 30 * kT; 
    fene_params["R0"] = 1.5;
    angle_params["K"] = 20 * kT;
    angle_params["theta0"] = 160 * boost::math::constants::pi<double>() / 180;
    dihedral_params["K"] = 10 * kT;
    double collision_threshold = 1;
    int max_tries_per_atom = 50;
    int max_n_backtracks = 50;  

    PolymerConfiguration<double> config = generateKMer<double, 10>(
        lj_params, fene_params, AngleMode::COSINE, angle_params,
        dihedral_params, r0, collision_threshold, max_tries_per_atom, 
        max_n_backtracks, rng, uniform_dist, Units::NANO, 300
    );
    std::cout << config.getSegment(0, 10) << std::endl; 

    return 0; 
}
