/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     3/25/2026
 */

#ifndef POLYMER_CONFIGURATION_HPP
#define POLYMER_CONFIGURATION_HPP

#include <fstream>
#include <stdexcept>
#include <cmath>
#include <string>
#include <limits>
#include <regex>
#include <unordered_map>
#include <functional>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/random.hpp>
#include "utils.hpp"

using std::abs;
using boost::multiprecision::abs; 
using std::pow; 
using boost::multiprecision::pow;
using std::exp; 
using boost::multiprecision::exp; 
using std::min; 
using boost::multiprecision::min;
using std::log10; 
using boost::multiprecision::log10; 
using std::isinf; 

using namespace Eigen;

enum class Units
{
    MICRO,
    NANO
};

enum class AngleMode
{
    COSINE,
    GAUSSIAN
};

enum class ReptationDirection
{
    HEAD,
    TAIL
};

/**
 * A class for storing, manipulating, analyzing, and comparing linear polymer
 * configurations. 
 */
template <typename T>
class PolymerConfiguration 
{
    private:
        int length; 
        Matrix<T, Dynamic, 3> r;
        T temp;

        /**
         * Get the energy arising from all *non-bonded* interactions between
         * the given segment and the atoms along the polymer with indices
         * [0, ..., idx - 1] and [idx + n, ..., N - 1], where n is the segment
         * length and N is the polymer length.
         *
         * This method omits the non-bonded interactions between bonded atoms.
         *
         * @param segment Input segment. 
         * @param idx Index demarcating the polymer atoms to consider. 
         * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
         * @param neighbor_threshold Distance threshold for identifying
         *                           neighboring (non-bonded) atoms. 
         * @returns Non-bonded interaction energy between segment and polymer.  
         */
        T getSegmentNonbondedInteractionEnergy(const Ref<const Matrix<T, Dynamic, 3> >& segment,
                                               const int idx,
                                               std::unordered_map<std::string, T>& lj_params,  
                                               const T neighbor_threshold) const
        {
            // Check that the specified polymer indices to slice out of the 
            // polymer are valid
            const int n = segment.rows(); 
            if (idx < 0 || idx + n - 1 >= this->length)
                throw std::runtime_error(
                    "Specified segment cannot be inserted into polymer at specified index"
                ); 

            // Get the non-bonded interaction energy ...
            //
            // First identify all pairs of neighboring atoms within the segment
            Matrix<T, Dynamic, 3> neighbors_within = getNeighbors<T>(
                segment, neighbor_threshold
            );

            // Then identify all pairs of neighboring atoms (p, q), where p
            // lies within the entire polymer and q lies within the segment
            const int n1 = idx; 
            const int n2 = this->length - idx - n; 
            Matrix<T, Dynamic, 3> r_sub(n1 + n2, 3);
            r_sub(Eigen::seqN(0, n1), Eigen::all) = this->r(Eigen::seqN(0, n1), Eigen::all); 
            r_sub(Eigen::seqN(n1, n2), Eigen::all) = this->r(Eigen::seqN(idx + n, n2), Eigen::all); 
            Matrix<T, Dynamic, 3> neighbors_between = getNeighbors<T>(
                r_sub, segment, neighbor_threshold
            );

            // Calculate the non-bonded interaction energy, omitting bonded 
            // pairs 
            T energy_curr = 0; 
            for (int i = 0; i < neighbors_within.rows(); ++i)
            {
                int j = neighbors_within(i, 0); 
                int k = neighbors_within(i, 1); 
                if (abs(j - k) > 1)
                {
                    energy_curr += lj<T>(
                        neighbors_within(i, 2), lj_params["eps"], lj_params["sigma"],
                        true
                    );
                }
            }
            for (int i = 0; i < neighbors_between.rows(); ++i)
            {
                int j = neighbors_between(i, 0);    // Atom in polymer 
                int k = neighbors_between(i, 1);    // Atom in segment
                bool adjacent = ((k == 0 && j == idx - 1) || (k == n - 1 && j == idx + n));
                if (!adjacent)
                    energy_curr += lj<T>(
                        neighbors_between(i, 2), lj_params["eps"], lj_params["sigma"],
                        true
                    );
            }

            return energy_curr; 
        } 

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
                                      std::unordered_map<std::string, T>& dihedral_params) const
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
            // First identify all pairs of neighboring atoms within the segment
            Matrix<T, Dynamic, 3> neighbors_within = getNeighbors<T>(
                segment, neighbor_threshold
            );

            // Then identify all pairs of neighboring atoms (p, q), where p
            // lies within the entire polymer and q lies within the segment
            const int n1 = idx; 
            const int n2 = this->length - idx - n; 
            Matrix<T, Dynamic, 3> r_sub(n1 + n2, 3);
            r_sub(Eigen::seqN(0, n1), Eigen::all) = this->r(Eigen::seqN(0, n1), Eigen::all); 
            r_sub(Eigen::seqN(n1, n2), Eigen::all) = this->r(Eigen::seqN(idx + n, n2), Eigen::all); 
            Matrix<T, Dynamic, 3> neighbors_between = getNeighbors<T>(
                r_sub, segment, neighbor_threshold
            );

            // Calculate the non-bonded interaction energy 
            T energy_curr = 0; 
            for (int i = 0; i < neighbors_within.rows(); ++i)
                energy_curr += lj<T>(
                    neighbors_within(i, 2), lj_params["eps"], lj_params["sigma"],
                    true
                );
            for (int i = 0; i < neighbors_between.rows(); ++i)
                energy_curr += lj<T>(
                    neighbors_between(i, 2), lj_params["eps"], lj_params["sigma"],
                    true
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
                adj_angles(n_adj_angles - 1, Eigen::seqN(0, 3)) = u / u.norm();
                adj_angles(n_adj_angles - 1, Eigen::seqN(3, 3)) = v / v.norm();  
            }
            if (idx > 0)
            {
                n_adj_angles++; 
                adj_angles.conservativeResize(n_adj_angles, 6);
                u = this->r.row(idx - 1) - segment.row(0); 
                v = segment.row(1) - segment.row(0);
                adj_angles(n_adj_angles - 1, Eigen::seqN(0, 3)) = u / u.norm();
                adj_angles(n_adj_angles - 1, Eigen::seqN(3, 3)) = v / v.norm();  
            }
            if (idx + n < this->length)
            {
                n_adj_angles++; 
                adj_angles.conservativeResize(n_adj_angles, 6);
                u = segment.row(n - 2) - segment.row(n - 1); 
                v = this->r.row(idx + n) - segment.row(n - 1); 
                adj_angles(n_adj_angles - 1, Eigen::seqN(0, 3)) = u / u.norm();
                adj_angles(n_adj_angles - 1, Eigen::seqN(3, 3)) = v / v.norm();  
            }
            if (idx + n + 1 < this->length)
            {
                n_adj_angles++; 
                adj_angles.conservativeResize(n_adj_angles, 6);
                u = segment.row(n - 1) - this->r.row(idx + n); 
                v = this->r.row(idx + n + 1) - this->r.row(idx + n); 
                adj_angles(n_adj_angles - 1, Eigen::seqN(0, 3)) = u / u.norm();
                adj_angles(n_adj_angles - 1, Eigen::seqN(3, 3)) = v / v.norm();  
            }
            for (int i = 0; i < n_adj_angles; ++i)
            {
                Matrix<T, 3, 1> u = adj_angles(i, Eigen::seqN(0, 3)); 
                Matrix<T, 3, 1> v = adj_angles(i, Eigen::seqN(3, 3)); 
                T theta = acosSafe<T>(u.dot(v));
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
         * Empty constructor. 
         *
         * A single atom is placed at the origin, and the temperature is 
         * assumed to be 300 K.
         */
        PolymerConfiguration()
        {
            this->length = 1; 
            this->r = Matrix<T, Dynamic, 3>::Zero(1, 3);
            this->temp = 300;
            this->kT = static_cast<T>(1.380649e-2) * temp;    // Assume nano units
        }

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
            this->temp = temp;  

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
         * Return the units. 
         *
         * @returns Units. 
         */
        Units getUnits() const 
        {
            if (log10(this->kT / this->temp) > -2)    // Boltzmann's constant = 1.380649e-2
                return Units::NANO; 
            else 
                return Units::MICRO; 
        }

        /**
         * Return the temperature. 
         *
         * @returns Temperature. 
         */
        T getTemp() const 
        {
            return this->temp; 
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
         * Get the radius of gyration of the polymer configuration. 
         *
         * @returns Radius of gyration. 
         */
        T radiusOfGyration() const 
        {
            // Get the deviation of each atom from the center of mass 
            Matrix<T, Dynamic, 3> deviations = this->r.rowwise() - this->r.colwise().mean();

            // Get the radius of gyration 
            T sum = 0; 
            for (int i = 0; i < this->length; ++i)
                sum += deviations.row(i).dot(deviations.row(i));
            return sqrt(sum / this->length);  
        }

        /**
         * Get the unit vectors tangent to each bond along the polymer
         * configuration. 
         *
         * @returns Array of tangent vectors. 
         */
        Matrix<T, Dynamic, 3> tangentVectors() const
        {
            // Get the unit vectors along the bonds
            Matrix<T, Dynamic, 3> bonds = (
                this->r(Eigen::seq(1, this->length - 1), Eigen::all) -
                this->r(Eigen::seq(0, this->length - 2), Eigen::all)
            );
            Matrix<T, Dynamic, 1> bond_lengths = bonds.rowwise().norm();
            return (bonds.array().colwise() / bond_lengths.array()).matrix(); 
        }
        
        /**
         * Get the average bond length. 
         *
         * @returns Mean bond length. 
         */
        T meanBondLength() const 
        {
            Matrix<T, Dynamic, 3> bonds = (
                this->r(Eigen::seq(1, this->length - 1), Eigen::all) -
                this->r(Eigen::seq(0, this->length - 2), Eigen::all)
            );
            return bonds.rowwise().norm().mean();
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
         * Get the minimum distance between the given atom and the polymer
         * under periodic boundary conditions.  
         *
         * @param p Input atomic coordinates.
         * @param xlen Length of periodic box in x. 
         * @param ylen Length of periodic box in y. 
         * @param zlen Length of periodic box in z. 
         * @returns Minimum distance between the atom and the polymer, assuming
         *          periodic boundary conditions. 
         */
        T getMinDistPeriodic(const Ref<const Matrix<T, 3, 1> >& p, const T xlen, 
                             const T ylen, const T zlen) const 
        {
            Matrix<T, Dynamic, 1> periodic_dists(this->length);
            for (int i = 0; i < this->length; ++i)
                periodic_dists(i) = periodicDistVec<T>(p, this->r.row(i), xlen, ylen, zlen).norm(); 

            return periodic_dists.minCoeff(); 
        }

        /**
         * Get the center of mass of the polymer, assuming that every atom 
         * has the same mass. 
         *
         * @returns Center of mass. 
         */
        Matrix<T, 3, 1> centerOfMass() const 
        {
            return this->r.colwise().mean(); 
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
            this->r(Eigen::seqN(1, this->length - 1), Eigen::all)
                = this->r(Eigen::seqN(0, this->length - 1), Eigen::all).eval();  
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
         * Pop the atom at the tail of the polymer. 
         */
        void popAtomFromTail()
        {
            this->r.conservativeResize(this->length - 1, 3);
            this->length--; 
        }

        /**
         * Pop the atom at the head of the polymer. 
         */
        void popAtomFromHead()
        {
            this->r(Eigen::seqN(1, this->length - 1), Eigen::all)
                = this->r(Eigen::seqN(0, this->length - 1), Eigen::all).eval(); 
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
         * Change the polymer according to a multimer reptation move towards
         * the tail, i.e., remove the first m atoms and add the given segment
         * to the other end. 
         * 
         * @param segment Atomic coordinates of new segment to be added to 
         *                the tail.
         */
        void reptateTowardsTailMultimer(const Ref<const Matrix<T, Dynamic, 3> >& segment)
        {
            // Copy over the current polymer coordinates
            const int m = segment.rows();  
            this->r(Eigen::seqN(0, this->length - m), Eigen::all)
                = this->r(Eigen::seqN(m, this->length - m), Eigen::all).eval();

            // Add the new segment 
            this->r(Eigen::seqN(this->length - m, m), Eigen::all) = segment; 
        }

        /**
         * Change the polymer according to a multimer reptation move towards
         * the head, i.e., remove the final m atoms and add the given segment
         * to the other end. 
         * 
         * @param segment Atomic coordinates of new segment to be added to 
         *                the head.
         */
        void reptateTowardsHeadMultimer(const Ref<const Matrix<T, Dynamic, 3> >& segment)
        {
            // Copy over the current polymer coordinates
            const int m = segment.rows();  
            this->r(Eigen::seqN(m, this->length - m), Eigen::all)
                = this->r(Eigen::seqN(0, this->length - m), Eigen::all).eval();

            // Add the new segment 
            this->r(Eigen::seqN(0, m), Eigen::all) = segment;
        }

        /**
         * Rotate the segment [0, ..., idx - 1] by the given angle about the 
         * given axis, with the indicated atom serving as the center.
         *
         * @param idx Index demarcating the head segment to rotate. 
         * @param theta Angle to rotate the segment by. 
         * @param u Rotation axis. 
         * @param idx_center Index of atom serving as the center of rotation.
         */
        void rotateHeadSegment(const int idx, const T theta,
                               const Ref<const Matrix<T, 3, 1> >& u, 
                               const int idx_center)
        {
            // The rotation center should be outside the segment being rotated
            if (idx_center < idx)
                throw std::runtime_error(
                    "Specified invalid center for rotation of head segment"
                ); 

            // Get the rotation matrix 
            Matrix<T, 3, 3> rot = getRotation<T>(u, theta); 

            // For each atom in the segment ...
            Matrix<T, 3, 1> center = this->r.row(idx_center); 
            for (int i = 0; i < idx; ++i)
            {
                // Transform the atom according to the rotation 
                Matrix<T, 3, 1> delta = this->r.row(i) - center; 
                this->r.row(i) = center + rot * delta; 
            }
        }

        /**
         * Rotate the segment [idx, ..., n - 1], where n is the polymer length,
         * by the given angle about the given axis, with the indicated atom
         * serving as the center.
         *
         * @param idx Index demarcating the head segment to rotate. 
         * @param theta Angle to rotate the segment by. 
         * @param u Rotation axis. 
         * @param idx_center Index of atom serving as the center of rotation.
         */
        void rotateTailSegment(const int idx, const T theta,
                               const Ref<const Matrix<T, 3, 1> >& u, 
                               const int idx_center)
        {
            // The rotation center should be outside the segment being rotated
            if (idx_center >= idx)
                throw std::runtime_error(
                    "Specified invalid center for rotation of head segment"
                ); 

            // Get the rotation matrix 
            Matrix<T, 3, 3> rot = getRotation<T>(u, theta); 

            // For each atom in the segment ...
            Matrix<T, 3, 1> center = this->r.row(idx_center); 
            for (int i = idx; i < this->length - 1; ++i)
            {
                // Transform the atom according to the rotation 
                Matrix<T, 3, 1> delta = this->r.row(i) - center; 
                this->r.row(i) = center + rot * delta; 
            }
        } 

        /**
         * Get the energetic contributions of the non-bonded (repulsive)
         * interactions between all atoms to the energy of the current polymer
         * configuration.
         *
         * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
         * @param neighbor_threshold Distance threshold for identifying
         *                           neighboring (non-bonded) atoms.
         * @param nonconsecutive If true, omit interactions between consecutive
         *                       atoms.  
         * @returns Non-bonded interaction energy. 
         */
        T getNonbondedEnergy(std::unordered_map<std::string, T>& lj_params, 
                             const T neighbor_threshold,
                             const bool nonconsecutive = false) const
        {
            T energy = 0.0; 

            // Identify all pairs of neighboring atoms 
            Matrix<T, Dynamic, 3> neighbors = getNeighbors<T>(this->r, neighbor_threshold);

            // Calculate all non-bonded interaction energies 
            for (int i = 0; i < neighbors.rows(); ++i)
            {
                // Skip over non-consecutive pairs if desired
                int j = static_cast<int>(neighbors(i, 0)); 
                int k = static_cast<int>(neighbors(i, 1)); 
                if (!nonconsecutive || (nonconsecutive && abs(k - j) > 1))
                {
                    //T term = lj<T>(
                    //    neighbors(i, 2), lj_params["eps"], lj_params["sigma"],
                    //    true
                    //);
                    //std::cout << "NONBONDED: " << neighbors(i, 0) << " " << neighbors(i, 1) << " " << neighbors(i, 2) << " " << term << std::endl; 
                    energy += lj<T>(
                        neighbors(i, 2), lj_params["eps"], lj_params["sigma"],
                        true
                    );
                }
            }

            return energy; 
        } 

        /**
         * Get the energetic contributions of the bonded interactions between
         * consecutive atoms to the energy of the current polymer configuration.
         *
         * The energetic contributions of repulsive Lennard-Jones interactions 
         * between consecutive atoms is also included, if desired. 
         *
         * @param fene_params FENE parameters.
         * @param include_lj If true, include the energetic contributions of 
         *                   repulsive Lennard-Jones interactions between 
         *                   consecutive atoms. 
         * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
         * @returns Bonded interaction energy. 
         */
        T getBondEnergy(std::unordered_map<std::string, T>& fene_params,
                        const bool include_lj = false, 
                        const std::unordered_map<std::string, T>& lj_params = {}) const
        {
            T energy = 0.0; 

            // Calculate all bond energies 
            for (int i = 0; i < this->length - 1; ++i)
            {
                Matrix<T, 3, 1> u = this->r.row(i + 1) - this->r.row(i);
                T bond_energy = bondFene<T>(u.norm(), fene_params["K"], fene_params["R0"]);

                // If the i-th bond energy is infinite, just return infinity 
                if (isinf(bond_energy))
                {
                    return std::numeric_limits<T>::infinity();
                } 
                else
                {
                    // Add the Lennard-Jones energy if desired
                    if (include_lj)
                        bond_energy += lj<T>(
                            u.norm(), lj_params.at("eps"), lj_params.at("sigma"),
                            true
                        );
                    energy += bond_energy; 
                } 
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
         * Get the total energy of the current polymer configuration. 
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
         * @returns Total energy.  
         */
        T getTotalEnergy(std::unordered_map<std::string, T>& lj_params,  
                         const T neighbor_threshold, 
                         std::unordered_map<std::string, T>& fene_params,
                         const AngleMode angle_mode,  
                         std::unordered_map<std::string, T>& angle_params,
                         std::unordered_map<std::string, T>& dihedral_params) const
        {
            T energy = 0; 
            energy += this->getNonbondedEnergy(lj_params, neighbor_threshold);
            energy += this->getBondEnergy(fene_params); 
            energy += this->getBondAngleEnergy(angle_mode, angle_params); 
            energy += this->getDihedralAngleEnergy(dihedral_params); 

            return energy; 
        }

        /**
         * Get the residual (non-bonded) energy of the proposed reptation move.
         *
         * This function calculates the non-bonded energy between the new atom
         * and the other atoms that would remain in the polymer configuration
         * after reptating in either direction. 
         *
         * Note that the reptation direction does not matter for this
         * calculation.
         *
         * @param r_new Position of new atom. 
         * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
         * @param neighbor_threshold Distance threshold for identifying
         *                           neighboring (non-bonded) atoms. 
         * @returns Residual energy of the proposed reptation move. 
         */
        T getReptationResidualEnergy(const Ref<const Matrix<T, 3, 1> >& r_new, 
                                     std::unordered_map<std::string, T>& lj_params,  
                                     const T neighbor_threshold) const
        {
            const int n = this->length;
            T energy = 0; 

            // Get the energy between the new atom and every other atom 
            // that would not be bonded to it after reptation
            //
            // The direction does not matter for this calculation 
            for (int i = 1; i < n - 1; ++i)    // Omit atoms 0 and (n - 1)
            {
                T dist = (this->r.row(i) - r_new.transpose()).norm();
                if (dist < neighbor_threshold)
                    energy += lj<T>(
                        dist, lj_params["eps"], lj_params["sigma"], true
                    ); 
            }

            return energy; 
        }

        /**
         * Get the total energy difference between the current polymer
         * configuration and the configuration that would arise from reptating
         * the polymer in the given direction by adding the given atom. 
         *
         * @param direction Reptation direction. 
         * @param r_new Position of new atom. 
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
         * @returns Energy difference due to reptation.
         */
        T getReptationEnergyDifference(const ReptationDirection direction, 
                                       const Ref<const Matrix<T, 3, 1> >& r_new,
                                       std::unordered_map<std::string, T>& lj_params,  
                                       const T neighbor_threshold, 
                                       std::unordered_map<std::string, T>& fene_params,
                                       const AngleMode angle_mode,  
                                       std::unordered_map<std::string, T>& angle_params,
                                       std::unordered_map<std::string, T>& dihedral_params) const
        {
            const int n = this->length; 
            T energy_curr = 0; 
            T energy_new = 0; 
            if (direction == ReptationDirection::HEAD)
            {
                // Get the non-bonded energy contribution from atom (n - 1)
                // in the current configuration 
                for (int i = 0; i < n - 2; ++i)    // Omit atom (n - 2)
                {
                    T dij = (this->r.row(i) - this->r.row(n - 1)).norm(); 
                    if (dij < neighbor_threshold)
                        energy_curr += lj<T>(
                            dij, lj_params["eps"], lj_params["sigma"], true
                        ); 
                }

                // Then get the bond energy between atoms (n - 2) and (n - 1)
                Matrix<T, 3, 1> u1 = this->r.row(n - 2) - this->r.row(n - 1); 
                T d1 = u1.norm(); 
                energy_curr += lj<T>(d1, lj_params["eps"], lj_params["sigma"], true); 
                energy_curr += bondFene<T>(d1, fene_params["K"], fene_params["R0"]);

                // Then get the bond angle energy between atoms (n - 3), 
                // (n - 2), and (n - 1)
                Matrix<T, 3, 1> n1 = u1 / d1; 
                Matrix<T, 3, 1> u2 = this->r.row(n - 3) - this->r.row(n - 2); 
                Matrix<T, 3, 1> n2 = u2 / u2.norm();
                T theta = acosSafe<T>((-n1).dot(n2)); 
                if (angle_mode == AngleMode::COSINE)
                    energy_curr += angleCosine<T>(
                        theta, angle_params["K"], angle_params["theta0"]
                    ); 
                else 
                    energy_curr += angleDualGaussianMixture<T>(
                        theta, angle_params["A1"], angle_params["A2"], 
                        angle_params["w1"], angle_params["w2"],
                        angle_params["theta1"], angle_params["theta2"],
                        this->kT 
                    );

                // Then get the dihedral angle energy between atoms (n - 4),
                // (n - 3), (n - 2), (n - 1)
                T phi = getDihedral<T>(
                    this->r.row(n - 4), this->r.row(n - 3), this->r.row(n - 2),
                    this->r.row(n - 1)
                ); 
                energy_curr += dihedralHarmonic<T>(
                    phi, dihedral_params["K"], dihedral_params["d"],
                    dihedral_params["n"]
                );

                // Move onto the new configuration ...
                //
                // Get the energy contribution that would arise from
                // introducing the new atom at the head and removing atom 
                // (n - 1)
                for (int i = 1; i < n - 1; ++i)    // Omit atoms 0 and (n - 1)
                {
                    T dij = (this->r.row(i) - r_new.transpose()).norm(); 
                    if (dij < neighbor_threshold)
                        energy_new += lj<T>(
                            dij, lj_params["eps"], lj_params["sigma"], true
                        ); 
                }

                // Then get the bond energy between the new atom and atom 0
                u1 = this->r.row(0) - r_new.transpose();
                d1 = u1.norm(); 
                energy_new += lj<T>(d1, lj_params["eps"], lj_params["sigma"], true); 
                energy_new += bondFene<T>(d1, fene_params["K"], fene_params["R0"]);

                // Then get the bond angle energy between the new atom, atom 0,
                // and atom 1
                n1 = u1 / d1; 
                u2 = this->r.row(1) - this->r.row(0); 
                n2 = u2 / u2.norm();
                theta = acosSafe<T>((-n1).dot(n2)); 
                if (angle_mode == AngleMode::COSINE)
                    energy_new += angleCosine<T>(
                        theta, angle_params["K"], angle_params["theta0"]
                    ); 
                else 
                    energy_new += angleDualGaussianMixture<T>(
                        theta, angle_params["A1"], angle_params["A2"], 
                        angle_params["w1"], angle_params["w2"],
                        angle_params["theta1"], angle_params["theta2"],
                        this->kT 
                    );

                // Then get the dihedral angle energy between the new atom, 
                // atom 0, atom 1, and atom 2
                phi = getDihedral<T>(
                    r_new, this->r.row(0), this->r.row(1), this->r.row(2)
                );
                energy_new += dihedralHarmonic<T>(
                    phi, dihedral_params["K"], dihedral_params["d"],
                    dihedral_params["n"]
                );
            }
            else 
            {
                // Get the energy contribution from atom 0 in the current 
                // configuration 
                for (int i = 2; i < n; ++i)        // Omit atom 1
                {
                    T dij = (this->r.row(i) - this->r.row(0)).norm(); 
                    if (dij < neighbor_threshold)
                        energy_curr += lj<T>(
                            dij, lj_params["eps"], lj_params["sigma"], true
                        ); 
                }

                // Then get the bond energy between atoms 0 and 1
                Matrix<T, 3, 1> u1 = this->r.row(1) - this->r.row(0);
                T d1 = u1.norm(); 
                energy_curr += lj<T>(d1, lj_params["eps"], lj_params["sigma"], true); 
                energy_curr += bondFene<T>(d1, fene_params["K"], fene_params["R0"]);

                // Then get the bond angle energy between atoms 0, 1, 2
                Matrix<T, 3, 1> n1 = u1 / d1; 
                Matrix<T, 3, 1> u2 = this->r.row(2) - this->r.row(1);
                Matrix<T, 3, 1> n2 = u2 / u2.norm();
                T theta = acosSafe<T>((-n1).dot(n2)); 
                if (angle_mode == AngleMode::COSINE)
                    energy_curr += angleCosine<T>(
                        theta, angle_params["K"], angle_params["theta0"]
                    ); 
                else 
                    energy_curr += angleDualGaussianMixture<T>(
                        theta, angle_params["A1"], angle_params["A2"], 
                        angle_params["w1"], angle_params["w2"],
                        angle_params["theta1"], angle_params["theta2"],
                        this->kT 
                    );

                // Then get the dihedral angle energy between atoms 0, 1, 2, 3
                T phi = getDihedral<T>(
                    this->r.row(0), this->r.row(1), this->r.row(2), this->r.row(3)
                ); 
                energy_curr += dihedralHarmonic<T>(
                    phi, dihedral_params["K"], dihedral_params["d"],
                    dihedral_params["n"]
                );

                // Move onto the new configuration ... 
                //
                // Get the energy contribution that would arise from
                // introducing the new atom at the tail and removing atom 0
                for (int i = 1; i < n - 1; ++i)    // Omit atoms 0 and (n - 1)
                {
                    T dij = (this->r.row(i) - r_new.transpose()).norm(); 
                    if (dij < neighbor_threshold)
                        energy_new += lj<T>(
                            dij, lj_params["eps"], lj_params["sigma"], true
                        ); 
                }

                // Then get the bond energy between the new atom and atom (n - 1)
                u1 = this->r.row(n - 1) - r_new.transpose();
                d1 = u1.norm(); 
                energy_new += lj<T>(d1, lj_params["eps"], lj_params["sigma"], true); 
                energy_new += bondFene<T>(d1, fene_params["K"], fene_params["R0"]);

                // Then get the bond angle energy between the new atom, atom
                // (n - 1), and atom (n - 2)
                n1 = u1 / d1; 
                u2 = this->r.row(n - 2) - this->r.row(n - 1);
                n2 = u2 / u2.norm();
                theta = acosSafe<T>((-n1).dot(n2)); 
                if (angle_mode == AngleMode::COSINE)
                    energy_new += angleCosine<T>(
                        theta, angle_params["K"], angle_params["theta0"]
                    ); 
                else 
                    energy_new += angleDualGaussianMixture<T>(
                        theta, angle_params["A1"], angle_params["A2"], 
                        angle_params["w1"], angle_params["w2"],
                        angle_params["theta1"], angle_params["theta2"],
                        this->kT 
                    );

                // Then get the dihedral angle energy between the new atom,
                // atom (n - 1), atom (n - 2), atom (n - 3)
                phi = getDihedral<T>(
                    this->r.row(n - 3), this->r.row(n - 2), this->r.row(n - 1),
                    r_new
                ); 
                energy_new += dihedralHarmonic<T>(
                    phi, dihedral_params["K"], dihedral_params["d"],
                    dihedral_params["n"]
                );
            }

            // Return the energy difference 
            return energy_new - energy_curr;
        }

        /**
         * Get the residual (non-bonded) energy of the proposed position for 
         * the i-th atom (for some i = 0, ..., K - 1) in a multimer (K-atom)
         * reptation move.
         *
         * The positions of the previous atoms, j = 0, ..., i - 1, are also
         * given. 
         *
         * This function calculates the non-bonded energy between the new atom
         * and the other atoms that would remain in the polymer configuration
         * after reptating by K atoms in either direction. 
         *
         * @param direction Reptation direction.
         * @param K Number of atoms to reptate by. 
         * @param i Index of the new atom in the reptation move. 
         * @param segment Atomic coordinates of the preceding segment of atoms,
         *                j = 0, ..., i - 1. Must have i rows. 
         * @param r_new Position of new (i-th) atom. 
         * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
         * @param neighbor_threshold Distance threshold for identifying
         *                           neighboring (non-bonded) atoms. 
         * @returns Residual energy of the proposed reptation move. 
         */
        T getMultimerReptationResidualEnergy(const ReptationDirection direction, 
                                             const int K, const int i, 
                                             const Ref<const Matrix<T, Dynamic, 3> >& segment, 
                                             const Ref<const Matrix<T, 3, 1> >& r_new, 
                                             std::unordered_map<std::string, T>& lj_params,  
                                             const T neighbor_threshold) const
        {
            const int n = this->length;
            if (K <= 0 || i < 0 || i >= K || segment.rows() != i)
            {
                throw std::runtime_error(
                    "Invalid multimer reptation move data (K, i, segment)"
                ); 
            }
            T energy = 0;

            // Get the energy between the new atom and every other atom 
            // that would not be bonded to it after reptation
            if (direction == ReptationDirection::HEAD)
            {
                // Omit atoms n - K, ..., n - 1 within the current configuration
                //
                // If i == 0, then also omit atom 0, which would be bonded
                // to the new atom after reptation
                int min_idx = (i == 0 ? 1 : 0); 
                for (int j = min_idx; j < n - K; ++j)
                {
                    T dist = (this->r.row(j) - r_new.transpose()).norm();
                    if (dist < neighbor_threshold)
                        energy += lj<T>(
                            dist, lj_params["eps"], lj_params["sigma"], true
                        ); 
                }

                // Omit the final atom in the preceding segment
                //
                // Here, we assume that atom j == 0 is bonded to the 0-th
                // atom in the current configuration, atom j == 1 is bonded 
                // to atom j == 0, etc. 
                for (int j = 0; j < i - 1; ++j)
                {
                    T dist = (segment.row(j) - r_new.transpose()).norm(); 
                    if (dist < neighbor_threshold)
                        energy += lj<T>(
                            dist, lj_params["eps"], lj_params["sigma"], true
                        ); 
                }
            }
            else    // direction == ReptationDirection::TAIL
            {
                // Omit atoms 0, ..., K - 1 within the current configuration
                //
                // If i == 0, then also omit atom n - 1, which would be bonded
                // to the new atom after reptation
                int max_idx = (i == 0 ? n - 2 : n - 1);    // Inclusive 
                for (int j = K; j <= max_idx; ++j)
                {
                    T dist = (this->r.row(j) - r_new.transpose()).norm();
                    if (dist < neighbor_threshold)
                        energy += lj<T>(
                            dist, lj_params["eps"], lj_params["sigma"], true
                        ); 
                }

                // Omit the final atom in the preceding segment
                //
                // Here, we assume that atom j == 0 is bonded to the (n-1)-th 
                // atom in the current configuration, atom j == 1 is bonded
                // to atom j == 0, etc. 
                for (int j = 0; j < i - 1; ++j)
                {
                    T dist = (segment.row(j) - r_new.transpose()).norm(); 
                    if (dist < neighbor_threshold)
                        energy += lj<T>(
                            dist, lj_params["eps"], lj_params["sigma"], true
                        ); 
                }
            }

            return energy; 
        }

        /**
         * Get the *non-bonded* energy difference between the current polymer
         * configuration and the configuration that would arise from reptating
         * the polymer in the given direction by the given segment.
         *
         * This function omits the non-bonded energetic contribution between
         * the old/new terminal atoms and their bonded neighbors.  
         *
         * @param direction Reptation direction. 
         * @param segment Atomic coordinates of new segment.
         * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
         * @param neighbor_threshold Distance threshold for identifying
         *                           neighboring (non-bonded) atoms. 
         * @returns Nonbonded energy difference due to multimer reptation. 
         */
        T getMultimerReptationNonbondedEnergyDifference(const ReptationDirection direction, 
                                                        const Ref<const Matrix<T, Dynamic, 3> >& segment, 
                                                        std::unordered_map<std::string, T>& lj_params,  
                                                        const T neighbor_threshold) const
        {
            const int n = this->length;
            const int m = segment.rows();    // Segment length 
            T energy_curr = 0; 
            T energy_new = 0; 
            if (direction == ReptationDirection::HEAD)
            {
                // Get the non-bonded energy contribution from the final m 
                // atoms (n - m, ..., n - 1) in the current configuration
                //
                // Omit the interaction between (n - m - 1) and (n - m) 
                for (int i = 0; i < n - m; ++i)
                {
                    int min_idx = (i == n - m - 1 ? n - m + 1 : n - m); 
                    for (int j = min_idx; j < n; ++j)
                    {
                        T dij = (this->r.row(i) - this->r.row(j)).norm(); 
                        if (dij < neighbor_threshold)
                            energy_curr += lj<T>(
                                dij, lj_params["eps"], lj_params["sigma"], true
                            );
                    } 
                }

                // Get the energy contribution that would arise from
                // introducing the new segment at the head and removing
                // the final m atoms
                //
                // Omit the interaction between atom 0 in the current 
                // configuration and the last atom in the new segment 
                for (int i = 0; i < n - m; ++i)
                {
                    int max_idx = (i == 0 ? m - 2 : m - 1); 
                    for (int j = 0; j <= max_idx; ++j)
                    {
                        T dij = (this->r.row(i) - segment.row(j)).norm(); 
                        if (dij < neighbor_threshold)
                            energy_new += lj<T>(
                                dij, lj_params["eps"], lj_params["sigma"], true
                            );
                    } 
                }
            }
            else 
            {
                // Get the energy contribution from the first m atoms in the
                // current configuration 
                //
                // Omit the interaction between (m - 1) and m
                for (int i = m; i < n; ++i)
                {
                    int max_idx = (i == m ? m - 2 : m - 1); 
                    for (int j = 0; j <= max_idx; ++j)
                    {
                        T dij = (this->r.row(i) - this->r.row(j)).norm(); 
                        if (dij < neighbor_threshold)
                            energy_curr += lj<T>(
                                dij, lj_params["eps"], lj_params["sigma"], true
                            );
                    } 
                }

                // Get the energy contribution that would arise from
                // introducing the new segment at the tail and removing the
                // first m atoms 
                //
                // Omit the interaction between atom (n - 1) in the current
                // configuration and the first atom in the new segment
                for (int i = m; i < n; ++i)
                {
                    int min_idx = (i == n - 1 ? 1 : 0); 
                    for (int j = min_idx; j < m; ++j)
                    {
                        T dij = (this->r.row(i) - segment.row(j)).norm(); 
                        if (dij < neighbor_threshold)
                            energy_new += lj<T>(
                                dij, lj_params["eps"], lj_params["sigma"], true
                            );
                    } 
                }
            }

            // Return the energy difference 
            return energy_new - energy_curr; 
        }

        /**
         * Get the *non-bonded* energy difference between the current polymer
         * configuration and the configuration that would arise from replacing
         * the current segment at the given index with the given segment.
         *
         * This function omits the non-bonded energetic contribution between
         * all bonded pairs of atoms. 
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
        T getSegmentReplacementNonbondedEnergyDifference(const Ref<const Matrix<T, Dynamic, 3> >& segment,
                                                         const int idx,
                                                         std::unordered_map<std::string, T>& lj_params,  
                                                         const T neighbor_threshold) const 
        {
            // Get the current segment 
            const int n = segment.rows(); 
            Matrix<T, Dynamic, 3> segment_curr = this->getSegment(idx, n);

            // Get the energy difference 
            T energy_curr = this->getSegmentNonbondedInteractionEnergy(
                segment_curr, idx, lj_params, neighbor_threshold
            ); 
            T energy_new = this->getSegmentNonbondedInteractionEnergy(
                segment, idx, lj_params, neighbor_threshold
            ); 
            return energy_new - energy_curr; 
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
                                                std::unordered_map<std::string, T>& dihedral_params) const
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

        /**
         * Write the polymer configuration to file in LAMMPS data format.
         *
         * The polymer coordinates are wrapped into the fundamental cell, 
         * to enforce periodic boundary conditions.  
         *
         * @param filename Output filename. 
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
         * @param header Header string. 
         * @param xmin, xmax Domain limits along x-axis.
         * @param ymin, ymax Domain limits along y-axis. 
         * @param zmin, zmax Domain limits along z-axis.
         * @param mass Atom mass.  
         */
        void writeLammps(const std::string& filename,
                         std::unordered_map<std::string, T>& lj_params, 
                         std::unordered_map<std::string, T>& fene_params, 
                         const AngleMode angle_mode,  
                         std::unordered_map<std::string, T>& angle_params, 
                         std::unordered_map<std::string, T>& dihedral_params, 
                         const std::string& header, const T xmin, const T xmax,
                         const T ymin, const T ymax, const T zmin, const T zmax,
                         const T mass)
        {
            std::ofstream outfile(filename);
            outfile << std::setprecision(10);  

            // Write header 
            outfile << header << "\n\n"; 

            // Write numbers of atoms, bonds, angles, and dihedrals
            //
            // Only count the numbers of angles or dihedrals if the potentials
            // are non-trivial
            const bool no_angles = (angle_mode == AngleMode::COSINE && angle_params["K"] == 0);
            const bool no_dihedrals = (dihedral_params["K"] == 0); 
            const int n_angles = (no_angles ? 0 : this->length - 2); 
            const int n_dihedrals = (no_dihedrals ? 0 : this->length - 3);  
            outfile << this->length << " atoms\n"
                    << this->length - 1 << " bonds\n"
                    << n_angles << " angles\n"
                    << n_dihedrals << " dihedrals\n"
                    << "0 impropers\n\n"; 

            // Write numbers of atom, bond, angle, and dihedral types
            int n_angle_types = !no_angles; 
            int n_dihedral_types = !no_dihedrals;
            outfile << "1 atom types\n1 bond types\n"
                    << n_angle_types << " angle types\n"
                    << n_dihedral_types << " dihedral types\n"
                    << "0 improper types\n\n";

            // Write box dimensions 
            outfile << xmin << " " << xmax << " xlo xhi\n"
                    << ymin << " " << ymax << " ylo yhi\n"
                    << zmin << " " << zmax << " zlo zhi\n\n"; 

            // Write atom masses 
            outfile << "Masses\n\n"
                    << "1 " << mass << "\n\n";

            // Write Lennard-Jones parameters
            outfile << "PairIJ Coeffs\n\n"
                    << "1 1 " << lj_params["eps"] << " "
                    << lj_params["sigma"] << " "
                    << pow(2, 1. / 6.) * lj_params["sigma"] << "\n\n";

            // Write FENE parameters
            outfile << "Bond Coeffs\n\n"
                    << "1 " << fene_params["K"] << " "
                    << fene_params["R0"] << " "
                    << lj_params["eps"] << " "
                    << lj_params["sigma"] << "\n\n"; 

            // Write angle potential parameters, as long as they are not trivial
            if (!no_angles && angle_mode == AngleMode::COSINE)
            {
                outfile << "Angle Coeffs\n\n"
                        << "1 " << angle_params["K"] << " "
                        << 180 * angle_params["theta0"] / boost::math::constants::pi<T>() << "\n\n"; 
            }
            else if (!no_angles && angle_mode == AngleMode::GAUSSIAN)
            {
                outfile << "Angle Coeffs\n\n"
                        << "1 " << this->temp << " "
                        << "2 " << angle_params["A1"] << " "
                        << angle_params["w1"] << " "
                        << 180 * angle_params["theta1"] / boost::math::constants::pi<T>() << " "
                        << angle_params["A2"] << " "
                        << angle_params["w2"] << " "
                        << 180 * angle_params["theta2"] / boost::math::constants::pi<T>() << "\n\n"; 
            }

            // Write dihedral potential parameters, as long as they are not
            // trivial 
            T dihedral_d, dihedral_n;
            if (dihedral_params.find("d") == dihedral_params.end())
                dihedral_d = 1; 
            else 
                dihedral_d = dihedral_params["d"]; 
            if (dihedral_params.find("n") == dihedral_params.end())
                dihedral_n = 1; 
            else 
                dihedral_n = dihedral_params["n"];
            if (!no_dihedrals)
            {
                outfile << "Dihedral Coeffs\n\n"
                        << "1 " << dihedral_params["K"] << " " 
                        << dihedral_d << " "
                        << dihedral_n << "\n\n";
            }

            // Write atom coordinates (all mapped to the fundamental cell
            // under periodic boundary conditions) 
            outfile << "Atoms\n\n";
            const int xlen = xmax - xmin; 
            const int ylen = ymax - ymin;
            const int zlen = zmax - zmin;
            for (int i = 0; i < this->length; ++i)
            {
                // Atom ID, molecule ID, atom type, x, y, z
                //
                // Atom IDs must be 1-indexed
                //
                // The last three values are the image flags 
                Matrix<T, 3, 1> r_mapped = mapToFundamentalCell<T>(
                    this->r.row(i), xlen, ylen, zlen, xmin, ymin, zmin
                ); 
                int image_x = static_cast<int>(floor((this->r(i, 0) - xmin) / xlen)); 
                int image_y = static_cast<int>(floor((this->r(i, 1) - ymin) / ylen)); 
                int image_z = static_cast<int>(floor((this->r(i, 2) - zmin) / zlen)); 
                outfile << i + 1 << " 1 1 " << r_mapped(0) << " "
                        << r_mapped(1) << " "
                        << r_mapped(2) << " "
                        << image_x << " " << image_y << " " << image_z 
                        << std::endl; 
            }
            outfile << std::endl; 

            // Write bonds 
            outfile << "Bonds\n\n"; 
            for (int i = 0; i < this->length - 1; ++i)
            {
                // Bond ID, bond type, atom i, atom j
                //
                // Atom IDs and bond IDs must be 1-indexed
                outfile << i + 1 << " 1 " << i + 1 << " " << i + 2 << std::endl; 
            }
            outfile << std::endl; 

            // Write angles, as long as they are not trivial
            if (!no_angles)
            {
                outfile << "Angles\n\n"; 
                for (int i = 0; i < this->length - 2; ++i)
                {
                    // Angle ID, angle type, atom i, atom j, atom k
                    //
                    // Atom IDs and angle IDs must be 1-indexed
                    outfile << i + 1 << " 1 " << i + 1 << " " << i + 2 << " "
                            << i + 3 << std::endl; 
                }
                outfile << std::endl;
            } 

            // Write dihedrals, as long as they are not trivial
            if (!no_dihedrals)
            { 
                outfile << "Dihedrals\n\n"; 
                for (int i = 0; i < this->length - 3; ++i)
                {
                    // Dihedral ID, dihedral type, atom i, atom j, atom k, atom l
                    //
                    // Atom IDs and dihedral IDs must be 1-indexed
                    outfile << i + 1 << " 1 " << i + 1 << " " << i + 2 << " "
                            << i + 3 << " " << i + 4 << std::endl; 
                }
                outfile << std::endl;
            } 
        }
};

/**
 * Parse the given LAMMPS data file. 
 *
 * @param filename Input filename.
 * @param units Units used in the LAMMPS file. 
 * @param temp Temperature.  
 * @returns Polymer configuration in the given file, along with all potential
 *          parameters.  
 */
template <typename T>
std::tuple<PolymerConfiguration<T>,
           std::unordered_map<std::string, T>,
           std::unordered_map<std::string, T>,
           AngleMode,
           std::unordered_map<std::string, T>,
           std::unordered_map<std::string, T> > parseLammps(const std::string& filename,
                                                            const Units units, 
                                                            const T temp)
{
    std::ifstream infile(filename);

    // Begin parsing the file ... 
    std::stringstream ss; 
    std::string line, token;
    int length = 0; 
    std::regex pattern("([0-9]+) atoms");
    std::smatch matches;  
    while (std::getline(infile, line))
    {
        // Skip to the line with the number of atoms
        if (std::regex_match(line, matches, pattern))
        {
            length = std::stoi(matches[1].str());
            break;
        }
    } 
    Matrix<T, Dynamic, 3> coords = Matrix<T, Dynamic, 3>::Zero(length, 3);

    // Keep parsing the file until we encounter the Lennard-Jones parameters 
    while (std::getline(infile, line))
    {
        if (line == "PairIJ Coeffs")
            break; 
    } 

    // Parse the Lennard-Jones parameters
    std::unordered_map<std::string, T> lj_params; 
    std::getline(infile, line);    // Skip blank line
    std::getline(infile, line);
    ss << line;  
    std::getline(ss, token, ' ');    // Skip first two entries in the line 
    std::getline(ss, token, ' ');
    std::getline(ss, token, ' ');    // Epsilon
    lj_params["eps"] = static_cast<T>(std::stod(token)); 
    std::getline(ss, token, ' ');    // Sigma (no need to parse last parameter)
    lj_params["sigma"] = static_cast<T>(std::stod(token)); 

    // Parse the FENE parameters
    std::unordered_map<std::string, T> fene_params; 
    std::getline(infile, line);    // Skip header and blank lines
    std::getline(infile, line);
    std::getline(infile, line);
    std::getline(infile, line);
    ss.clear(); 
    ss.str(std::string()); 
    ss << line; 
    std::getline(ss, token, ' ');    // Skip first entry in the line 
    std::getline(ss, token, ' ');    // K
    fene_params["K"] = static_cast<T>(std::stod(token)); 
    std::getline(ss, token, ' ');    // R0 (no need to parse remaining parameters)
    fene_params["R0"] = static_cast<T>(std::stod(token)); 

    // Parse the angle potential parameters
    AngleMode angle_mode;  
    std::unordered_map<std::string, T> angle_params; 
    std::getline(infile, line);    // Skip header and blank lines
    std::getline(infile, line);
    std::getline(infile, line);
    std::getline(infile, line);
    ss.clear(); 
    ss.str(std::string()); 
    ss << line;
    
    // First parse the number of parameters in the line 
    int n_params = 0; 
    while (std::getline(ss, token, ' '))
        n_params++; 

    // Distinguish between the two possible potentials 
    if (n_params == 3)
    {
        angle_mode = AngleMode::COSINE; 
        ss.clear(); 
        ss.str(std::string()); 
        ss << line; 
        std::getline(ss, token, ' ');    // Skip first entry in the line 
        std::getline(ss, token, ' ');    // K
        angle_params["K"] = static_cast<T>(std::stod(token));
        std::getline(ss, token, ' ');    // theta0 (convert to radians)
        angle_params["theta0"] = (
            static_cast<T>(std::stod(token)) * boost::math::constants::pi<T>() / 180
        );
    }
    else     // n_params == 9
    {
        angle_mode = AngleMode::GAUSSIAN; 
        ss.clear(); 
        ss.str(std::string()); 
        ss << line; 
        std::getline(ss, token, ' ');    // Skip first three entries in the line
        std::getline(ss, token, ' ');
        std::getline(ss, token, ' ');
        std::getline(ss, token, ' ');    // A1
        angle_params["A1"] = static_cast<T>(std::stod(token)); 
        std::getline(ss, token, ' ');    // w1
        angle_params["w1"] = static_cast<T>(std::stod(token)); 
        std::getline(ss, token, ' ');    // theta1 (convert to radians)
        angle_params["theta1"] = (
            static_cast<T>(std::stod(token)) * boost::math::constants::pi<T>() / 180
        ); 
        std::getline(ss, token, ' ');    // A2
        angle_params["A2"] = static_cast<T>(std::stod(token)); 
        std::getline(ss, token, ' ');    // w2
        angle_params["w2"] = static_cast<T>(std::stod(token)); 
        std::getline(ss, token, ' ');    // theta2 (convert to radians)
        angle_params["theta2"] = (
            static_cast<T>(std::stod(token)) * boost::math::constants::pi<T>() / 180
        );
    }

    // Parse the dihedral potential parameters
    std::unordered_map<std::string, T> dihedral_params; 
    std::getline(infile, line);    // Skip header and blank lines
    std::getline(infile, line);
    std::getline(infile, line);
    std::getline(infile, line);
    ss.clear(); 
    ss.str(std::string()); 
    ss << line;
    std::getline(ss, token, ' ');    // Skip first entry in the line 
    std::getline(ss, token, ' ');    // K
    dihedral_params["K"] = static_cast<T>(std::stod(token)); 
    std::getline(ss, token, ' ');    // d
    dihedral_params["d"] = static_cast<T>(std::stod(token));
    std::getline(ss, token, ' ');    // n
    dihedral_params["n"] = static_cast<T>(std::stod(token)); 
 
    // Parse the atomic coordinates
    std::getline(infile, line);    // Skip header and blank lines
    std::getline(infile, line);
    std::getline(infile, line);
    for (int i = 0; i < length; ++i)
    { 
        std::getline(infile, line);
        ss.clear(); 
        ss.str(std::string()); 
        ss << line;
        std::getline(ss, token, ' ');    // Skip first three entries in the line
        std::getline(ss, token, ' '); 
        std::getline(ss, token, ' '); 
        std::getline(ss, token, ' ');    // x-coordinate
        coords(i, 0) = static_cast<T>(std::stod(token));
        std::getline(ss, token, ' ');    // y-coordinate
        coords(i, 1) = static_cast<T>(std::stod(token)); 
        std::getline(ss, token, ' ');    // z-coordinate
        coords(i, 2) = static_cast<T>(std::stod(token)); 
    }

    // Generate polymer configuration and return 
    PolymerConfiguration<T> config(coords, units, temp); 
    return std::make_tuple(
        config, lj_params, fene_params, angle_mode, angle_params,
        dihedral_params
    ); 
}

/**
 * Generate a random K-mer in which the inter-atom distances, bond lengths, 
 * bond angles, and dihedral angles follow the given potentials.
 *
 * @param K Polymer length. 
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
 * @param bond_length_cdf Pre-defined CDF for bond length distribution.  
 * @param units Units for keeping track of Boltzmann's constant. 
 * @param temp Temperature (in Kelvin). 
 * @returns Resulting polymer configuration.  
 */
template <typename T>
PolymerConfiguration<T> generateKMer(const int K,
                                     std::unordered_map<std::string, T>& lj_params,
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
                                     const Ref<const Matrix<T, Dynamic, 2> >& bond_length_cdf,
                                     const T xmax = 0, const T ymax = 0, 
                                     const T zmax = 0,
                                     const Units units = Units::NANO,
                                     const T temp = 300)
{
    const T kT = (
        units == Units::MICRO ? static_cast<T>(1.380649e-8) * temp : 
        static_cast<T>(1.380649e-2) * temp
    );
    const T xlen = (xmax > 0 ? 2 * xmax : std::numeric_limits<T>::infinity());
    const T ylen = (ymax > 0 ? 2 * ymax : std::numeric_limits<T>::infinity()); 
    const T zlen = (zmax > 0 ? 2 * ymax : std::numeric_limits<T>::infinity());  

    // Define the angle sampling function  
    std::function<T(boost::random::mt19937&)> sample_angle;
    if (angle_mode == AngleMode::COSINE)
    {
        sample_angle = [&angle_params, &uniform_dist, &kT](boost::random::mt19937& rng_) -> T
        {
            return sampleAngleCosine<T>(
                angle_params["K"], angle_params["theta0"], kT, rng_, 
                uniform_dist
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
                kT, rng_, uniform_dist
            );
        };
    }
    else 
    {
        throw std::runtime_error("Invalid angle potential mode specified"); 
    }

    // Sample an initial bond length 
    T length = sampleFene<T>(rng, uniform_dist, bond_length_cdf);
    T angle, dihedral;  

    // Generate a PolymerConfiguration<T> instance with the first 2 atoms 
    Matrix<T, Dynamic, 3> coords(K, 3); 
    coords.row(0) = r0; 
    coords.row(1) = r0 + length * randomDir<T, 3>(rng, uniform_dist); 
    PolymerConfiguration<T> config(coords(Eigen::seqN(0, 2), Eigen::all), units, temp); 

    // Define a collision function 
    auto collision = [&collision_threshold, &xlen, &ylen, &zlen](PolymerConfiguration<T>& config, const Ref<const Matrix<T, 3, 1> >& r) -> bool
    {
        // Get the periodic distance with every atom within the growing 
        // K-mer (except for the atom to which it will be bonded)
        Matrix<T, Dynamic, 3> coords = config.getSegment(0, config.getLength() - 1);
        if (!isinf(xlen) && !isinf(ylen) && !isinf(zlen))
        {
            for (int i = 0; i < coords.rows(); ++i)
            {
                if (periodicDistVec<T>(r, coords.row(i), xlen, ylen, zlen).norm() < collision_threshold)
                    return true;
            }
            return false; 
        }
        else 
        {
            for (int i = 0; i < coords.rows(); ++i)
            {
                if ((coords.row(i) - r.transpose()).norm() < collision_threshold)
                    return true; 
            }
            return false;
        }
    };  

    // Add a 3rd atom ...
    //
    // Keep generating a new atom until no collision is detected 
    Matrix<T, 3, 1> new_atom;
    bool found_collision = true;  
    while (found_collision)
    {
        length = sampleFene<T>(rng, uniform_dist, bond_length_cdf);
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
            length = sampleFene<T>(rng, uniform_dist, bond_length_cdf);
            angle = sample_angle(rng);
            dihedral = sampleDihedralHarmonic<T>(
                dihedral_params["K"], kT, rng, uniform_dist
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

/**
 * Generate a random coil of length K, i.e., a K-mer with fixed bond lengths,
 * random angles, and random dihedrals. 
 *
 * @param K Polymer length.
 * @param bond_length Bond length. 
 * @param r0 Position of 0-th atom. 
 * @param rng Random number generator. 
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @param units Units for keeping track of Boltzmann's constant. 
 * @param temp Temperature (in Kelvin). 
 * @returns Resulting polymer configuration.  
 */
template <typename T>
PolymerConfiguration<T> generateKMerRandomCoil(const int K, const T bond_length, 
                                               const Ref<const Matrix<T, 3, 1> >& r0,
                                               boost::random::mt19937& rng,
                                               boost::random::uniform_01<>& uniform_dist,
                                               const Units units = Units::NANO, 
                                               const T temp = 300.0)
{
    // Generate each atom along the chain ... 
    //
    // Start with the first two atoms 
    Matrix<T, Dynamic, 3> coords(K, 3); 
    coords.row(0) = r0; 
    coords(1, 0) = r0(0) + bond_length; 
    coords(1, 1) = r0(1); 
    coords(1, 2) = r0(2);

    // Then generate each subsequent atom with a random bond angle
    for (int i = 2; i < K; ++i)
    {
        // Generate the next bond vector
        T u = -1 + 2 * uniform_dist(rng); 
        T r = sqrt(1 - u * u); 
        T phi = boost::math::constants::two_pi<T>() * uniform_dist(rng); 
        Matrix<T, 3, 1> v; 
        v << r * cos(phi), r * sin(phi), u;
        coords.row(i) = coords.row(i - 1) + bond_length * v.transpose();  
    } 

    // Generate a PolymerConfiguration<T> instance
    PolymerConfiguration<T> config(coords, units, temp);
    return config;  
}

/**
 * Generate a freely rotating chain of length K, i.e., a K-mer with fixed bond
 * lengths and angles, and random dihedrals.
 *
 * @param K Polymer length.
 * @param bond_length Bond length.
 * @param bond_angle Bond angle.  
 * @param r0 Position of 0-th atom. 
 * @param rng Random number generator. 
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @param units Units for keeping track of Boltzmann's constant. 
 * @param temp Temperature (in Kelvin). 
 * @returns Resulting polymer configuration.  
 */
template <typename T>
PolymerConfiguration<T> generateKMerFreelyRotatingChain(const int K,
                                                        const T bond_length,
                                                        const T bond_angle, 
                                                        const Ref<const Matrix<T, 3, 1> >& r0,
                                                        boost::random::mt19937& rng,
                                                        boost::random::uniform_01<>& uniform_dist,
                                                        const Units units = Units::NANO, 
                                                        const T temp = 300.0)
{
    // Generate each atom along the chain ... 
    //
    // Start with the first two atoms 
    Matrix<T, Dynamic, 3> coords(K, 3); 
    coords.row(0) = r0; 
    coords(1, 0) = r0(0) + bond_length; 
    coords(1, 1) = r0(1); 
    coords(1, 2) = r0(2);

    // Then generate each subsequent atom with the same bond length and angle,
    // but with a random dihedral 
    for (int i = 2; i < K; ++i)
    {
        // Generate the next bond vector
        coords.row(i) = generateNextAtom<T>(
            coords.row(i - 2), coords.row(i - 1), bond_length, bond_angle, 
            rng, uniform_dist
        ); 
    } 

    // Generate a PolymerConfiguration<T> instance
    PolymerConfiguration<T> config(coords, units, temp);
    return config;  
}

/**
 * Given arrays of tangent vectors along an ensemble of polymer configurations, 
 * get the corresponding autocorrelation in the tangent vector direction along
 * each polymer configuration, for the given increment k.
 *
 * @param tangent_vectors Collection of arrays of tangent vectors along each 
 *                        polymer configuration in an ensemble. 
 * @param k Input increment. 
 * @returns Array of n values (n = ensemble size), where the i-th value is 
 *          the autocorrelation in the tangent vector direction along the 
 *          i-th configuration, for the increment k.
 */
template <typename T>
Matrix<T, Dynamic, 1> getTangentVectorAutocorrelation(std::vector<Matrix<T, Dynamic, 3> >& tangent_vectors, 
                                                      const int k)
{
    // For each configuration ...
    const int n = tangent_vectors.size();              // Ensemble size 
    const int n_bonds = tangent_vectors[0].rows();     // Polymer length
    Matrix<T, Dynamic, 1> autocorrs_per_config = Matrix<T, Dynamic, 1>::Zero(n); 
    for (int i = 0; i < n; ++i)
    { 
        // Calculate the average of the dot product between the j-th and 
        // (j + k)-th tangent vectors in the i-th configuration
        for (int j = 0; j < n_bonds - k; ++j)
        {
            Matrix<T, 3, 1> u = tangent_vectors[i].row(j); 
            Matrix<T, 3, 1> v = tangent_vectors[i].row(j + k);
            autocorrs_per_config(i) += u.dot(v); 
        }
        autocorrs_per_config(i) /= (n_bonds - k);
    }

    return autocorrs_per_config;
}

#endif
