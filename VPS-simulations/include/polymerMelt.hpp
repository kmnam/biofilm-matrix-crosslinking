/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     4/3/2026
 */

#ifndef POLYMER_MELT_HPP
#define POLYMER_MELT_HPP

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
#include "polymerConfiguration.hpp"
#include "polymerEnsemble.hpp"

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

/**
 * A class for storing, manipulating, analyzing, and comparing linear polymer
 * melt configurations, each consisting of at least one polymer.  
 */
template <typename T>
class PolymerMeltConfiguration 
{
    private:
        int n; 
        std::vector<int> lengths;  
        T temp;
        PolymerEnsemble<T> configs; 

    public:
        T kT;     // Boltzmann's constant times temperature

        /**
         * Empty constructor. 
         *
         * A single atom is placed at the origin, and the temperature is 
         * assumed to be 300 K.
         */
        PolymerMeltConfiguration()
        {
            this->n = 1; 
            this->lengths.push_back(1);
            PolymerConfiguration<T> config;
            this->configs.push_back(config); 
            this->temp = 300;
            this->kT = static_cast<T>(1.380649e-2) * temp;    // Assume nano units
        }

        /**
         * Default constructor.
         *
         * @param n Number of polymers. 
         * @param r Atomic coordinates for each polymer.  
         * @param units Units for keeping track of Boltzmann's constant. 
         * @param temp Temperature (in Kelvin). 
         */
        PolymerMeltConfiguration(const int n,
                                 const std::vector<Matrix<T, Dynamic, 3> >& r, 
                                 const Units units, const T temp)
        {
            this->n = n; 

            // Check that the given number of polymers matches the number 
            // of entries in the coordinate vector 
            if (this->n != r.size())
                throw std::runtime_error(
                    "Number of polymers does not match coordinate vector"
                );
            for (auto it = r.begin(); it != r.end(); ++it)
            {
                // Instantiate each polymer configuration 
                PolymerConfiguration<T> config(*it, units, temp);
                this->configs.push_back(config); 
                this->lengths.push_back(it->rows()); 
            }
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
        ~PolymerMeltConfiguration()
        {
        } 

        /**
         * Return the bond lengths for the i-th polymer.  
         *
         * @param i Polymer index.
         * @returns Vector of bond lengths for the i-th polymer.  
         */
        Matrix<T, Dynamic, 1> bondLengths(const int i) const
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            return this->configs[i].bondLengths();  
        }

        /**
         * Return the length of the i-th polymer.
         *
         * @param i Polymer index.
         * @returns Length of the i-th polymer. 
         */
        int getLength(const int i) const 
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            return this->lengths[i];
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
         * Return the bond angles for the i-th polymer.
         *
         * @param i Polymer index.
         * @returns Vector of bond angles for the i-th polymer. 
         */
        Matrix<T, Dynamic, 1> bondAngles(const int i) const
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            return this->configs[i].bondAngles(); 
        }

        /**
         * Return the dihedral angles for the i-th polymer.  
         *
         * @param i Polymer index.
         * @returns Vector of dihedral angles for the i-th polymer. 
         */
        Matrix<T, Dynamic, 1> dihedralAngles(const int i) const 
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            return this->configs[i].dihedralAngles(); 
        }

        /**
         * Get the radius of gyration of the i-th polymer. 
         *
         * @param i Polymer index.
         * @returns Radius of gyration of the i-th polymer. 
         */
        T radiusOfGyration(const int i) const 
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            return this->configs[i].radiusOfGyration(); 
        }

        /**
         * Get, for the i-th polymer, the unit vectors tangent to each bond
         * along the polymer configuration. 
         *
         * @param i Polymer index.
         * @returns Array of tangent vectors for the i-th polymer.  
         */
        Matrix<T, Dynamic, 3> tangentVectors(const int i) const
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            return this->configs[i].tangentVectors(); 
        }
        
        /**
         * Get the average bond length for the i-th polymer.  
         *
         * @param i Polymer index. 
         * @returns Mean bond length for the i-th polymer. 
         */
        T meanBondLength(const int i) const 
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            return this->configs[i].meanBondLength(); 
        }

        /**
         * Get the atomic coordinates of the segment from `atom_idx` to
         * `atom_idx + seg_length` in the i-th polymer.
         *
         * @param i Polymer index.
         * @param atom_idx Index of first atom in the segment. 
         * @param seg_length Segment length.
         * @returns Atom coordinates of the segment.  
         */
        Matrix<T, Dynamic, 3> getSegment(const int i, const int atom_idx,
                                         const int seg_length) const 
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            return this->configs[i].getSegment(atom_idx, seg_length); 
        }

        /**
         * Get the minimum distance between the given atom and the i-th 
         * polymer. 
         *
         * @param i Polymer index.
         * @param p Input atomic coordinates. 
         * @returns Minimum distance between the atom and the i-th polymer. 
         */
        T getMinDist(const int i, const Ref<const Matrix<T, 3, 1> >& p) const 
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            return this->configs[i].getMinDist(p); 
        }

        /**
         * Get the center of mass of the i-th polymer, assuming that every atom 
         * has the same mass. 
         *
         * @param i Polymer index. 
         * @returns Center of mass of the i-th polymer. 
         */
        Matrix<T, 3, 1> centerOfMass(const int i) const 
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            return this->configs[i].centerOfMass(); 
        }

        /**
         * Replace the segment starting from atom `atom_idx` within the i-th
         * polymer with the given segment. 
         *
         * @param i Polymer index.
         * @param segment Array of atom coordinates for the new segment.
         * @param atom_idx Index of first atom to replace. 
         */
        void replaceSegment(const int i,
                            const Ref<const Matrix<T, Dynamic, 3> >& segment,
                            const int atom_idx)
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            this->configs[i].replaceSegment(segment, atom_idx); 
        }

        /**
         * Append the given atom onto the tail of the i-th polymer. 
         *
         * @param i Polymer index.
         * @param r Atom coordinates for the new atom. 
         */
        void appendAtomToTail(const int i, const Ref<const Matrix<T, 3, 1> >& r)
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            this->configs[i].appendAtomToTail(r);
            this->lengths[i] += 1;  
        }

        /**
         * Append the given atom onto the head of the i-th polymer. 
         *
         * @param i Polymer index.
         * @param r Atom coordinates for the new atom. 
         */
        void appendAtomToHead(const int i, const Ref<const Matrix<T, 3, 1> >& r)
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            this->configs[i].appendAtomToHead(r);
            this->lengths[i] += 1;  
        }

        /**
         * Append the given segment onto the tail of the i-th polymer. 
         *
         * @param i Polymer index.
         * @param segment Array of atom coordinates for the new segment. 
         */
        void appendSegmentToTail(const int i, 
                                 const Ref<const Matrix<T, Dynamic, 3> >& segment)
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            this->configs[i].appendSegmentToTail(segment);
            this->lengths[i] += segment.rows(); 
        }

        /**
         * Append the given segment onto the head of the i-th polymer.
         *
         * @param i Polymer index.
         * @param segment Array of atom coordinates for the new segment.  
         */
        void appendSegmentToHead(const int i,
                                 const Ref<const Matrix<T, Dynamic, 3> >& segment)
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            this->configs[i].appendSegmentToHead(segment);
            this->lengths[i] += segment.rows(); 
        }

        /**
         * Pop the atom at the tail of the i-th polymer. 
         *
         * @param i Polymer index.
         */
        void popAtomFromTail(const int i)
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            this->configs[i].popAtomFromTail();
            this->lengths[i] -= 1;  
        }

        /**
         * Pop the atom at the head of the i-th polymer. 
         *
         * @param i Polymer index.
         */
        void popAtomFromHead(const int i)
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            this->configs[i].popAtomFromHead();
            this->lengths[i] -= 1; 
        }

        /**
         * Pop the given segment from the tail of the i-th polymer. 
         *
         * @param i Polymer index.
         * @param atom_idx Index of first atom to remove from the polymer. 
         */
        void popSegmentFromTail(const int i, const int atom_idx)
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            this->configs[i].popSegmentFromTail(atom_idx);
            const int tail_length = this->lengths[i] - atom_idx; 
            this->lengths[i] -= tail_length; 
        }

        /**
         * Pop the given segment from the head of the polymer. 
         *
         * @param i Polymer index.
         * @param atom_idx Index of last atom to remove from the polymer. 
         */
        void popSegmentFromHead(const int i, const int atom_idx)
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            this->configs[i].popSegmentFromHead(atom_idx);
            const int head_length = atom_idx + 1; 
            this->lengths[i] -= head_length;  
        }

        /**
         * Change the i-th polymer according to a reptation move towards the
         * tail, i.e., remove the 0-th atom and add the given atom to the
         * other end.
         * 
         * @param i Polymer index. 
         * @param r New atom to be added to the tail. 
         */
        void reptateTowardsTail(const int i, const Ref<const Matrix<T, 3, 1> >& r)
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            this->configs[i].reptateTowardsTail(r); 
        }

        /**
         * Change the i-th polymer according to a reptation move towards the
         * head, i.e., remove the last atom and add the given atom to the
         * other end.
         * 
         * @param i Polymer index. 
         * @param r New atom to be added to the head. 
         */
        void reptateTowardsHead(const int i, const Ref<const Matrix<T, 3, 1> >& r)
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            this->configs[i].reptateTowardsHead(r); 
        }

        /**
         * Change the i-th polymer according to a multimer reptation move
         * towards the tail, i.e., remove the first m atoms and add the given
         * segment to the other end. 
         * 
         * @param i Polymer index. 
         * @param segment Atomic coordinates of new segment to be added to 
         *                the tail.
         */
        void reptateTowardsTailMultimer(const int i, 
                                        const Ref<const Matrix<T, Dynamic, 3> >& segment)
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            this->configs[i].reptateTowardsTailMultimer(segment); 
        }

        /**
         * Change the i-th polymer according to a multimer reptation move
         * towards the head, i.e., remove the final m atoms and add the given
         * segment to the other end. 
         * 
         * @param i Polymer index. 
         * @param segment Atomic coordinates of new segment to be added to 
         *                the head.
         */
        void reptateTowardsHeadMultimer(const int i, 
                                        const Ref<const Matrix<T, Dynamic, 3> >& segment)
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            this->configs[i].reptateTowardsHeadMultimer(segment); 
        }

        /**
         * Rotate the segment [0, ..., atom_idx - 1] in the i-th polymer by
         * the given angle about the given axis, with the indicated atom
         * serving as the center.
         *
         * @param i Polymer index. 
         * @param atom_idx Index demarcating the head segment to rotate. 
         * @param theta Angle to rotate the segment by. 
         * @param u Rotation axis. 
         * @param idx_center Index of atom serving as the center of rotation.
         */
        void rotateHeadSegment(const int i, const int atom_idx, const T theta,
                               const Ref<const Matrix<T, 3, 1> >& u, 
                               const int idx_center)
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            this->configs[i].rotateHeadSegment(atom_idx, theta, u, idx_center); 
        }

        /**
         * Rotate the segment [idx, ..., n - 1] in the i-th polymer, where n
         * is the polymer length, by the given angle about the given axis,
         * with the indicated atom serving as the center.
         *
         * @param i Polymer index. 
         * @param atom_idx Index demarcating the head segment to rotate. 
         * @param theta Angle to rotate the segment by. 
         * @param u Rotation axis. 
         * @param idx_center Index of atom serving as the center of rotation.
         */
        void rotateTailSegment(const int i, const int atom_idx, const T theta,
                               const Ref<const Matrix<T, 3, 1> >& u, 
                               const int idx_center)
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            this->configs[i].rotateTailSegment(atom_idx, theta, u, idx_center); 
        } 

        /**
         * Get the energetic contributions of the non-bonded (repulsive)
         * interactions between all atoms to the energy of the i-th polymer
         * configuration.
         *
         * @param i Polymer index. 
         * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
         * @param neighbor_threshold Distance threshold for identifying
         *                           neighboring (non-bonded) atoms.
         * @param nonconsecutive If true, omit interactions between consecutive
         *                       atoms.  
         * @returns Non-bonded interaction energy for the i-th polymer.  
         */
        T getNonbondedEnergy(const int i, std::unordered_map<std::string, T>& lj_params, 
                             const T neighbor_threshold,
                             const bool nonconsecutive = false) const
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            // Start with the non-bonded energy of the i-th polymer by itself 
            T energy = this->configs[i].getNonbondedEnergy(
                lj_params, neighbor_threshold, nonconsecutive
            );

            // Look for further non-bonded interactions with the other polymers
            const int ni = this->lengths[i]; 
            Matrix<T, Dynamic, 3> ri = this->configs[i].getSegment(0, ni);  
            for (int j = 0; j < this->n; ++j)
            {
                if (i != j)
                {
                    const int nj = this->lengths[j]; 
                    Matrix<T, Dynamic, 3> rj = this->configs[j].getSegment(0, nj);  
                    for (int k = 0; k < nj; ++k)
                    {
                        for (int p = 0; p < ni; ++p)
                        {
                            T dij = (rj.row(k) - ri.row(p)).norm();
                            if (dij < neighbor_threshold) 
                                energy += lj<T>(
                                    dij, lj_params["eps"], lj_params["sigma"], true
                                );
                        }
                    }
                }
            }

            return energy;  
        } 

        /**
         * Get the energetic contributions of the non-bonded (repulsive)
         * interactions between all atoms in the melt. 
         *
         * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
         * @param neighbor_threshold Distance threshold for identifying
         *                           neighboring (non-bonded) atoms.
         * @param nonconsecutive If true, omit interactions between consecutive
         *                       atoms.  
         * @returns Non-bonded interaction energy for the i-th polymer.  
         */
        T getTotalNonbondedEnergy(std::unordered_map<std::string, T>& lj_params, 
                                  const T neighbor_threshold,
                                  const bool nonconsecutive = false) const
        {
            // Start with the non-bonded energy within each polymer ... 
            T energy = 0; 
            for (int i = 0; i < this->n; ++i)
            {
                energy += this->configs[i].getNonbondedEnergy(
                    lj_params, neighbor_threshold, nonconsecutive
                );
            }

            // Then get the non-bonded energy between each pair of polymers ...
            for (int i = 0; i < this->n; ++i)
            { 
                const int ni = this->lengths[i]; 
                Matrix<T, Dynamic, 3> ri = this->configs[i].getSegment(0, ni);  
                for (int j = 0; j < this->n; ++j)
                {
                    if (i != j)
                    {
                        const int nj = this->lengths[j]; 
                        Matrix<T, Dynamic, 3> rj = this->configs[j].getSegment(0, nj);  
                        for (int k = 0; k < nj; ++k)
                        {
                            for (int p = 0; p < ni; ++p)
                            {
                                T dij = (rj.row(k) - ri.row(p)).norm();
                                if (dij < neighbor_threshold) 
                                    energy += lj<T>(
                                        dij, lj_params["eps"], lj_params["sigma"], true
                                    );
                            }
                        }
                    }
                }
            }

            return energy;  
        } 

        /**
         * Get the energetic contributions of the bonded interactions between
         * consecutive atoms to the energy of the i-th polymer configuration.
         *
         * The energetic contributions of repulsive Lennard-Jones interactions 
         * between consecutive atoms is also included, if desired. 
         *
         * @param i Polymer index. 
         * @param fene_params FENE parameters.
         * @param include_lj If true, include the energetic contributions of 
         *                   repulsive Lennard-Jones interactions between 
         *                   consecutive atoms. 
         * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
         * @returns Bonded interaction energy for the i-th polymer. 
         */
        T getBondEnergy(const int i, std::unordered_map<std::string, T>& fene_params,
                        const bool include_lj = false, 
                        const std::unordered_map<std::string, T>& lj_params = {}) const
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            return this->configs[i].getBondEnergy(fene_params, include_lj, lj_params); 
        }

        /**
         * Get the energetic contributions of the bonded interactions between
         * consecutive atoms to the energy of the entire melt. 
         *
         * The energetic contributions of repulsive Lennard-Jones interactions 
         * between consecutive atoms is also included, if desired. 
         *
         * @param fene_params FENE parameters.
         * @param include_lj If true, include the energetic contributions of 
         *                   repulsive Lennard-Jones interactions between 
         *                   consecutive atoms. 
         * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
         * @returns Bonded interaction energy for the i-th polymer. 
         */
        T getTotalBondEnergy(std::unordered_map<std::string, T>& fene_params,
                             const bool include_lj = false, 
                             const std::unordered_map<std::string, T>& lj_params = {}) const
        {
            T energy = 0; 
            for (int i = 0; i < this->n; ++i)
                energy += this->getBondEnergy(i, fene_params, include_lj, lj_params);

            return energy; 
        }

        /**
         * Get the energetic contributions of the bond angles to the energy 
         * of the i-th polymer configuration.
         *
         * @param i Polymer index. 
         * @param angle_mode Angle potential type.  
         * @param angle_params Angle potential parameters. Must include the 
         *                     cosine potential parameters (K and theta0) or
         *                     the dual Gaussian mixture potential parameters
         *                     (A1, A2, w1, w2, theta1, theta2). 
         * @returns Bond angle energy for the i-th polymer.  
         */
        T getBondAngleEnergy(const int i, const AngleMode angle_mode, 
                             std::unordered_map<std::string, T>& angle_params) const
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            return this->configs[i].getBondAngleEnergy(angle_mode, angle_params); 
        }

        /**
         * Get the energetic contributions of the bond angles to the energy 
         * of the entire melt. 
         *
         * @param angle_mode Angle potential type.  
         * @param angle_params Angle potential parameters. Must include the 
         *                     cosine potential parameters (K and theta0) or
         *                     the dual Gaussian mixture potential parameters
         *                     (A1, A2, w1, w2, theta1, theta2). 
         * @returns Bond angle energy for the i-th polymer.  
         */
        T getTotalBondAngleEnergy(const AngleMode angle_mode, 
                                  std::unordered_map<std::string, T>& angle_params) const
        {
            T energy = 0;
            for (int i = 0; i < this->n; ++i)
                energy += this->getBondAngleEnergy(i, angle_mode, angle_params); 
            
            return energy; 
        }

        /**
         * Get the energetic contributions of the dihedral angles along the 
         * polymer to the energy of the i-th polymer configuration.
         *
         * @param i Polymer index. 
         * @param dihedral_params Dihedral angle potential parameters. 
         * @returns Dihedral angle energy for the i-th polymer. 
         */
        T getDihedralAngleEnergy(const int i, 
                                 std::unordered_map<std::string, T>& dihedral_params) const
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            return this->configs[i].getDihedralAngleEnergy(dihedral_params); 
        }

        /**
         * Get the energetic contributions of the dihedral angles along the 
         * polymer to the energy of the entire melt. 
         *
         * @param dihedral_params Dihedral angle potential parameters. 
         * @returns Dihedral angle energy for the i-th polymer. 
         */
        T getTotalDihedralAngleEnergy(std::unordered_map<std::string, T>& dihedral_params) const
        {
            T energy = 0;
            for (int i = 0; i < this->n; ++i)
                energy += this->getDihedralAngleEnergy(i, dihedral_params); 
            
            return energy; 
        }

        /**
         * Get the total energy of the i-th polymer configuration. 
         *
         * @param i Polymer index.
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
         * @returns Total energy for the i-th polymer.  
         */
        T getTotalEnergy(const int i, std::unordered_map<std::string, T>& lj_params,  
                         const T neighbor_threshold, 
                         std::unordered_map<std::string, T>& fene_params,
                         const AngleMode angle_mode,  
                         std::unordered_map<std::string, T>& angle_params,
                         std::unordered_map<std::string, T>& dihedral_params) const
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            return this->configs[i].getTotalEnergy(
                lj_params, neighbor_threshold, fene_params, angle_mode, 
                angle_params, dihedral_params
            ); 
        }

        /**
         * Get the *non-bonded* energy difference between the current melt
         * configuration and the configuration that would arise from reptating
         * the i-th polymer in the given direction by adding the given atom. 
         *
         * This function omits the non-bonded energetic contribution between
         * the old/new terminal atoms and their bonded neighbors.  
         *
         * @param i Polymer index. 
         * @param direction Reptation direction. 
         * @param r_new Position of new atom. 
         * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
         * @param neighbor_threshold Distance threshold for identifying
         *                           neighboring (non-bonded) atoms. 
         * @returns Nonbonded energy difference due to reptation. 
         */
        T getReptationNonbondedEnergyDifference(const int i,
                                                const ReptationDirection direction, 
                                                const Ref<const Matrix<T, 3, 1> >& r_new,
                                                std::unordered_map<std::string, T>& lj_params,  
                                                const T neighbor_threshold) const
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            // Get the energy difference within the i-th polymer 
            T energy_diff_within = this->configs[i].getReptationNonbondedEnergyDifference(
                direction, r_new, lj_params, neighbor_threshold
            ); 

            // Run through all the remaining polymers ... 
            T energy_curr = 0; 
            T energy_new = 0;
            const int ni = this->lengths[i];
            Matrix<T, Dynamic, 3> ri = this->configs[i].getSegment(0, ni);   
            for (int j = 0; j < this->n; ++j)
            {
                if (i != j)
                {
                    const int nj = this->lengths[j];
                    Matrix<T, Dynamic, 3> rj = this->configs[j].getSegment(0, nj);  
                    if (direction == ReptationDirection::HEAD)
                    {
                        // Get the non-bonded energy contribution from atom
                        // (ni - 1) in the current i-th polymer configuration
                        for (int k = 0; k < nj; ++k)
                        {
                            T dij = (rj.row(k) - ri.row(ni - 1)).norm(); 
                            if (dij < neighbor_threshold)
                                energy_curr += lj<T>(
                                    dij, lj_params["eps"], lj_params["sigma"], true
                                ); 
                        }

                        // Get the energy contribution that would arise from
                        // introducing the new atom at the head and removing atom 
                        // (ni - 1)
                        for (int k = 0; k < nj; ++k)
                        {
                            T dij = (rj.row(k) - r_new.transpose()).norm();
                            if (dij < neighbor_threshold)
                                energy_new += lj<T>(
                                    dij, lj_params["eps"], lj_params["sigma"], true
                                ); 
                        }
                    }
                    else 
                    {
                        // Get the energy contribution from atom 0 in the current 
                        // i-th polymer configuration 
                        for (int k = 0; k < nj; ++k)
                        {
                            T dij = (rj.row(k) - ri.row(0)).norm(); 
                            if (dij < neighbor_threshold)
                                energy_curr += lj<T>(
                                    dij, lj_params["eps"], lj_params["sigma"], true
                                ); 
                        }

                        // Get the energy contribution that would arise from
                        // introducing the new atom at the tail and removing atom 0
                        for (int k = 0; k < nj; ++k)
                        {
                            T dij = (rj.row(k) - r_new.transpose()).norm(); 
                            if (dij < neighbor_threshold)
                                energy_new += lj<T>(
                                    dij, lj_params["eps"], lj_params["sigma"], true
                                ); 
                        }
                    }
                }
            }

            // Return the energy difference 
            return energy_diff_within + energy_new - energy_curr; 
        }

        /**
         * Get the total energy difference between the current melt 
         * configuration and the configuration that would arise from reptating
         * the i-th polymer in the given direction by adding the given atom. 
         *
         * @param i Polymer index. 
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
        T getReptationEnergyDifference(const int i, const ReptationDirection direction, 
                                       const Ref<const Matrix<T, 3, 1> >& r_new,
                                       std::unordered_map<std::string, T>& lj_params,  
                                       const T neighbor_threshold, 
                                       std::unordered_map<std::string, T>& fene_params,
                                       const AngleMode angle_mode,  
                                       std::unordered_map<std::string, T>& angle_params,
                                       std::unordered_map<std::string, T>& dihedral_params) const
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            // Get the energy difference within the i-th polymer 
            T energy_diff_within = this->configs[i].getReptationEnergyDifference(
                direction, r_new, lj_params, neighbor_threshold, fene_params, 
                angle_mode, angle_params, dihedral_params
            ); 

            // Run through all the remaining polymers ... 
            T energy_curr = 0; 
            T energy_new = 0; 
            const int ni = this->lengths[i];
            Matrix<T, Dynamic, 3> ri = this->configs[i].getSegment(0, ni);
            for (int j = 0; j < this->n; ++j)
            {
                if (i != j)
                {
                    const int nj = this->lengths[j]; 
                    Matrix<T, Dynamic, 3> rj = this->configs[j].getSegment(0, nj); 
                    if (direction == ReptationDirection::HEAD)
                    {
                        // Get the non-bonded energy contribution from atom
                        // (ni - 1) in the current i-th polymer configuration 
                        for (int k = 0; k < nj; ++k)
                        {
                            T dij = (rj.row(k) - ri.row(ni - 1)).norm(); 
                            if (dij < neighbor_threshold)
                                energy_curr += lj<T>(
                                    dij, lj_params["eps"], lj_params["sigma"], true
                                ); 
                        }

                        // Get the energy contribution that would arise from
                        // introducing the new atom at the head and removing atom 
                        // (ni - 1)
                        for (int k = 0; k < nj; ++k)
                        {
                            T dij = (rj.row(k) - r_new.transpose()).norm(); 
                            if (dij < neighbor_threshold)
                                energy_new += lj<T>(
                                    dij, lj_params["eps"], lj_params["sigma"], true
                                ); 
                        }
                    }
                    else 
                    {
                        // Get the energy contribution from atom 0 in the current 
                        // configuration 
                        for (int k = 0; k < nj; ++k)
                        {
                            T dij = (rj.row(k) - ri.row(0)).norm(); 
                            if (dij < neighbor_threshold)
                                energy_curr += lj<T>(
                                    dij, lj_params["eps"], lj_params["sigma"], true
                                ); 
                        }

                        // Get the energy contribution that would arise from
                        // introducing the new atom at the tail and removing atom 0
                        for (int k = 0; k < nj; ++k)
                        {
                            T dij = (rj.row(k) - r_new.transpose()).norm(); 
                            if (dij < neighbor_threshold)
                                energy_new += lj<T>(
                                    dij, lj_params["eps"], lj_params["sigma"], true
                                ); 
                        }
                    }
                }
            }

            // Return the energy difference 
            return energy_diff_within + energy_new - energy_curr;
        }

        /**
         * Get the *non-bonded* energy difference between the current melt 
         * configuration and the configuration that would arise from reptating
         * the i-th polymer in the given direction by the given segment.
         *
         * This function omits the non-bonded energetic contribution between
         * the old/new terminal atoms and their bonded neighbors.  
         *
         * @param i Polymer index. 
         * @param direction Reptation direction. 
         * @param segment Atomic coordinates of new segment.
         * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
         * @param neighbor_threshold Distance threshold for identifying
         *                           neighboring (non-bonded) atoms. 
         * @returns Nonbonded energy difference due to multimer reptation. 
         */
        T getMultimerReptationNonbondedEnergyDifference(const int i, 
                                                        const ReptationDirection direction, 
                                                        const Ref<const Matrix<T, Dynamic, 3> >& segment, 
                                                        std::unordered_map<std::string, T>& lj_params,  
                                                        const T neighbor_threshold) const
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            // Get the energy difference within the i-th polymer 
            T energy_diff_within = this->configs[i].getMultimerReptationNonbondedEnergyDifference(
                direction, segment, lj_params, neighbor_threshold
            ); 

            // Run through all the remaining polymers ... 
            T energy_curr = 0; 
            T energy_new = 0;
            const int ni = this->lengths[i];
            const int m = segment.rows();  
            Matrix<T, Dynamic, 3> ri = this->configs[i].getSegment(0, ni); 
            for (int j = 0; j < this->n; ++j)
            {
                if (i != j)
                {
                    const int nj = this->lengths[j];
                    Matrix<T, Dynamic, 3> rj = this->configs[j].getSegment(0, nj); 
                    if (direction == ReptationDirection::HEAD)
                    {
                        // Get the non-bonded energy contribution from the final m 
                        // atoms (n - m, ..., n - 1) in the current i-th polymer
                        // configuration
                        for (int k = 0; k < nj; ++k)
                        {
                            for (int p = ni - m; p < ni; ++p)
                            {
                                T dij = (rj.row(k) - ri.row(p)).norm(); 
                                if (dij < neighbor_threshold)
                                    energy_curr += lj<T>(
                                        dij, lj_params["eps"], lj_params["sigma"],
                                        true
                                    );
                            } 
                        }

                        // Get the energy contribution that would arise from
                        // introducing the new segment at the head and removing
                        // the final m atoms
                        for (int k = 0; k < nj; ++k)
                        {
                            for (int p = 0; p < m; ++p)
                            {
                                T dij = (rj.row(k) - segment.row(p)).norm(); 
                                if (dij < neighbor_threshold)
                                    energy_new += lj<T>(
                                        dij, lj_params["eps"], lj_params["sigma"],
                                        true
                                    );
                            } 
                        }
                    }
                    else 
                    {
                        // Get the energy contribution from the first m atoms
                        // in the current i-th polymer configuration 
                        for (int k = 0; k < nj; ++k)
                        {
                            for (int p = 0; p < m; ++p)
                            {
                                T dij = (rj.row(k) - ri.row(p)).norm(); 
                                if (dij < neighbor_threshold)
                                    energy_curr += lj<T>(
                                        dij, lj_params["eps"], lj_params["sigma"],
                                        true
                                    );
                            } 
                        }

                        // Get the energy contribution that would arise from
                        // introducing the new segment at the tail and removing
                        // the first m atoms 
                        for (int k = 0; k < nj; ++k)
                        {
                            for (int p = 0; p < m; ++p)
                            {
                                T dij = (rj.row(k) - segment.row(p)).norm(); 
                                if (dij < neighbor_threshold)
                                    energy_new += lj<T>(
                                        dij, lj_params["eps"], lj_params["sigma"],
                                        true
                                    );
                            } 
                        }
                    }
                }
            }

            // Return the energy difference 
            return energy_diff_within + energy_new - energy_curr; 
        }

        /**
         * Get the *non-bonded* energy difference between the current melt 
         * configuration and the configuration that would arise from replacing
         * the current segment at the given index in the i-th polymer with the
         * given segment.
         *
         * This function omits the non-bonded energetic contribution between
         * all bonded pairs of atoms. 
         *
         * @param i Polymer index. 
         * @param segment Input segment. 
         * @param atom_idx Index demarcating the polymer atoms to consider. 
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
        T getSegmentReplacementNonbondedEnergyDifference(const int i, 
                                                         const Ref<const Matrix<T, Dynamic, 3> >& segment,
                                                         const int atom_idx,
                                                         std::unordered_map<std::string, T>& lj_params,  
                                                         const T neighbor_threshold) const 
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            // Get the energy difference within the i-th polymer 
            T energy_diff_within = this->configs[i].getSegmentReplacementNonbondedEnergyDifference(
                segment, atom_idx, lj_params, neighbor_threshold
            ); 

            // Run through all the remaining polymers ... 
            T energy_curr = 0; 
            T energy_new = 0;
            const int ni = this->lengths[i];
            const int m = segment.rows();  
            Matrix<T, Dynamic, 3> ri = this->configs[i].getSegment(0, ni); 
            for (int j = 0; j < this->n; ++j)
            {
                if (i != j)
                {
                    const int nj = this->lengths[j];
                    Matrix<T, Dynamic, 3> rj = this->configs[j].getSegment(0, nj); 
                    
                    // Get the non-bonded energy contribution from the m-atom
                    // segment in the current i-th polymer configuration
                    for (int k = 0; k < nj; ++k)
                    {
                        for (int p = atom_idx; p < atom_idx + m; ++p)
                        {
                            T dij = (rj.row(k) - ri.row(p)).norm(); 
                            if (dij < neighbor_threshold)
                                energy_curr += lj<T>(
                                    dij, lj_params["eps"], lj_params["sigma"],
                                    true
                                );
                        } 
                    }

                    // Get the energy contribution that would arise from
                    // introducing the new m-atom segment
                    for (int k = 0; k < nj; ++k)
                    {
                        for (int p = 0; p < m; ++p)
                        {
                            T dij = (rj.row(k) - segment.row(p)).norm(); 
                            if (dij < neighbor_threshold)
                                energy_new += lj<T>(
                                    dij, lj_params["eps"], lj_params["sigma"],
                                    true
                                );
                        } 
                    }
                }
            }

            // Return the energy difference 
            return energy_diff_within + energy_new - energy_curr; 
        } 

        /**
         * Get the energy difference between the current melt configuration 
         * and the configuration that would arise from replacing the current
         * segment at the given index in the i-th polymer configuration with
         * the given segment. 
         *
         * @param i Polymer index.
         * @param segment Input segment. 
         * @param atom_idx Index demarcating the polymer atoms to consider. 
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
        T getSegmentReplacementEnergyDifference(const int i, 
                                                const Ref<const Matrix<T, Dynamic, 3> >& segment,
                                                const int atom_idx,
                                                std::unordered_map<std::string, T>& lj_params,  
                                                const T neighbor_threshold, 
                                                std::unordered_map<std::string, T>& fene_params,
                                                const AngleMode angle_mode,  
                                                std::unordered_map<std::string, T>& angle_params,
                                                std::unordered_map<std::string, T>& dihedral_params) const
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            // Get the energy difference within the i-th polymer 
            T energy_diff_within = this->configs[i].getSegmentReplacementEnergyDifference(
                segment, atom_idx, lj_params, neighbor_threshold, fene_params, 
                angle_mode, angle_params, dihedral_params
            ); 

            // Run through all the remaining polymers ... 
            T energy_curr = 0; 
            T energy_new = 0;
            const int ni = this->lengths[i];
            const int m = segment.rows();  
            Matrix<T, Dynamic, 3> ri = this->configs[i].getSegment(0, ni); 
            for (int j = 0; j < this->n; ++j)
            {
                if (i != j)
                {
                    const int nj = this->lengths[j];
                    Matrix<T, Dynamic, 3> rj = this->configs[j].getSegment(0, nj); 
                    
                    // Get the non-bonded energy contribution from the m-atom
                    // segment in the current i-th polymer configuration
                    for (int k = 0; k < nj; ++k)
                    {
                        for (int p = atom_idx; p < atom_idx + m; ++p)
                        {
                            T dij = (rj.row(k) - ri.row(p)).norm(); 
                            if (dij < neighbor_threshold)
                                energy_curr += lj<T>(
                                    dij, lj_params["eps"], lj_params["sigma"],
                                    true
                                );
                        } 
                    }

                    // Get the energy contribution that would arise from
                    // introducing the new m-atom segment
                    for (int k = 0; k < nj; ++k)
                    {
                        for (int p = 0; p < m; ++p)
                        {
                            T dij = (rj.row(k) - segment.row(p)).norm(); 
                            if (dij < neighbor_threshold)
                                energy_new += lj<T>(
                                    dij, lj_params["eps"], lj_params["sigma"],
                                    true
                                );
                        } 
                    }
                }
            }

            // Return the energy difference 
            return energy_diff_within + energy_new - energy_curr; 
        }

        /**
         * Write the melt configuration to file in LAMMPS data format.
         *
         * The polymers are 1-indexed in the output file, in accordance with
         * LAMMPS convention.
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
            int n_atoms = 0;
            int n_bonds = 0; 
            int n_angles = 0; 
            int n_dihedrals = 0; 
            const bool no_angles = (angle_mode == AngleMode::COSINE && angle_params["K"] == 0);
            const bool no_dihedrals = (dihedral_params["K"] == 0); 
            for (int i = 0; i < this->n; ++i)
            {
                n_atoms += this->lengths[i];
                n_bonds += (this->lengths[i] - 1);
                if (!no_angles)
                    n_angles += (this->lengths[i] - 2);
                if (!no_dihedrals) 
                    n_dihedrals += (this->lengths[i] - 3); 
            } 
            outfile << n_atoms << " atoms\n"
                    << n_bonds << " bonds\n"
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
            int offset = 0; 
            for (int i = 0; i < this->n; ++i)
            {
                const int ni = this->lengths[i]; 
                Matrix<T, Dynamic, 3> ri = this->configs[i].getSegment(0, ni);  
                for (int j = 0; j < ni; ++j)
                {
                    // Atom ID, molecule ID, atom type, x, y, z, nx, ny, nz
                    //
                    // Atom and molecule IDs must be 1-indexed
                    //
                    // The last three values are the image flags
                    int atom_id = offset + j + 1; 
                    int mol_id = i + 1; 
                    Matrix<T, 3, 1> rij_mapped = mapToFundamentalCell<T>(
                        ri.row(j), xlen, ylen, zlen, xmin, ymin, zmin
                    );
                    int image_x = static_cast<int>(floor((ri(j, 0) - xmin) / xlen)); 
                    int image_y = static_cast<int>(floor((ri(j, 1) - ymin) / ylen)); 
                    int image_z = static_cast<int>(floor((ri(j, 2) - zmin) / zlen)); 
                    outfile << atom_id << " " << mol_id << " 1 "
                            << rij_mapped(0) << " "
                            << rij_mapped(1) << " "
                            << rij_mapped(2) << " "
                            << image_x << " " << image_y << " " << image_z
                            << std::endl; 
                }
                offset += ni;  
            }
            outfile << std::endl; 

            // Write bonds 
            outfile << "Bonds\n\n";
            int atom_offset = 0;
            int bond_offset = 0;  
            for (int i = 0; i < this->n; ++i)
            {
                const int ni = this->lengths[i]; 
                for (int j = 0; j < ni - 1; ++j)
                {
                    // Bond ID, bond type, atom i, atom j
                    //
                    // Atom and bond IDs must be 1-indexed
                    int atom_id1 = atom_offset + j + 1;
                    int atom_id2 = atom_offset + j + 2; 
                    int bond_id = bond_offset + j + 1;   
                    outfile << bond_id << " 1 " << atom_id1 << " " << atom_id2 << std::endl;
                }
                atom_offset += ni;
                bond_offset += (ni - 1);  
            }
            outfile << std::endl; 

            // Write angles, as long as they are not trivial 
            if (!no_angles)
            {
                outfile << "Angles\n\n";
                atom_offset = 0; 
                int angle_offset = 0; 
                for (int i = 0; i < this->n; ++i)
                {
                    const int ni = this->lengths[i];  
                    for (int j = 0; j < ni - 2; ++j)
                    {
                        // Angle ID, angle type, atom i, atom j, atom k
                        //
                        // Atom and angle IDs must be 1-indexed
                        int atom_id1 = atom_offset + j + 1; 
                        int atom_id2 = atom_offset + j + 2; 
                        int atom_id3 = atom_offset + j + 3; 
                        int angle_id = angle_offset + j + 1; 
                        outfile << angle_id << " 1 " << atom_id1 << " " << atom_id2 << " "
                                << atom_id3 << std::endl; 
                    }
                    atom_offset += ni; 
                    angle_offset += (ni - 2); 
                }
                outfile << std::endl;
            } 

            // Write dihedrals, as long as they are not trivial 
            if (!no_dihedrals)
            {
                outfile << "Dihedrals\n\n";
                atom_offset = 0; 
                int dihedral_offset = 0; 
                for (int i = 0; i < this->n; ++i)
                {
                    const int ni = this->lengths[i]; 
                    for (int j = 0; j < ni - 3; ++j)
                    {
                        // Dihedral ID, dihedral type, atom i, atom j, atom k, atom l
                        //
                        // Atom IDs and dihedral IDs must be 1-indexed
                        int atom_id1 = atom_offset + j + 1; 
                        int atom_id2 = atom_offset + j + 2; 
                        int atom_id3 = atom_offset + j + 3; 
                        int atom_id4 = atom_offset + j + 4; 
                        int dihedral_id = dihedral_offset + j + 1; 
                        outfile << dihedral_id << " 1 " << atom_id1 << " "
                                << atom_id2 << " " << atom_id3 << " "
                                << atom_id4 << std::endl; 
                    }
                    atom_offset += ni; 
                    dihedral_offset += (ni - 3); 
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
 * @returns Polymer melt configuration in the given file, along with all
 *          potential parameters.  
 */
template <typename T>
std::tuple<PolymerMeltConfiguration<T>,
           std::unordered_map<std::string, T>,
           std::unordered_map<std::string, T>,
           AngleMode,
           std::unordered_map<std::string, T>,
           std::unordered_map<std::string, T> > parseMeltLammps(const std::string& filename,
                                                                const Units units, 
                                                                const T temp)
{
    std::ifstream infile(filename);

    // Begin parsing the file ...
    //
    // Keep parsing the file until we encounter the Lennard-Jones parameters 
    std::stringstream ss; 
    std::string line, token;
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
    //
    // Store each molecule's coordinates in a dictionary 
    std::vector<Matrix<T, Dynamic, 3> > coords;
    std::vector<int> lengths; 
    std::getline(infile, line);    // Skip header and blank lines
    std::getline(infile, line);
    std::getline(infile, line);
    while (std::getline(infile, line)) 
    {
        // Parse until the line is empty 
        if (line.empty())
            break; 

        // Parse the entries in the line 
        ss.clear(); 
        ss.str(std::string()); 
        ss << line;
        std::getline(ss, token, ' ');    // Atom ID (skip) 
        std::getline(ss, token, ' ');    // Molecule ID
        int mol_id = std::stoi(token); 
        std::getline(ss, token, ' ');    // Atom type (skip)
        std::getline(ss, token, ' ');    // x-coordinate
        T x = static_cast<T>(std::stod(token));
        std::getline(ss, token, ' ');    // y-coordinate
        T y = static_cast<T>(std::stod(token)); 
        std::getline(ss, token, ' ');    // z-coordinate
        T z = static_cast<T>(std::stod(token));

        // Has this molecule been encountered previously? 
        if (mol_id > coords.size())
        {
            // If not, initialize a new array
            Matrix<T, Dynamic, 3> new_coords(1, 3); 
            new_coords << x, y, z;
            coords.push_back(new_coords); 
            lengths.push_back(1);  
        }
        else 
        {
            // If so, append onto the array
            lengths[mol_id - 1]++; 
            coords[mol_id - 1].conservativeResize(lengths[mol_id - 1], 3);
            coords[mol_id - 1](lengths[mol_id - 1] - 1, 0) = x; 
            coords[mol_id - 1](lengths[mol_id - 1] - 1, 1) = y; 
            coords[mol_id - 1](lengths[mol_id - 1] - 1, 2) = z;  
        }
    }

    // Generate polymer configurations and return
    const int n = lengths.size(); 
    PolymerMeltConfiguration<T> configs(n, coords, units, temp); 
    return std::make_tuple(
        configs, lj_params, fene_params, angle_mode, angle_params,
        dihedral_params
    ); 
}

class TooManyBacktracksError : public std::runtime_error
{
    public:
        explicit TooManyBacktracksError(const std::string& msg) : std::runtime_error(msg) {}
}; 
class TooManySeedPositionsError : public std::runtime_error
{
    public:
        explicit TooManySeedPositionsError(const std::string& msg) : std::runtime_error(msg) {}
};
class PolymerGenerationError : public std::runtime_error
{
    public:
        explicit PolymerGenerationError(const std::string& msg) : std::runtime_error(msg) {}
}; 

/**
 * Generate the configuration of the first K-mer in a melt of K-mers, in
 * which the inter-atom distances, bond lengths, bond angles, and dihedral
 * angles follow the given potentials.
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
 * @param intra_collision_threshold Distance threshold for identifying atoms that 
 *                                  are too close to each other. 
 * @param max_tries_per_atom Maximum number of attempts to place each atom
 *                           before backtracking. 
 * @param max_n_backtracks Maximum number of backtracks. 
 * @param rng Random number generator. 
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @param xmax x-coordinate of first monomer of each chain is sampled from
 *             [-xmax, xmax], which is the x-range of the fundamental cell
 *             under periodic boundary conditions. 
 * @param ymax y-coordinate of first monomer of each chain is sampled from
 *             [-ymax, ymax], which is the y-range of the fundamental cell 
 *             under periodic boundary conditions. 
 * @param zmax z-coordinate of first monomer of each chain is sampled from
 *             [-zmax, zmax], which is the z-range of the fundamental cell
 *             under periodic boundary conditions. 
 * @param bond_length_cdf Pre-defined CDF for bond length distribution.  
 * @param units Units for keeping track of Boltzmann's constant. 
 * @param temp Temperature (in Kelvin).
 * @param verbose If true, print intermittent output to stdout.  
 * @returns Resulting polymer configuration.  
 */
template <typename T>
PolymerConfiguration<T> generateFirstKMer(const int K, 
                                          std::unordered_map<std::string, T>& lj_params,
                                          std::unordered_map<std::string, T>& fene_params,
                                          const AngleMode angle_mode,  
                                          std::unordered_map<std::string, T>& angle_params, 
                                          std::unordered_map<std::string, T>& dihedral_params,
                                          const T intra_collision_threshold, 
                                          const int max_tries_per_atom,
                                          const int max_n_backtracks,
                                          boost::random::mt19937& rng,
                                          boost::random::uniform_01<>& uniform_dist,
                                          const T xmax, const T ymax, const T zmax,
                                          const Ref<const Matrix<T, Dynamic, 2> >& bond_length_cdf,  
                                          const Units units = Units::NANO,
                                          const T temp = 300) 
{
    const T kT = (
        units == Units::MICRO ? static_cast<T>(1.380649e-8) * temp : 
        static_cast<T>(1.380649e-2) * temp
    );
    const T xlen = 2 * xmax; 
    const T ylen = 2 * ymax; 
    const T zlen = 2 * zmax;   

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

    // Seed the K-mer
    Matrix<T, 3, 1> r0; 
    r0 << -xmax + 2 * xmax * uniform_dist(rng), 
          -ymax + 2 * ymax * uniform_dist(rng), 
          -zmax + 2 * zmax * uniform_dist(rng);

    // Generate the second atom and a new PolymerConfiguration<T> instance
    Matrix<T, Dynamic, 3> coords(K, 3); 
    T length = sampleFene<T>(rng, uniform_dist, bond_length_cdf);
    coords.row(0) = r0;
    coords.row(1) = r0 + length * randomDir<T, 3>(rng, uniform_dist);  
    PolymerConfiguration<T> config(
        coords(Eigen::seqN(0, 2), Eigen::all), units, temp
    );

    // Define a collision function for the K-mer 
    std::function<bool(const Ref<const Matrix<T, 3, 1> >&)> collision_intra
        = [&config, &intra_collision_threshold, &xlen, &ylen, &zlen](const Ref<const Matrix<T, 3, 1> >& r) -> bool
    {
        // Get the periodic distance with every atom within the growing
        // K-mer (except for the atom to which it will be bonded)
        Matrix<T, Dynamic, 3> coords_ = config.getSegment(0, config.getLength() - 1);
        for (int i = 0; i < coords_.rows(); ++i) 
        {
            if (periodicDistVec<T>(r, coords_.row(i), xlen, ylen, zlen).norm() < intra_collision_threshold)
                return true;
        }
        return false;
    };  

    // Add a 3rd atom ...
    //
    // Keep generating a new atom until no collision is detected 
    Matrix<T, 3, 1> new_atom;
    bool found_collision = true;  
    while (found_collision)
    {
        length = sampleFene<T>(rng, uniform_dist, bond_length_cdf); 
        T angle = sample_angle(rng); 
        new_atom = generateNextAtom<T>(
            coords.row(0), coords.row(1), length, angle, rng, uniform_dist
        );
        found_collision = collision_intra(new_atom);  
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
            T angle = sample_angle(rng);
            T dihedral = sampleDihedralHarmonic<T>(
                dihedral_params["K"], kT, rng, uniform_dist
            );
            new_atom = generateNextAtomDihedral<T>(
                r1, r2, r3, length, angle, dihedral, rng, uniform_dist 
            );
            found_collision = collision_intra(new_atom); 
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
            throw TooManyBacktracksError(
                "Sampling procedure backtracked into first 3 atoms; try "
                "sampling more positions per atom"
            );
        }

        // If we have exceeded the maximum number of backtracks, try 
        // sampling again from scratch, since we are on the first polymer
        if (n_backtracks > max_n_backtracks)
        {
            throw TooManyBacktracksError(
                "Sampling procedure exceeded maximum number of backtracks; try "
                "sampling more positions per atom"
            );
        } 
    }

    return config; 
} 

/**
 * Generate the configuration of the i-th K-mer in a melt of K-mers, in
 * which the inter-atom distances, bond lengths, bond angles, and dihedral
 * angles follow the given potentials.
 *
 * @param K Polymer length.
 * @param melt_coords Polymer configurations accumulated thus far. 
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
 * @param intra_collision_threshold Distance threshold for identifying atoms
 *                                  that are too close to each other.
 * @param inter_collision_threshold 
 * @param max_tries_per_atom Maximum number of attempts to place each atom
 *                           before backtracking. 
 * @param max_tries_per_kmer Maximum number of attempts to generate a full
 *                           K-mer. 
 * @param max_tries_per_seed Maximum number of attempts to seed a K-mer. 
 * @param max_n_backtracks Maximum number of backtracks.
 * @param rng Random number generator. 
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @param xmax x-coordinate of first monomer of each chain is sampled from
 *             [-xmax, xmax], which is the x-range of the fundamental cell
 *             under periodic boundary conditions. 
 * @param ymax y-coordinate of first monomer of each chain is sampled from
 *             [-ymax, ymax], which is the y-range of the fundamental cell 
 *             under periodic boundary conditions. 
 * @param zmax z-coordinate of first monomer of each chain is sampled from
 *             [-zmax, zmax], which is the z-range of the fundamental cell
 *             under periodic boundary conditions. 
 * @param bond_length_cdf Pre-defined CDF for bond length distribution.  
 * @param units Units for keeping track of Boltzmann's constant. 
 * @param temp Temperature (in Kelvin).
 * @returns Resulting polymer configuration.  
 */
template <typename T>
PolymerConfiguration<T> generateNextKMer(const int K,
                                         const Ref<const Matrix<T, Dynamic, 3> >& melt_coords,  
                                         std::unordered_map<std::string, T>& lj_params,
                                         std::unordered_map<std::string, T>& fene_params,
                                         const AngleMode angle_mode,  
                                         std::unordered_map<std::string, T>& angle_params, 
                                         std::unordered_map<std::string, T>& dihedral_params,
                                         const T intra_collision_threshold,
                                         const T inter_collision_threshold,  
                                         const int max_tries_per_atom,
                                         const int max_tries_per_kmer, 
                                         const int max_tries_per_seed, 
                                         const int max_n_backtracks,
                                         boost::random::mt19937& rng,
                                         boost::random::uniform_01<>& uniform_dist,
                                         const T xmax, const T ymax, const T zmax,
                                         const Ref<const Matrix<T, Dynamic, 2> >& bond_length_cdf,  
                                         const Units units = Units::NANO,
                                         const T temp = 300)
{
    const T kT = (
        units == Units::MICRO ? static_cast<T>(1.380649e-8) * temp : 
        static_cast<T>(1.380649e-2) * temp
    );
    const T xlen = 2 * xmax; 
    const T ylen = 2 * ymax; 
    const T zlen = 2 * zmax;   

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

    // While we have not exhausted the number of attempts ...
    int n_tries = 0;  
    while (n_tries < max_tries_per_kmer)
    {
        bool collect_config = true; 

        // Define a collision function with every K-mer generated thus far
        auto collision_inter = [&melt_coords, &inter_collision_threshold, &xlen, &ylen, &zlen](const Ref<const Matrix<T, 3, 1> >& r) -> bool
        {
            // Get the periodic distance with every atom in every K-mer
            for (int i = 0; i < melt_coords.rows(); ++i)
            {
                Matrix<T, 3, 1> dist = periodicDistVec<T>(
                    r, melt_coords.row(i), xlen, ylen, zlen 
                );
                if (dist.norm() < inter_collision_threshold)
                    return true;
            }
            return false;  
        };

        // Generate a new random point that is far from every K-mer generated
        // thus far 
        Matrix<T, 3, 1> r0_new;
        int n_tries_per_seed = 0;
        while (n_tries_per_seed < max_tries_per_seed)
        {
            r0_new << -xmax + 2 * xmax * uniform_dist(rng), 
                      -ymax + 2 * ymax * uniform_dist(rng), 
                      -zmax + 2 * zmax * uniform_dist(rng); 
            if (!collision_inter(r0_new))
                break; 
            else 
                n_tries_per_seed++; 
        }
        // If the polymer could not be seeded, then start sampling from
        // scratch
        if (n_tries_per_seed >= max_tries_per_seed)
        {
            throw TooManySeedPositionsError(
                "Sampling procedure failed to seed polymer without collisions"
            ); 
        }

        // Generate a new K-mer, ensuring that each new atom does not collide
        // with the growing K-mer as well as the other K-mers
        //
        // Generate a PolymerConfiguration<T> instance with the first 2 atoms 
        Matrix<T, Dynamic, 3> coords(K, 3);
        T length = sampleFene<T>(rng, uniform_dist, bond_length_cdf);
        coords.row(0) = r0_new;
        coords.row(1) = r0_new + length * randomDir<T, 3>(rng, uniform_dist);  
        PolymerConfiguration<T> config(
            coords(Eigen::seqN(0, 2), Eigen::all), units, temp
        ); 

        // Define another collision function for the growing K-mer 
        auto collision_intra = [&config, &intra_collision_threshold, &xlen, &ylen, &zlen](const Ref<const Matrix<T, 3, 1> >& r) -> bool
        {
            // Get the periodic distance with every atom within the growing
            // K-mer (except for the atom to which it will be bonded)
            Matrix<T, Dynamic, 3> coords_ = config.getSegment(0, config.getLength() - 1); 
            for (int i = 0; i < coords_.rows(); ++i)
            {
                if (periodicDistVec<T>(r, coords_.row(i), xlen, ylen, zlen).norm() < intra_collision_threshold)
                    return true;
            }
            return false;
        };  

        // Add a 3rd atom ...
        //
        // Keep generating a new atom until no collision is detected 
        bool found_collision = true;
        Matrix<T, 3, 1> new_atom;  
        while (found_collision)
        {
            length = sampleFene<T>(rng, uniform_dist, bond_length_cdf); 
            T angle = sample_angle(rng); 
            new_atom = generateNextAtom<T>(
                coords.row(0), coords.row(1), length, angle, rng, uniform_dist
            );
            found_collision = collision_intra(new_atom);  
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
            int n_tries_per_atom = 0;
            found_collision = true; 
            while (found_collision && n_tries_per_atom < max_tries_per_atom)
            { 
                length = sampleFene<T>(rng, uniform_dist, bond_length_cdf); 
                T angle = sample_angle(rng);
                T dihedral = sampleDihedralHarmonic<T>(
                    dihedral_params["K"], kT, rng, uniform_dist
                );
                new_atom = generateNextAtomDihedral<T>(
                    r1, r2, r3, length, angle, dihedral, rng, uniform_dist 
                );
                found_collision = collision_intra(new_atom); 
                n_tries_per_atom++; 
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
                // Try re-seeding the polymer
                collect_config = false; 
                break;  
            }

            // If we have exceeded the maximum number of backtracks, try
            // re-seeding the polymer 
            if (n_backtracks > max_n_backtracks)
            {
                collect_config = false;
                break; 
            } 
        }
        
        // Test that the new K-mer does not collide with every K-mer 
        // generated thus far
        if (collect_config)
        {
            for (int j = 0; j < K; ++j)
            {
                if (collision_inter(coords.row(j)))
                {
                    collect_config = false; 
                    break;
                } 
            }
        }

        // If so, collect that K-mer 
        if (collect_config)
            return config;  
        else    // Otherwise, try sampling that K-mer again  
            n_tries++; 
    }

    throw PolymerGenerationError(
        "Sampling procedure failed to generate polymer within specified number "
        "of attempts"
    ); 
} 

/**
 * Generate a configuration of a melt of M K-mers, in which the inter-atom
 * distances, bond lengths, bond angles, and dihedral angles follow the given
 * potentials.
 *
 * @param K Polymer length.
 * @param M Number of polymers. 
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
 * @param intra_collision_threshold Distance threshold for identifying atoms
 *                                  that are too close to each other.
 * @param inter_collision_threshold 
 * @param max_tries_per_atom Maximum number of attempts to place each atom
 *                           before backtracking.
 * @param max_tries_per_kmer Maximum number of attempts to generate a full
 *                           K-mer. 
 * @param max_tries_per_seed Maximum number of attempts to seed a K-mer. 
 * @param max_n_backtracks Maximum number of backtracks.
 * @param max_n_restarts Maximum number of restarts. 
 * @param rng Random number generator. 
 * @param uniform_dist Pre-defined instance of standard uniform distribution.
 * @param xmax x-coordinate of first monomer of each chain is sampled from
 *             [-xmax, xmax], which is the x-range of the fundamental cell
 *             under periodic boundary conditions. 
 * @param ymax y-coordinate of first monomer of each chain is sampled from
 *             [-ymax, ymax], which is the y-range of the fundamental cell 
 *             under periodic boundary conditions. 
 * @param zmax z-coordinate of first monomer of each chain is sampled from
 *             [-zmax, zmax], which is the z-range of the fundamental cell
 *             under periodic boundary conditions. 
 * @param bond_length_cdf Pre-defined CDF for bond length distribution.  
 * @param units Units for keeping track of Boltzmann's constant. 
 * @param temp Temperature (in Kelvin).
 * @param verbose If true, print intermittent output to stdout.  
 * @returns Resulting polymer configuration.  
 */
template <typename T>
PolymerMeltConfiguration<T> generateKMerMelt(const int K, const int M, 
                                             std::unordered_map<std::string, T>& lj_params,
                                             std::unordered_map<std::string, T>& fene_params,
                                             const AngleMode angle_mode,  
                                             std::unordered_map<std::string, T>& angle_params, 
                                             std::unordered_map<std::string, T>& dihedral_params,
                                             const T intra_collision_threshold,
                                             const T inter_collision_threshold,  
                                             const int max_tries_per_atom,
                                             const int max_tries_per_kmer,
                                             const int max_tries_per_seed,  
                                             const int max_n_backtracks,
                                             const int max_n_restarts, 
                                             boost::random::mt19937& rng,
                                             boost::random::uniform_01<>& uniform_dist,
                                             const T xmax, const T ymax,
                                             const T zmax,
                                             const Ref<const Matrix<T, Dynamic, 2> >& bond_length_cdf,  
                                             const Units units = Units::NANO,
                                             const T temp = 300, 
                                             const bool verbose = false)
{
    // Collect polymer coordinates 
    Matrix<T, Dynamic, 3> coords_all;

    // While we have not exhausted the number of restarts ...
    int n_restarts = 0;
    int n_collected = 0;  
    while (n_restarts < max_n_restarts)
    {
        // Start with the very first K-mer ... 
        bool restart = false;
        PolymerConfiguration<T> init_config; 
        try
        {
            init_config = generateFirstKMer<T>(
                K, lj_params, fene_params, angle_mode, angle_params, dihedral_params,
                intra_collision_threshold, max_tries_per_atom, max_n_backtracks,
                rng, uniform_dist, xmax, ymax, zmax, bond_length_cdf, units, temp  
            );
        }
        catch (const TooManyBacktracksError& e)
        {
            // If an exception is raised, try sampling again from scratch,
            // since we are on the first polymer
            restart = true; 
            n_restarts++; 
            continue; 
        }
        
        // Otherwise, collect the coordinates for the first K-mer
        Matrix<T, Dynamic, 3> init_coords = init_config.getSegment(0, K); 
        coords_all = init_coords;
        n_collected++;  
        if (verbose)
            std::cout << "... generated chain 0\n"; 

        // Iteratively generate each subsequent K-mer ... 
        while (n_collected < M)
        {
            PolymerConfiguration<T> config;
            try
            {
                config = generateNextKMer<T>(
                    K, coords_all, lj_params, fene_params, angle_mode, angle_params,
                    dihedral_params, intra_collision_threshold,
                    inter_collision_threshold, max_tries_per_atom, 
                    max_tries_per_kmer, max_tries_per_seed, max_n_backtracks, 
                    rng, uniform_dist, xmax, ymax, zmax, bond_length_cdf, units, 
                    temp 
                );  
            }
            catch (const TooManySeedPositionsError& e)
            {
                restart = true;
                break;
            }
            catch (const PolymerGenerationError& e)
            {
                restart = true; 
                break; 
            }
            
            // Collect the K-mer
            Matrix<T, Dynamic, 3> coords = config.getSegment(0, K); 
            coords_all.conservativeResize((n_collected + 1) * K, 3);
            coords_all(Eigen::seqN(n_collected * K, K), Eigen::all) = coords;
            n_collected++; 
            if (verbose) 
                std::cout << "... generated chain " << n_collected - 1 << std::endl;  
        }

        // Restart sampling from scratch if we have exceeded the maximum number
        // of tries for a particular K-mer, or if restarting from scratch is
        // otherwise desired
        if (restart)
        {
            if (verbose)
                std::cout << "... restarting sampling from scratch\n";
            coords_all = Matrix<T, Dynamic, 3>::Zero(0, 3);  
            n_collected = 0; 
            n_restarts++;
        }
        else
        { 
            if (verbose)
                std::cout << "done with sampling!\n";
            break; 
        }
    }

    // If the desired number of K-mers could not be generated, raise an
    // exception
    if (n_collected < M)
    {
        throw std::runtime_error(
            "Sampling procedure exceeded maximum number of polymer generation "
            "attempts"
        );
    }

    // Generate the polymer melt 
    std::vector<Matrix<T, Dynamic, 3> > coords_all_vec;
    for (int i = 0; i < M; ++i) 
        coords_all_vec.push_back(coords_all(Eigen::seqN(i * K, K), Eigen::all)); 
    PolymerMeltConfiguration<T> melt_config(M, coords_all_vec, units, temp); 

    return melt_config;  
}

#endif
