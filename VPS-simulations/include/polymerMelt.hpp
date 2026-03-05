/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     3/4/2026
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
            int n_atoms = 0;
            int n_bonds = 0; 
            int n_angles = 0; 
            int n_dihedrals = 0;  
            for (int i = 0; i < this->n; ++i)
            {
                n_atoms += this->lengths[i];
                n_bonds += (this->lengths[i] - 1);
                n_angles += (this->lengths[i] - 2); 
                n_dihedrals += (this->lengths[i] - 3); 
            } 
            outfile << n_atoms << " atoms\n"
                    << n_bonds << " bonds\n"
                    << n_angles << " angles\n"
                    << n_dihedrals << " dihedrals\n"
                    << "0 impropers\n\n"; 

            // Write numbers of atom, bond, angle, and dihedral types
            outfile << "1 atom types\n1 bond types\n1 angle types\n"
                    << "1 dihedral types\n0 improper types\n\n";

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

            // Write angle potential parameters
            if (angle_mode == AngleMode::COSINE)
            {
                outfile << "Angle Coeffs\n\n"
                        << "1 " << angle_params["K"] << " "
                        << 180 * angle_params["theta0"] / boost::math::constants::pi<T>() << "\n\n"; 
            }
            else if (angle_mode == AngleMode::GAUSSIAN)
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

            // Write dihedral potential parameters
            T dihedral_d, dihedral_n;
            if (dihedral_params.find("d") == dihedral_params.end())
                dihedral_d = 1; 
            else 
                dihedral_d = dihedral_params["d"]; 
            if (dihedral_params.find("n") == dihedral_params.end())
                dihedral_n = 1; 
            else 
                dihedral_n = dihedral_params["n"]; 
            outfile << "Dihedral Coeffs\n\n"
                    << "1 " << dihedral_params["K"] << " " 
                    << dihedral_d << " "
                    << dihedral_n << "\n\n";

            // Write atom coordinates 
            outfile << "Atoms\n\n";
            int offset = 0; 
            for (int i = 0; i < this->n; ++i)
            {
                const int ni = this->lengths[i]; 
                Matrix<T, Dynamic, 3> ri = this->configs[i].getSegment(0, ni);  
                for (int j = 0; j < ni; ++j)
                {
                    // Atom ID, molecule ID, atom type, x, y, z
                    //
                    // Atom and molecule IDs must be 1-indexed
                    int atom_id = offset + j + 1; 
                    int mol_id = i + 1; 
                    outfile << atom_id << " " << mol_id << " 1 " << ri(j, 0) << " "
                            << ri(j, 1) << " " << ri(j, 2) << std::endl; 
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

            // Write angles 
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

            // Write dihedrals 
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

/**
 * Generate a pair of random K-mer configurations, in which (1) the inter-atom
 * distances, bond lengths, bond angles, and dihedral angles follow the given
 * potentials, and (2) the centers of mass of the two K-mers is close to the
 * given distance.
 *
 * @param K Polymer length.
 * @param dist Distance between centers of mass of the two K-mers.  
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
 * @param collision_threshold Distance threshold for identifying atoms that 
 *                            are too close to each other. 
 * @param max_tries_per_atom Maximum number of attempts to place each atom
 *                           before backtracking. 
 * @param max_n_backtracks Maximum number of backtracks. 
 * @param rng Random number generator. 
 * @param uniform_dist Pre-defined instance of standard uniform distribution. 
 * @param units Units for keeping track of Boltzmann's constant. 
 * @param temp Temperature (in Kelvin). 
 * @returns Resulting polymer configuration.  
 */
template <typename T>
PolymerMeltConfiguration<T> generateKMerPair(const int K, const T dist, 
                                             const T center_dist_tol, 
                                             std::unordered_map<std::string, T>& lj_params,
                                             std::unordered_map<std::string, T>& fene_params,
                                             const AngleMode angle_mode,  
                                             std::unordered_map<std::string, T>& angle_params, 
                                             std::unordered_map<std::string, T>& dihedral_params,
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

    // Generate the first K-mer
    PolymerConfiguration<T> config1 = generateKMer<T>(
        K, lj_params, fene_params, angle_mode, angle_params, dihedral_params, 
        Matrix<T, 3, 1>::Zero(), collision_threshold, max_tries_per_atom, 
        max_n_backtracks, rng, uniform_dist, units, temp
    ); 
    Matrix<T, Dynamic, 3> coords1 = config1.getSegment(0, K);
    Matrix<T, 3, 1> center1 = config1.centerOfMass(); 

    // Generate the target center-of-mass for the second K-mer
    Matrix<T, 3, 1> dir = randomDir<T, 3>(rng, uniform_dist);
    Matrix<T, 3, 1> target_center = center1 + dist * dir;

    // Initialize coordinates for the second K-mer
    Matrix<T, Dynamic, 3> coords2(1, 3); 
    coords2.row(0) = target_center.transpose();
    PolymerConfiguration<T> config2(coords2, units, temp); 

    // Define a collision function with the first K-mer 
    auto collision1 = [&config1, &collision_threshold](const Ref<const Matrix<T, 3, 1> >& r) -> bool
    {
        return (config1.getMinDist(r) < collision_threshold);
    }; 

    // Define a collision function with the second K-mer 
    auto collision2 = [&collision_threshold](PolymerConfiguration<T> config2, const Ref<const Matrix<T, 3, 1> >& r) -> bool
    {
        return (config2.getMinDist(r) < collision_threshold); 
    };

    // Add the remaining atoms ... 
    T length, angle, dihedral;
    int n_backtracks = 0;
    while (config2.getLength() < K)
    {
        // If we have exceeded the maximum number of backtracks, raise 
        // an exception 
        if (n_backtracks > max_n_backtracks)
        {
            throw std::runtime_error(
                "Sampling procedure exceeded maximum number of backtracks; try "
                "sampling more positions per atom"
            );
        } 

        // Get the current polymer coordinates
        int n2 = config2.getLength(); 
        coords2 = config2.getSegment(0, n2);

        // First add an atom at the tail of the current segment
        //
        // If we are merely adding the second atom, just sample a bond length
        Matrix<T, 3, 1> new_atom;
        int n_tries = 0; 
        bool found_new_atom = false; 
        if (n2 == 1)
        {
            while (!found_new_atom && n_tries < max_tries_per_atom)
            {
                length = sampleFene<T>(
                    lj_params["eps"], lj_params["sigma"], fene_params["K"],
                    fene_params["R0"], config2.kT, rng, uniform_dist, 50 
                );
                dir = randomDir<T, 3>(rng, uniform_dist);  
                new_atom = coords2.row(n2 - 1) + length * dir.transpose();

                // Check if there is a collision
                if (!collision1(new_atom))
                {
                    // Check if the proposed new atom does not change the 
                    // center of mass within the given tolerance
                    Matrix<T, 3, 1> new_center = (coords2.row(0) + new_atom.transpose()) / 2; 
                    T curr_dist = (center1 - new_center).norm();  
                    if (abs(curr_dist - dist) < center_dist_tol)
                        found_new_atom = true;  
                }
                n_tries++; 
            } 
        }
        else    // coords2.rows() should be >= 3
        {
            while (!found_new_atom && n_tries < max_tries_per_atom) 
            {
                length = sampleFene<T>(
                    lj_params["eps"], lj_params["sigma"], fene_params["K"],
                    fene_params["R0"], config2.kT, rng, uniform_dist, 50 
                );
                angle = sample_angle(rng);
                dihedral = sampleDihedralHarmonic<T>(
                    dihedral_params["K"], config2.kT, rng, uniform_dist
                );
                Matrix<T, 3, 1> r1 = coords2.row(n2 - 3); 
                Matrix<T, 3, 1> r2 = coords2.row(n2 - 2); 
                Matrix<T, 3, 1> r3 = coords2.row(n2 - 1); 
                new_atom = generateNextAtomDihedral<T>(
                    r1, r2, r3, length, angle, dihedral, rng, uniform_dist
                );

                // Check if there is a collision
                if (!collision1(new_atom) && !collision2(config2, new_atom))
                {
                    // Check if the proposed new atom does not change the 
                    // center of mass within the given tolerance
                    Matrix<T, 3, 1> curr_center = coords2.colwise().mean();
                    Matrix<T, 3, 1> new_center = (n2 * curr_center + new_atom) / (n2 + 1);
                    T curr_dist = (center1 - new_center).norm(); 
                    if (abs(curr_dist - dist) < center_dist_tol)
                        found_new_atom = true;  
                }
                n_tries++; 
            }
        }

        // If a new atom was successfully found, move onto the next 
        if (found_new_atom)
        { 
            // Add the atom to the tail
            config2.appendAtomToTail(new_atom);
        }
        // Otherwise, backtrack to the previous atom unless doing so 
        // encroaches into the first 3 atoms 
        else if (n2 > 3) 
        {
            // Remove the atom at the tail
            config2.popAtomFromTail(); 
            n_backtracks++; 
            continue;  
        }
        else 
        {
            throw std::runtime_error(
                "Sampling procedure backtracked into first 3 atoms; try "
                "sampling more positions per atom"
            ); 
        }

        // Have we reached the desired polymer length?
        n2 = config2.getLength();
        coords2 = config2.getSegment(0, n2);
        if (n2 == K)
            break;  

        // Then add an atom at the head of the current segment 
        //
        // If we are merely adding the third atom, just sample a bond length 
        // and a bond angle 
        new_atom = Matrix<T, 3, 1>::Zero();
        n_tries = 0;
        found_new_atom = false; 
        if (n2 == 2)
        {
            while (!found_new_atom && n_tries < max_tries_per_atom)
            {
                length = sampleFene<T>(
                    lj_params["eps"], lj_params["sigma"], fene_params["K"],
                    fene_params["R0"], config2.kT, rng, uniform_dist, 50 
                );
                angle = sample_angle(rng);
                Matrix<T, 3, 1> r1 = coords2.row(1); 
                Matrix<T, 3, 1> r2 = coords2.row(0); 
                new_atom = generateNextAtom<T>(
                    r1, r2, length, angle, rng, uniform_dist
                );
                
                // Check if there is a collision
                if (!collision1(new_atom) && !collision2(config2, new_atom))
                {
                    // Check if the proposed new atom does not change the 
                    // center of mass within the given tolerance
                    Matrix<T, 3, 1> curr_center = coords2.colwise().mean();
                    Matrix<T, 3, 1> new_center = (2 * curr_center + new_atom) / 3;
                    T curr_dist = (center1 - new_center).norm(); 
                    if (abs(curr_dist - dist) < center_dist_tol)
                        found_new_atom = true; 
                }
                n_tries++;  
            } 
        }
        else    // coords2.rows() should be >= 4
        {
            while (!found_new_atom && n_tries < max_tries_per_atom)
            {
                length = sampleFene<T>(
                    lj_params["eps"], lj_params["sigma"], fene_params["K"],
                    fene_params["R0"], config2.kT, rng, uniform_dist, 50 
                );
                angle = sample_angle(rng);
                dihedral = sampleDihedralHarmonic<T>(
                    dihedral_params["K"], config2.kT, rng, uniform_dist
                );
                Matrix<T, 3, 1> r1 = coords2.row(2); 
                Matrix<T, 3, 1> r2 = coords2.row(1); 
                Matrix<T, 3, 1> r3 = coords2.row(0);
                new_atom = generateNextAtomDihedral<T>(
                    r1, r2, r3, length, angle, dihedral, rng, uniform_dist
                );

                // Check if there is a collision
                if (!collision1(new_atom) && !collision2(config2, new_atom))
                {
                    // Check if the proposed new atom does not change the 
                    // center of mass within the given tolerance
                    Matrix<T, 3, 1> curr_center = coords2.colwise().mean();
                    Matrix<T, 3, 1> new_center = (n2 * curr_center + new_atom) / (n2 + 1); 
                    T curr_dist = (center1 - new_center).norm(); 
                    if (abs(curr_dist - dist) < center_dist_tol)
                        found_new_atom = true; 
                }
                n_tries++; 
            }
        }

        // If a new atom was successfully found, move onto the next 
        if (found_new_atom)
        { 
            // Add the atom to the head 
            config2.appendAtomToHead(new_atom); 
        }
        // Otherwise, backtrack to the previous atom unless doing so 
        // encroaches into the first 3 atoms 
        else if (n2 > 3) 
        {
            // Remove the atom at the tail
            config2.popAtomFromHead(); 
            n_backtracks++; 
            continue;  
        }
        else 
        {
            throw std::runtime_error(
                "Sampling procedure backtracked into first 3 atoms; try "
                "sampling more positions per atom"
            ); 
        }

        n2 = config2.getLength();
        coords2 = config2.getSegment(0, n2);
    }
   
    coords2 = config2.getSegment(0, config2.getLength());
    std::vector<Matrix<T, Dynamic, 3> > coords_all; 
    coords_all.push_back(coords1); 
    coords_all.push_back(coords2); 
    PolymerMeltConfiguration<T> melt_config(2, coords_all, units, temp); 

    return melt_config;  
}

#endif
