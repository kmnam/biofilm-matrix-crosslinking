/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     4/10/2026
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
using std::max; 
using boost::multiprecision::max; 
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
        T xmin; 
        T xmax; 
        T ymin; 
        T ymax; 
        T zmin; 
        T zmax;
        T xlen; 
        T ylen; 
        T zlen;  

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
            
            // Assume an infinitely large domain 
            this->xmin = -std::numeric_limits<T>::infinity();
            this->xmax = std::numeric_limits<T>::infinity(); 
            this->ymin = -std::numeric_limits<T>::infinity(); 
            this->ymax = std::numeric_limits<T>::infinity(); 
            this->zmin = -std::numeric_limits<T>::infinity(); 
            this->zmax = std::numeric_limits<T>::infinity();
            this->xlen = std::numeric_limits<T>::infinity();
            this->ylen = std::numeric_limits<T>::infinity();
            this->zlen = std::numeric_limits<T>::infinity();
        }

        /**
         * Default constructor.
         *
         * @param n Number of polymers. 
         * @param r Atomic coordinates for each polymer.  
         * @param units Units for keeping track of Boltzmann's constant. 
         * @param temp Temperature (in Kelvin).
         * @param xmin, xmax Domain limits along x-axis.
         * @param ymin, ymax Domain limits along y-axis. 
         * @param zmin, zmax Domain limits along z-axis.
         */
        PolymerMeltConfiguration(const int n,
                                 const std::vector<Matrix<T, Dynamic, 3> >& r, 
                                 const Units units, const T temp, 
                                 const T xmin, const T xmax, const T ymin, 
                                 const T ymax, const T zmin, const T zmax)
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

            // Set domain limits 
            this->xmin = xmin; 
            this->xmax = xmax; 
            this->ymin = ymin; 
            this->ymax = ymax; 
            this->zmin = zmin; 
            this->zmax = zmax; 
            this->xlen = this->xmax - this->xmin; 
            this->ylen = this->ymax - this->ymin; 
            this->zlen = this->zmax - this->zmin;  
        }

        /**
         * Trivial destructor. 
         */
        ~PolymerMeltConfiguration()
        {
        }

        /**
         * Return the number of polymers in the melt.
         *
         * @returns Number of polymers in the melt.  
         */
        int numPolymers() const 
        {
            return this->n; 
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
         * This is the minimum distance under periodic boundary conditions.  
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

            // Compute the minimum distance under periodic boundary conditions
            T mindist = std::numeric_limits<T>::infinity();
            const int ni = this->lengths[i]; 
            Matrix<T, Dynamic, 3> coords = this->configs[i].getSegment(0, ni); 
            for (int j = 0; j < this->lengths[i]; ++j)
            {
                Matrix<T, 3, 1> dvec = periodicDistVec<T>(
                    p, coords.row(j), this->xlen, this->ylen, this->zlen
                );
                T dist = dvec.norm();  
                if (mindist < dist)
                    mindist = dist; 
            }

            return mindist;
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
         * Replace the polymers in the melt. 
         *
         * @param polymers Arrays of atom coordinates. 
         */
        void setPolymers(std::vector<Matrix<T, Dynamic, 3> >& polymers)
        {
            this->n = polymers.size();
            this->lengths.clear(); 
            this->configs.clear(); 
            for (int i = 0; i < this->n; ++i)
            {
                this->lengths.push_back(polymers[i].rows());
                this->configs.push_back(polymers[i]);
            } 
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
         * The non-bonded interaction length scale is assumed to be smaller 
         * than the periodic domain, such that, for any atom in the polymer,
         * at most one copy of each atom in the melt interacts with it.
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

            // Start with the non-bonded energy of the i-th polymer by itself ... 
            T energy = 0;
            const int ni = this->lengths[i];
            Matrix<T, Dynamic, 3> ri = this->configs[i].getSegment(0, ni);  
            for (int j = 0; j < ni; ++j)
            {
                for (int k = j + 1; k < ni; ++k)
                {
                    if (!nonconsecutive || (nonconsecutive && abs(k - j) > 1))
                    {
                        Matrix<T, 3, 1> dvec = periodicDistVec<T>(
                            ri.row(j), ri.row(k), this->xlen, this->ylen, this->zlen
                        );
                        T dist = dvec.norm();
                        if (dist < neighbor_threshold) 
                            energy += lj<T>(
                                dist, lj_params["eps"], lj_params["sigma"], true
                            );
                    }
                }
            } 

            // Look for further non-bonded interactions with the other polymers
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
                            Matrix<T, 3, 1> dvec = periodicDistVec<T>(
                                rj.row(k), ri.row(p), this->xlen, this->ylen, 
                                this->zlen
                            ); 
                            T dist = dvec.norm(); 
                            if (dist < neighbor_threshold) 
                                energy += lj<T>(
                                    dist, lj_params["eps"], lj_params["sigma"],
                                    true
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
         * The non-bonded interaction length scale is assumed to be smaller 
         * than the periodic domain, such that, for any atom in the polymer,
         * at most one copy of each atom in the melt interacts with it.
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
            T energy = 0;
            for (int i = 0; i < this->n; ++i)
            {
                // Start with the non-bonded energy along the i-th polymer ...
                const int ni = this->lengths[i];
                Matrix<T, Dynamic, 3> ri = this->configs[i].getSegment(0, ni);  
                for (int j = 0; j < ni; ++j)
                {
                    for (int k = j + 1; k < ni; ++k)
                    {
                        if (!nonconsecutive || (nonconsecutive && abs(k - j) > 1))
                        {
                            Matrix<T, 3, 1> dvec = periodicDistVec<T>(
                                ri.row(j), ri.row(k), this->xlen, this->ylen,
                                this->zlen
                            );
                            T dist = dvec.norm();
                            if (dist < neighbor_threshold) 
                                energy += lj<T>(
                                    dist, lj_params["eps"], lj_params["sigma"],
                                    true
                                );
                        }
                    }
                } 
                
                // Then get the non-bonded energy between the i-th polymer 
                // and every other polymer ... 
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
                                Matrix<T, 3, 1> dvec = periodicDistVec<T>(
                                    rj.row(k), ri.row(p), this->xlen, this->ylen, 
                                    this->zlen
                                ); 
                                T dist = dvec.norm(); 
                                if (dist < neighbor_threshold) 
                                    energy += lj<T>(
                                        dist, lj_params["eps"], lj_params["sigma"],
                                        true
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
         * The non-bonded interaction length scale is assumed to be smaller 
         * than the periodic domain, such that, for any atom in the polymer,
         * at most one copy of each atom in the melt interacts with it.
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

            // Get each energy contribution
            T energy = this->getNonbondedEnergy(i, lj_params, neighbor_threshold, true);
            energy += this->getBondEnergy(i, fene_params, true, lj_params); 
            energy += this->getBondAngleEnergy(i, angle_mode, angle_params); 
            energy += this->getDihedralAngleEnergy(i, dihedral_params); 

            return energy;  
        }

        /**
         * Get the residual (non-bonded) energy of the proposed reptation move
         * for the i-th polymer. 
         *
         * This function calculates the total non-bonded energy between the
         * new atom and the other atoms that would remain in the i-th polymer
         * configuration after reptating in either direction, plus every atom
         * in every other polymer in the melt.  
         *
         * Note that the reptation direction does not matter for this
         * calculation.
         *
         * @param i Polymer index. 
         * @param r_new Position of new atom. 
         * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
         * @param neighbor_threshold Distance threshold for identifying
         *                           neighboring (non-bonded) atoms. 
         * @returns Residual energy of the proposed reptation move. 
         */
        T getReptationResidualEnergy(const int i,
                                     const Ref<const Matrix<T, 3, 1> >& r_new,
                                     std::unordered_map<std::string, T>& lj_params,  
                                     const T neighbor_threshold) const
        {
            // Check that i is a valid index
            if (i < 0 || i >= this->n)
                throw std::runtime_error("Undefined polymer index");

            // Get the residual energy within the i-th polymer ...
            //
            // Get the energy between the new atom and every other atom 
            // that would not be bonded to it after reptation
            //
            // The direction does not matter for this calculation 
            T energy = 0;
            const int ni = this->lengths[i]; 
            Matrix<T, Dynamic, 3> ri = this->configs[i].getSegment(0, ni); 
            for (int j = 1; j < ni - 1; ++j)    // Omit atoms 0 and (ni - 1)
            {
                // Use the periodic distance for this calculation  
                Matrix<T, 3, 1> dvec = periodicDistVec<T>(
                    ri.row(j), r_new, this->xlen, this->ylen, this->zlen
                ); 
                T dist = dvec.norm();
                if (dist < neighbor_threshold)
                    energy += lj<T>(
                        dist, lj_params["eps"], lj_params["sigma"], true
                    ); 
            }

            // Run through all the other polymers ...
            for (int j = 0; j < this->n; ++j)
            {
                if (i != j)
                {
                    // Get the non-bonded energy between the new atom and 
                    // every atom in the j-th polymer
                    const int nj = this->lengths[j];
                    Matrix<T, Dynamic, 3> rj = this->configs[j].getSegment(0, nj);
                    
                    // Use the periodic distance for this calculation  
                    for (int k = 0; k < nj; ++k)
                    {
                        Matrix<T, 3, 1> dvec = periodicDistVec<T>(
                            rj.row(k), r_new, this->xlen, this->ylen, this->zlen
                        ); 
                        T dist = dvec.norm(); 
                        if (dist < neighbor_threshold)
                            energy += lj<T>(
                                dist, lj_params["eps"], lj_params["sigma"], true
                            ); 
                    }
                }
            }

            // Return the total energy 
            return energy; 
        }

        /**
         * Get the residual (non-bonded) energy of the proposed position for 
         * the i-th atom (for some i = 0, ..., K - 1) in a multimer (K-atom)
         * reptation move for the m-th polymer.
         *
         * The positions of the previous atoms, j = 0, ..., i - 1, are also
         * given.
         *
         * This function calculates the total non-bonded energy between the
         * new atom and the other atoms that would remain in the m-th polymer
         * configuration, plus every atom in every other polymer in the melt.  
         *
         * @param m Polymer index. 
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
        T getMultimerReptationResidualEnergy(const int m,  
                                             const ReptationDirection direction,
                                             const int K, const int i, 
                                             const Ref<const Matrix<T, Dynamic, 3> >& segment,
                                             const Ref<const Matrix<T, 3, 1> >& r_new, 
                                             std::unordered_map<std::string, T>& lj_params,  
                                             const T neighbor_threshold) const
        {
            // Check that m is a valid index
            if (m < 0 || m >= this->n)
                throw std::runtime_error("Undefined polymer index");

            // Get the residual energy within the m-th polymer ...
            //
            // Get the energy between the new atom and every other atom 
            // that would not be bonded to it after reptation
            T energy = 0;
            const int nm = this->lengths[m]; 
            Matrix<T, Dynamic, 3> rm = this->configs[m].getSegment(0, nm); 
            if (direction == ReptationDirection::HEAD)
            {
                // Omit atoms nm - K, ..., nm - 1 within the current configuration
                //
                // If i == 0, then also omit atom 0, which would be bonded
                // to the new atom after reptation
                int min_idx = (i == 0 ? 1 : 0); 
                for (int j = min_idx; j < nm - K; ++j)
                {
                    // Use the periodic distance for this calculation 
                    Matrix<T, 3, 1> dvec = periodicDistVec<T>(
                        rm.row(j), r_new, this->xlen, this->ylen, this->zlen
                    );  
                    T dist = dvec.norm(); 
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
                    // Use the periodic distance for this calculation 
                    Matrix<T, 3, 1> dvec = periodicDistVec<T>(
                        segment.row(j), r_new, this->xlen, this->ylen, this->zlen
                    ); 
                    T dist = dvec.norm(); 
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
                // If i == 0, then also omit atom nm - 1, which would be bonded
                // to the new atom after reptation
                int max_idx = (i == 0 ? nm - 2 : nm - 1);    // Inclusive 
                for (int j = K; j <= max_idx; ++j)
                {
                    // Use the periodic distance for this calculation 
                    Matrix<T, 3, 1> dvec = periodicDistVec<T>(
                        rm.row(j), r_new, this->xlen, this->ylen, this->zlen
                    );  
                    T dist = dvec.norm(); 
                    if (dist < neighbor_threshold)
                        energy += lj<T>(
                            dist, lj_params["eps"], lj_params["sigma"], true
                        ); 
                }

                // Omit the final atom in the preceding segment
                //
                // Here, we assume that atom j == 0 is bonded to the (nm-1)-th 
                // atom in the current configuration, atom j == 1 is bonded
                // to atom j == 0, etc. 
                for (int j = 0; j < i - 1; ++j)
                {
                    // Use the periodic distance for this calculation 
                    Matrix<T, 3, 1> dvec = periodicDistVec<T>(
                        segment.row(j), r_new, this->xlen, this->ylen, this->zlen
                    );
                    T dist = dvec.norm();  
                    if (dist < neighbor_threshold)
                        energy += lj<T>(
                            dist, lj_params["eps"], lj_params["sigma"], true
                        ); 
                }
            }
            
            // Run through all the other polymers ... 
            for (int j = 0; j < this->n; ++j)
            {
                if (m != j)
                {
                    // Get the non-bonded energy between the new atom and 
                    // every atom in the j-th polymer
                    const int nj = this->lengths[j];
                    Matrix<T, Dynamic, 3> rj = this->configs[j].getSegment(0, nj);

                    // Use the periodic distance for this calculation  
                    for (int k = 0; k < nj; ++k)
                    {
                        Matrix<T, 3, 1> dvec = periodicDistVec<T>(
                            rj.row(k), r_new, this->xlen, this->ylen, this->zlen
                        ); 
                        T dist = dvec.norm(); 
                        if (dist < neighbor_threshold)
                            energy += lj<T>(
                                dist, lj_params["eps"], lj_params["sigma"], true
                            ); 
                    }
                }
            }

            // Return the total energy 
            return energy; 
        }

        /**
         * Get the residual (non-bonded) energy of the proposed position for 
         * the i-th atom (for some i = 0, ..., K - 1) in a K-atom terminal
         * segment move for the m-th polymer. 
         *
         * The positions of the previous atoms, j = 0, ..., i - 1, are also
         * given. 
         *
         * This function calculates the total non-bonded energy between the
         * new atom and the other atoms that would remain in the m-th polymer
         * configuration, plus every atom in every other polymer in the melt.  
         *
         * @param m Polymer index.
         * @param terminal_end Terminal segment to be moved. 
         * @param K Terminal segment length. 
         * @param i Index of the new atom in the terminal segment move. 
         * @param segment Atomic coordinates of the preceding segment of atoms,
         *                j = 0, ..., i - 1. Must have i rows. 
         * @param r_new Position of new (i-th) atom. 
         * @param lj_params Lennard-Jones/Weeks-Chandler-Andersen parameters. 
         * @param neighbor_threshold Distance threshold for identifying
         *                           neighboring (non-bonded) atoms. 
         * @returns Residual energy of the proposed terminal segment move. 
         */
        T getTerminalSegmentReplacementResidualEnergy(const int m,
                                                      const TerminalSegmentEnd terminal_end,
                                                      const int K, const int i,  
                                                      const Ref<const Matrix<T, Dynamic, 3> >& segment,
                                                      const Ref<const Matrix<T, 3, 1> >& r_new, 
                                                      std::unordered_map<std::string, T>& lj_params,  
                                                      const T neighbor_threshold) const 
        {
            // Check that i is a valid index
            if (m < 0 || m >= this->n)
                throw std::runtime_error("Undefined polymer index");

            // Get the residual energy within the m-th polymer ... 
            //
            // Get the energy between the new atom and every other atom 
            // that would not be bonded to it after the move
            T energy = 0;
            const int nm = this->lengths[m]; 
            Matrix<T, Dynamic, 3> rm = this->configs[m].getSegment(0, nm); 
            if (terminal_end == TerminalSegmentEnd::HEAD)
            {
                // Omit atoms 0, ..., K - 1 within the current configuration
                //
                // If i == 0 (which corresponds to the closest atom to the 
                // polymer, i.e., atom K - 1), then also omit atom K, which 
                // would be bonded to the new atom after the move 
                int min_idx = (i == 0 ? K + 1 : K); 
                for (int j = min_idx; j < nm; ++j)
                {
                    // Use the periodic distance for this calculation
                    Matrix<T, 3, 1> dvec = periodicDistVec<T>(
                        rm.row(j), r_new, this->xlen, this->ylen, this->zlen
                    ); 
                    T dist = dvec.norm(); 
                    if (dist < neighbor_threshold)
                        energy += lj<T>(
                            dist, lj_params["eps"], lj_params["sigma"], true
                        ); 
                }

                // Omit the final atom in the preceding segment
                //
                // Here, we assume that atom j == 0 is bonded to the K-th
                // atom in the current configuration, atom j == 1 is bonded 
                // to atom j == 0, etc. 
                for (int j = 0; j < i - 1; ++j)
                {
                    // Use the periodic distance for this calculation
                    Matrix<T, 3, 1> dvec = periodicDistVec<T>(
                        segment.row(j), r_new, this->xlen, this->ylen, this->zlen
                    ); 
                    T dist = dvec.norm(); 
                    if (dist < neighbor_threshold)
                        energy += lj<T>(
                            dist, lj_params["eps"], lj_params["sigma"], true
                        ); 
                }
            }
            else    // terminal_end == TerminalSegmentEnd::TAIL
            {
                // Omit atoms n - K, ..., n - 1 within the current configuration
                //
                // If i == 0, then also omit atom n - K - 1, which would be
                // bonded to the new atom after reptation
                int max_idx = (i == 0 ? nm - K - 2 : nm - K - 1);    // Inclusive 
                for (int j = 0; j <= max_idx; ++j)
                {
                    // Use the periodic distance for this calculation 
                    Matrix<T, 3, 1> dvec = periodicDistVec<T>(
                        rm.row(j), r_new, this->xlen, this->ylen, this->zlen
                    );  
                    T dist = dvec.norm(); 
                    if (dist < neighbor_threshold)
                        energy += lj<T>(
                            dist, lj_params["eps"], lj_params["sigma"], true
                        ); 
                }

                // Omit the final atom in the preceding segment
                //
                // Here, we assume that atom j == 0 is bonded to the (n-K-1)-th 
                // atom in the current configuration, atom j == 1 is bonded
                // to atom j == 0, etc. 
                for (int j = 0; j < i - 1; ++j)
                {
                    // Use the periodic distance for this calculation
                    Matrix<T, 3, 1> dvec = periodicDistVec<T>(
                        segment.row(j), r_new, this->xlen, this->ylen, this->zlen
                    ); 
                    T dist = dvec.norm(); 
                    if (dist < neighbor_threshold)
                        energy += lj<T>(
                            dist, lj_params["eps"], lj_params["sigma"], true
                        ); 
                }
            }

            // Run through all the remaining polymers ... 
            for (int j = 0; j < this->n; ++j)
            {
                if (m != j)
                {
                    // Get the non-bonded energy between the new atom and 
                    // every atom in the j-th polymer
                    const int nj = this->lengths[j];
                    Matrix<T, Dynamic, 3> rj = this->configs[j].getSegment(0, nj); 
                   
                    // Use the periodic distance for this calculation  
                    for (int k = 0; k < nj; ++k)
                    {
                        Matrix<T, 3, 1> dvec = periodicDistVec<T>(
                            rj.row(k), r_new, this->xlen, this->ylen, this->zlen
                        ); 
                        T dist = dvec.norm(); 
                        if (dist < neighbor_threshold)
                            energy += lj<T>(
                                dist, lj_params["eps"], lj_params["sigma"], true
                            );
                    }
                }
            }

            // Return the total energy 
            return energy; 
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
         * @param mass Atom mass.  
         */
        void writeLammps(const std::string& filename,
                         std::unordered_map<std::string, T>& lj_params, 
                         std::unordered_map<std::string, T>& fene_params, 
                         const AngleMode angle_mode,  
                         std::unordered_map<std::string, T>& angle_params, 
                         std::unordered_map<std::string, T>& dihedral_params, 
                         const std::string& header, const T mass)
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
            outfile << this->xmin << " " << this->xmax << " xlo xhi\n"
                    << this->ymin << " " << this->ymax << " ylo yhi\n"
                    << this->zmin << " " << this->zmax << " zlo zhi\n\n"; 

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
                        ri.row(j), this->xlen, this->ylen, this->zlen,
                        this->xmin, this->ymin, this->zmin
                    );
                    int image_x = static_cast<int>(
                        floor((ri(j, 0) - this->xmin) / this->xlen)
                    ); 
                    int image_y = static_cast<int>(
                        floor((ri(j, 1) - this->ymin) / this->ylen)
                    ); 
                    int image_z = static_cast<int>(
                        floor((ri(j, 2) - this->zmin) / this->zlen)
                    ); 
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
    // Keep parsing the file until we encounter the box bounds 
    std::stringstream ss; 
    std::string line, token;
    while (std::getline(infile, line))
    {
        if (line.compare(line.size() - 7, 7, "xlo xhi") == 0)
            break;
    }
    ss << line; 
    std::getline(ss, token, ' ');
    const T xmin = static_cast<T>(std::stod(token));
    std::getline(ss, token, ' '); 
    const T xmax = static_cast<T>(std::stod(token)); 
    std::getline(infile, line);     // y-bounds 
    ss.clear(); 
    ss.str(std::string()); 
    ss << line; 
    std::getline(ss, token, ' ');
    const T ymin = static_cast<T>(std::stod(token)); 
    std::getline(ss, token, ' '); 
    const T ymax = static_cast<T>(std::stod(token));
    std::getline(infile, line);     // z-bounds
    ss.clear(); 
    ss.str(std::string());
    ss << line; 
    std::getline(ss, token, ' ');
    const T zmin = static_cast<T>(std::stod(token));
    std::getline(ss, token, ' '); 
    const T zmax = static_cast<T>(std::stod(token));

    // Parse the Lennard-Jones parameters
    std::unordered_map<std::string, T> lj_params;
    std::getline(infile, line);    // Skip header and blank lines
    std::getline(infile, line);
    std::getline(infile, line);
    std::getline(infile, line);    // Potential parameters
    ss.clear();
    ss.str(std::string());
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
    std::getline(infile, line);    // Potential parameters
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
    std::getline(infile, line);    // Potential parameters
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
    std::getline(infile, line);    // Potential parameters
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
 * Parse the given .lammpstrj file and return the trajectory of polymer 
 * melt configurations.  
 *
 * @param filename Input filename.
 * @param dt Timestep. 
 * @param units Units used in the LAMMPS file. 
 * @param temp Temperature. 
 * @param tmin Minimum timepoint (in timesteps). 
 * @param tmax Maximum timepoint (in timesteps). 
 * @returns Polymer melt configurations in the given file.
 */
template <typename T>
std::pair<std::vector<PolymerMeltConfiguration<T> >,
          std::vector<T> > parseMeltLammpstrj(const std::string& filename,
                                              const T dt, const Units units, 
                                              const T temp, const int tmin = 0,
                                              const int tmax = std::numeric_limits<int>::infinity())
{
    // Stitch together the ensemble, one configuration at a time ... 
    std::vector<PolymerMeltConfiguration<T> > ensemble; 
    std::vector<T> times; 

    // Parse the given file, up to the first timestep ... 
    std::ifstream infile(filename); 
    std::string line;
    int n_chains = 0; 

    // Parse the first line, which marks the first timestep 
    std::getline(infile, line);
    std::getline(infile, line);    // Timepoint
    std::getline(infile, line);
    std::getline(infile, line);    // Number of atoms
    const int n_atoms = std::stoi(line); 

    // Parse the x-, y-, and z-bounds 
    std::getline(infile, line);
    std::getline(infile, line);    // x-bounds
    std::stringstream ss;
    ss << line; 
    std::string token; 
    std::getline(ss, token, ' '); 
    const T xmin = static_cast<T>(std::stod(token));  
    std::getline(ss, token, ' '); 
    const T xmax = static_cast<T>(std::stod(token));  
    std::getline(infile, line);    // y-bounds
    ss.clear(); 
    ss.str(std::string()); 
    ss << line; 
    std::getline(ss, token, ' '); 
    const T ymin = static_cast<T>(std::stod(token)); 
    std::getline(ss, token, ' '); 
    const T ymax = static_cast<T>(std::stod(token)); 
    std::getline(infile, line);    // z-bounds 
    ss.clear(); 
    ss.str(std::string()); 
    ss << line; 
    std::getline(ss, token, ' '); 
    const T zmin = static_cast<T>(std::stod(token)); 
    std::getline(ss, token, ' '); 
    const T zmax = static_cast<T>(std::stod(token)); 
    std::getline(infile, line);

    // Parse each atom in the system  
    while (std::getline(infile, line))
    {
        // If we have reached a new timestep, break
        if (line.find("ITEM: TIMESTEP") == 0)
            break; 

        // Otherwise, parse the line
        std::stringstream ss; 
        std::string token;
        ss << line;
        std::getline(ss, token, ' ');    // Atom ID 
        std::getline(ss, token, ' ');    // Molecule ID
        int mol_id = std::stoi(token);
        n_chains = max(n_chains, mol_id);
    }
    const int length = n_atoms / n_chains; 
    infile.close();  

    // Re-parse the given file, this time to the end ...
    Matrix<T, Dynamic, Dynamic> coords(n_chains, 3 * length);  
    infile.open(filename); 
    while (std::getline(infile, line))
    {
        // If we have arrived at a new timestep ... 
        if (line.find("ITEM: TIMESTEP") == 0)
        {
            // Read the next line to get the timepoint
            std::getline(infile, line);
            double t_curr = std::stod(line) * dt; 
            times.push_back(t_curr); 

            // Read the next two lines to get the total number of atoms (which
            // should be the same throughout the file)
            std::getline(infile, line); 
            std::getline(infile, line); 

            // Skip over the next four lines, which give the box bounds 
            std::getline(infile, line); 
            std::getline(infile, line);  
            std::getline(infile, line);  
            std::getline(infile, line);

            // Then parse the atom coordinates
            std::getline(infile, line);    // Header line 
            for (int i = 0; i < n_chains * length; ++i)
            {
                // Parse the line 
                std::getline(infile, line); 
                std::stringstream ss; 
                std::string token;
                ss << line; 

                // Atom index
                std::getline(ss, token, ' ');
                int atom_idx = std::stoi(token) - 1; 

                // Molecule index 
                std::getline(ss, token, ' ');
                int mol_idx = std::stoi(token) - 1;

                // Skip over the next token (atom type) 
                std::getline(ss, token, ' ');

                // x-, y-, and z-coordinates 
                std::getline(ss, token, ' ');
                double rx = std::stod(token);
                std::getline(ss, token, ' '); 
                double ry = std::stod(token);
                std::getline(ss, token, ' ');
                double rz = std::stod(token); 

                // Collect the coordinates 
                int atom_idx_within_mol = atom_idx % length;
                coords(mol_idx, 3 * atom_idx_within_mol) = rx; 
                coords(mol_idx, 3 * atom_idx_within_mol + 1) = ry;
                coords(mol_idx, 3 * atom_idx_within_mol + 2) = rz; 

                // Skip over the remaining tokens 
            }

            // Collect the polymer configuration
            std::vector<Matrix<T, Dynamic, 3> > coords_vec; 
            for (int i = 0; i < n_chains; ++i)
            {
                Matrix<T, Dynamic, 3> coords_i(length, 3); 
                for (int j = 0; j < length; ++j)
                    coords_i.row(j) = coords(i, Eigen::seqN(3 * j, 3)); 
                coords_vec.push_back(coords_i); 
            } 
            ensemble.emplace_back(
                PolymerMeltConfiguration<T>(
                    n_chains, coords_vec, units, temp, xmin, xmax, ymin, ymax,
                    zmin, zmax
                )
            ); 
        }
    } 

    return std::pair(ensemble, times); 
}

/**
 * Parse the final configuration in the given file of polymer configurations. 
 *
 * @param filename Input filename.  
 * @param units Units for keeping track of Boltzmann's constant. 
 * @param temp Temperature (in Kelvin).
 * @returns Final polymer configuration in the given file, together with 
 *          the sampling parameters used to generate the ensemble.  
 */
template <typename T>
std::pair<PolymerMeltConfiguration<T>,
          std::unordered_map<std::string, T> > parseMeltFinalConfig(const std::string& filename,
                                                                    const Units units = Units::NANO, 
                                                                    const T temp = 300.0)
{
    std::unordered_map<std::string, T> params;
    T xmin, xmax, ymin, ymax, zmin, zmax;

    // Parse the given file ... 
    //
    // First, parse the sampling parameters
    std::ifstream infile(filename);  
    std::string line;
    while (std::getline(infile, line))
    {
        // If the line starts with "##", then parse
        if (line.find("##") == 0)
        {
            std::string token = line.substr(3, line.find(" = ") - 3);   // Remove leading "## "
            line.erase(0, line.find(" = ") + 3);
            if (token == "domain_xmin")
                xmin = static_cast<T>(std::stod(line));
            else if (token == "domain_xmax")
                xmax = static_cast<T>(std::stod(line));
            else if (token == "domain_ymin")
                ymin = static_cast<T>(std::stod(line));
            else if (token == "domain_ymax")
                ymax = static_cast<T>(std::stod(line));
            else if (token == "domain_zmin")
                zmin = static_cast<T>(std::stod(line));
            else if (token == "domain_zmax")
                zmax = static_cast<T>(std::stod(line)); 
            else if (token == "n_candidates")
                params["n_candidates"] = static_cast<T>(std::stoi(line));
            else if (token == "move_prob_reptation")
                params["move_prob_reptation"] = static_cast<T>(std::stod(line));  
            else if (token == "move_prob_multimer_reptation")
                params["move_prob_multimer_reptation"] = static_cast<T>(std::stod(line)); 
            else if (token == "move_prob_terminal_segment")
                params["move_prob_terminal_segment"] = static_cast<T>(std::stod(line)); 
            else if (token == "multimer_reptation_length")
                params["multimer_reptation_length"] = static_cast<T>(std::stoi(line));  
            else if (token == "terminal_segment_length")
                params["terminal_segment_length"] = static_cast<T>(std::stoi(line)); 
            else if (token == "n_bins_fene_cdf")
                params["n_bins_fene_cdf"] = static_cast<T>(std::stoi(line));  
            else if (token == "mod_collect")
                params["mod_collect"] = static_cast<T>(std::stoi(line));  
            else if (token == "mod_write")
                params["mod_write"] = static_cast<T>(std::stoi(line)); 
            else if (token == "max_stall")
                params["max_stall"] = static_cast<T>(std::stoi(line));
        }
        // If not, then we have encountered the first configuration,
        // so we must break 
        else 
        {
            break; 
        }
    }

    // Now parse the first configuration, to get the polymer lengths
    std::vector<int> lengths;
    int curr_length = 0;
    int curr_idx = 0; 
    while (std::getline(infile, line))
    {
        if (line.find("# CONFIG") == 0)   // If we reach the next configuration, break
        {
            break;
        }
        else
        {
            // Check the index of the polymer 
            std::stringstream ss; 
            ss << line; 
            std::string token; 
            std::getline(ss, token, '\t');    // Polymer index 
            int polymer_idx = std::stoi(token); 
            std::getline(ss, token, '\t');    // Atom index 
            int atom_idx = std::stoi(token); 
            if (polymer_idx > curr_idx)       // Polymer index should be one greater at most
            {
                lengths.push_back(curr_length); 
                curr_length = 0; 
                curr_idx = polymer_idx; 
            }
            else 
            {
                curr_length++; 
            } 
        } 
    }
    lengths.push_back(curr_length);    // Add the last polymer length
    int n_chains = lengths.size(); 

    // Now parse the rest of the file to get the final configuration 
    std::vector<std::vector<Matrix<T, Dynamic, 3> > > melt_coords;
    for (int i = 0; i < lengths.size(); ++i)
    {
        std::vector<Matrix<T, Dynamic, 3> > melt_coords_i;
        for (int j = 0; j < lengths[i]; ++i) 
            melt_coords_i.push_back(Matrix<T, Dynamic, 3>::Zero(lengths[i], 3));
        melt_coords.push_back(melt_coords_i); 
    } 
    int config_idx = 0; 
    while (std::getline(infile, line))
    {
        // If we reach a new configuration, keep parsing
        if (line.find("# CONFIG") == 0)
        {
            config_idx++; 
        }
        // If we reach an ensemble-level output line at the end of
        // the file, stop parsing
        else if (line.find("##" ) == 0)
        {
            break; 
        }
        // If we reach a configuration-level output line, keep parsing
        else if (line.find("# ") == 0)
        {
            // Do nothing
        }
        // Otherwise, the line specifies coordinates that should be
        // collected 
        else 
        {
            std::stringstream ss; 
            ss << line;
            std::string token;
            std::getline(ss, token, '\t');    // Polymer index
            int polymer_idx = std::stoi(token); 
            std::getline(ss, token, '\t');    // Atom index
            int atom_idx = std::stoi(token); 
            std::getline(ss, token, '\t');    // x-coordinate 
            melt_coords[config_idx][polymer_idx](atom_idx, 0) = static_cast<T>(std::stod(token));
            std::getline(ss, token, '\t');    // y-coordinate
            melt_coords[config_idx][polymer_idx](atom_idx, 1) = static_cast<T>(std::stod(token));
            std::getline(ss, token, '\t');    // z-coordinate
            melt_coords[config_idx][polymer_idx](atom_idx, 2) = static_cast<T>(std::stod(token));
        }
    }

    PolymerMeltConfiguration<T> config(
        n_chains, melt_coords[config_idx], units, temp, xmin, xmax, ymin, ymax,
        zmin, zmax
    ); 
    return std::make_pair(config, params); 
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
    PolymerMeltConfiguration<T> melt_config(
        M, coords_all_vec, units, temp, -xmax, xmax, -ymax, ymax, -zmax, zmax
    ); 

    return melt_config;  
}

#endif
