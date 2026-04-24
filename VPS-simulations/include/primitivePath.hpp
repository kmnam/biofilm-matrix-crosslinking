/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     4/24/2026
 */
#ifndef PRIMITIVE_PATHS_HPP
#define PRIMITIVE_PATHS_HPP

#include <fstream>
#include <vector>
#include "polymerMelt.hpp"

/**
 * Strip leading and trailing whitespace from a string. 
 *
 * @param str Input string.
 * @returns Stripped string.  
 */
std::string stripWhitespace(const std::string& str)
{
    const int start = str.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos)
        return ""; 
    const int end = str.find_last_not_of(" \t\n\r\f\v");
    
    return str.substr(start, end - start + 1);  
}

/**
 * Minimal class for a collection of primitive paths. 
 */
template <typename T>
class PrimitivePathCollection
{
    private:
        int n_chains; 
        std::vector<int> lengths; 
        std::vector<Matrix<T, Dynamic, 3> > coords; 
        std::vector<Matrix<T, Dynamic, 1> > contour_indices; 
        std::vector<Matrix<int, Dynamic, 2> > partner_nodes;

        /**
         * Given a corresponding polymer melt configuration, find the
         * (approximate) bead position in the polymer corresponding to the
         * given node in the given primitive path.
         *
         * The node index may be a floating-point number, in which case the 
         * returned position is an interpolation of two neighboring bead 
         * positions. 
         *
         * The node index also ranges between 1 and N, where N is the chain
         * length. 
         */
        Matrix<T, 3, 1> getContourBeadPosition(PolymerMeltConfiguration<T>& melt_config,
                                               const int path_idx,
                                               const double node_idx)
        {
            const int node_floor = static_cast<int>(floor(node_idx - 1)); 
            const int node_ceil = static_cast<int>(ceil(node_idx - 1));
            const double delta = (node_idx - 1) - node_floor;  
            Matrix<T, 3, 1> p = melt_config.getSegment(path_idx, node_floor, 1); 
            Matrix<T, 3, 1> q = melt_config.getSegment(path_idx, node_ceil, 1);
            Matrix<T, 3, 1> r = (q - p) / (q - p).norm();
            return p + delta * r;  
        }

    public:
        PrimitivePathCollection(const int n_chains, std::vector<int>& lengths,
                                std::vector<Matrix<T, Dynamic, 3> >& coords,
                                std::vector<Matrix<T, Dynamic, 1> >& contour_indices,
                                std::vector<Matrix<int, Dynamic, 2> >& partner_nodes) 
        {
            this->n_chains = n_chains; 
            this->lengths = lengths; 
            this->coords = coords; 
            this->contour_indices = contour_indices; 
            this->partner_nodes = partner_nodes; 
        }

        /**
         * Return the approximate bead indices in the polymer corresponding 
         * to each node in the given primitive path.
         *
         * @param path_idx Input path index. 
         * @param zero_indexed If true, return 0-indexed bead indices. 
         * @returns The (floating-point) index of the nearest bead to each node
         *          in the primitive path. 
         */
        Matrix<T, Dynamic, 1> getContourIndices(const int path_idx,
                                                const bool zero_indexed = true)
        {
            if (zero_indexed)
                return (this->contour_indices[path_idx].array() - 1).matrix(); 
            else 
                return this->contour_indices[path_idx];  
        }

        /**
         * Return the *nearest* bead indices in the polymer corresponding to
         * each node in the given primitive path.
         *
         * @param path_idx Input path index. 
         * @param zero_indexed If true, return 0-indexed bead indices. 
         * @returns The index of the nearest bead to each node in the primitive
         *          path.  
         */
        Matrix<int, Dynamic, 1> getNearestContourIndices(const int path_idx,
                                                         const bool zero_indexed = true)
        {
            Matrix<int, Dynamic, 1> idx = this->contour_indices[path_idx].array().round().matrix().template cast<int>(); 
            if (zero_indexed)
                return (idx.array() - 1).matrix(); 
            else 
                return idx;  
        }

        /**
         * Given a corresponding polymer melt configuration, find the
         * (approximate) bead positions in the polymer corresponding to each
         * node in the given primitive path.
         *
         * Each returned position is a linear interpolation of the two 
         * bead positions on either side of each node's contour index. 
         *
         * @param melt_config Corresponding polymer melt configuration, with 
         *                    same number of chains and matching chain lengths.
         * @param path_idx Input path index. 
         * @returns Interpolated bead positions corresponding to the nodes 
         *          along the primitive path.  
         */
        Matrix<T, Dynamic, 3> getContourBeadPositions(PolymerMeltConfiguration<T>& melt_config, 
                                                      const int path_idx)
        {
            const int path_length = this->lengths[path_idx];
            Matrix<T, Dynamic, 3> bead_positions(path_length, 3);  
            for (int i = 0; i < path_length; ++i)
            {
                // Get the contour index of the i-th node 
                double contour_idx = this->contour_indices[path_idx](i);

                // Get the neighboring beads and interpolate between them 
                const int bead_floor = static_cast<int>(floor(contour_idx - 1)); 
                const int bead_ceil = static_cast<int>(ceil(contour_idx - 1));
                const double delta = (contour_idx - 1) - bead_floor;  
                Matrix<T, 3, 1> p = melt_config.getSegment(path_idx, bead_floor, 1); 
                Matrix<T, 3, 1> q = melt_config.getSegment(path_idx, bead_ceil, 1);
                Matrix<T, 3, 1> r = (q - p) / (q - p).norm();
                bead_positions.row(i) = p + delta * r;  
            }

            return bead_positions; 
        }
};

/**
 * Parse the given Z1+SP.dat file, containing the coordinates of primitive
 * paths in a collection of configurations.  
 *
 * @param filename Input filename. 
 * @returns Collection of primitive path collections. 
 */
template <typename T>
std::vector<PrimitivePathCollection<T> > parseZ1SPFile(const std::string& filename)
{
    std::vector<PrimitivePathCollection<T> > collections; 

    // Parse the input file ... 
    std::ifstream infile(filename); 
    std::string line, token;
    std::stringstream ss;  
    
    // The first line contains the number of chains
    std::getline(infile, line); 
    line = stripWhitespace(line);
    int n_paths = std::stoi(line); 

    // The second line contains the box dimensions 
    std::getline(infile, line); 
    line = stripWhitespace(line); 
    ss << line; 
    ss >> token;
    T xlen = static_cast<T>(std::stod(token)); 
    ss >> token; 
    T ylen = static_cast<T>(std::stod(token)); 
    ss >> token; 
    T zlen = static_cast<T>(std::stod(token));

    // The third line contains the length of the first path in the first
    // collection
    std::getline(infile, line); 
    line = stripWhitespace(line); 
    int curr_path_length = std::stoi(line); 

    // Set up a new list of primitive paths
    std::vector<Matrix<T, Dynamic, 3> > curr_coll_paths;
    std::vector<Matrix<T, Dynamic, 1> > curr_coll_contour_indices; 
    std::vector<Matrix<int, Dynamic, 2> > curr_coll_partner_nodes;
    Matrix<T, Dynamic, 3> curr_path = Matrix<T, Dynamic, 3>::Zero(curr_path_length, 3); 
    Matrix<T, Dynamic, 1> curr_contour_indices = Matrix<T, Dynamic, 1>::Zero(curr_path_length); 
    Matrix<int, Dynamic, 2> curr_partner_nodes = Matrix<int, Dynamic, 2>::Zero(curr_path_length, 2); 
    int curr_node_idx = 0;

    // Run through the rest of the file ... 
    while (std::getline(infile, line))
    {
        // Parse the entries in the line 
        line = stripWhitespace(line);
        std::vector<std::string> entries; 
        ss.str(std::string()); 
        ss.clear(); 
        ss << line; 
        while (ss >> token)
            entries.push_back(token);

        // If we have reached a new path or collection ... 
        if (entries.size() == 1)
        {
            // If we have reached a new path in the current collection ... 
            if (curr_coll_paths.size() < n_paths - 1 && curr_node_idx == curr_path_length)
            {
                // Collect the current path 
                curr_coll_paths.push_back(curr_path); 
                curr_coll_contour_indices.push_back(curr_contour_indices);
                curr_coll_partner_nodes.push_back(curr_partner_nodes); 

                // Set up the new path 
                curr_path_length = std::stoi(entries[0]);
                curr_node_idx = 0; 
                curr_path = Matrix<T, Dynamic, 3>::Zero(curr_path_length, 3); 
                curr_contour_indices = Matrix<T, Dynamic, 1>::Zero(curr_path_length); 
                curr_partner_nodes = Matrix<int, Dynamic, 2>::Zero(curr_path_length, 2);  
            }
            // If we have reached a new collection of paths ... 
            else if (curr_coll_paths.size() == n_paths - 1 && curr_node_idx == curr_path_length)
            {
                // Collect the last path in the current collection 
                curr_coll_paths.push_back(curr_path); 
                curr_coll_contour_indices.push_back(curr_contour_indices);
                curr_coll_partner_nodes.push_back(curr_partner_nodes); 

                // Instantiate the current collection
                std::vector<int> lengths; 
                for (int i = 0; i < curr_coll_paths.size(); ++i)
                    lengths.push_back(curr_coll_paths[i].rows());  
                PrimitivePathCollection<T> collection(
                    n_paths, lengths, curr_coll_paths, curr_coll_contour_indices,
                    curr_coll_partner_nodes  
                ); 
                collections.push_back(collection); 
                
                // Set up the next collection 
                n_paths = std::stoi(entries[0]);
                curr_coll_paths = std::vector<Matrix<T, Dynamic, 3> >(); 
                curr_coll_contour_indices = std::vector<Matrix<T, Dynamic, 1> >(); 
                curr_coll_partner_nodes = std::vector<Matrix<int, Dynamic, 2> >();

                // Skip over the next line, which contains the box bounds 
                std::getline(infile, line); 

                // Read the next line, which contains the length of the first path
                std::getline(infile, line); 
                line = stripWhitespace(line); 
                curr_path_length = std::stoi(line);
                curr_node_idx = 0; 
                curr_path = Matrix<T, Dynamic, 3>::Zero(curr_path_length, 3); 
                curr_contour_indices = Matrix<T, Dynamic, 1>::Zero(curr_path_length); 
                curr_partner_nodes = Matrix<int, Dynamic, 2>::Zero(curr_path_length, 2);  
            } 
        }
        else    // Otherwise, the line specifies a node in the current path 
        {
            T rx = static_cast<T>(std::stod(entries[0]));    // Node x-coordinate
            T ry = static_cast<T>(std::stod(entries[1]));    // Node y-coordinate
            T rz = static_cast<T>(std::stod(entries[2]));    // Node z-coordinate
            T contour_idx = static_cast<T>(std::stod(entries[3]));   // Contour index
            int partner_chain_idx = (std::stoi(entries[4]) == 0 ? -1 : std::stoi(entries[5])); 
            int partner_node_idx = (std::stoi(entries[4]) == 0 ? -1 : std::stoi(entries[6]));
            curr_path(curr_node_idx, 0) = rx; 
            curr_path(curr_node_idx, 1) = ry; 
            curr_path(curr_node_idx, 2) = rz; 
            curr_contour_indices(curr_node_idx) = contour_idx; 
            curr_partner_nodes(curr_node_idx, 0) = partner_chain_idx; 
            curr_partner_nodes(curr_node_idx, 1) = partner_node_idx; 
            curr_node_idx++;  
        } 
    }

    // Collect the very last path ... 
    curr_coll_paths.push_back(curr_path); 
    curr_coll_contour_indices.push_back(curr_contour_indices);
    curr_coll_partner_nodes.push_back(curr_partner_nodes); 
    
    // ... and the very last collection 
    std::vector<int> lengths; 
    for (int i = 0; i < curr_coll_paths.size(); ++i)
        lengths.push_back(curr_coll_paths[i].rows());  
    PrimitivePathCollection<T> collection(
        n_paths, lengths, curr_coll_paths, curr_coll_contour_indices,
        curr_coll_partner_nodes  
    ); 
    collections.push_back(collection); 

    return collections; 
}

#endif
