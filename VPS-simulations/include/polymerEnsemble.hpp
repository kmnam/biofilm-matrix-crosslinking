/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     2/15/2026
 */

#ifndef POLYMER_ENSEMBLE_HPP
#define POLYMER_ENSEMBLE_HPP

#include <fstream>
#include <stdexcept>
#include <cmath>
#include <string>
#include <limits>
#include <unordered_map>
#include <functional>
#include <Eigen/Dense>
#include <boost/math/distributions/normal.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/random.hpp>
#include "utils.hpp"
#include "polymerConfiguration.hpp"

using std::pow; 
using boost::multiprecision::pow;

using namespace Eigen;

template <typename T>
using PolymerEnsemble = std::vector<PolymerConfiguration<T> >;

/**
 * Estimate the persistence length of the polymer from the given ensemble of
 * polymer configurations, using the tangent vector autocorrelation along 
 * each configuration. 
 *
 * @param ensemble Ensemble of polymer configurations. 
 * @returns Persistence length. 
 */
template <typename T>
T getPersistenceLength(PolymerEnsemble<T>& ensemble)
{
    // Check that there are at least two configurations 
    if (ensemble.size() < 2)
        throw std::runtime_error(
            "Invalid ensemble size for persistence length calculation"
        ); 

    // Get the tangent vectors along each configuration in the ensemble
    const int n = ensemble.size();  
    std::vector<Matrix<T, Dynamic, 3> > tangent_vectors; 
    for (int i = 0; i < n; ++i)
        tangent_vectors.push_back(ensemble[i].tangentVectors()); 

    // Get the mean bond length in each configuration
    T mean_bond_length = 0; 
    for (int i = 0; i < n; ++i)
    {
        T mean_i = ensemble[i].meanBondLength(); 
        mean_bond_length += ((mean_i - mean_bond_length) / (i + 1));
    } 

    // Get the 97.5-th percentile point of the standard normal 
    boost::math::normal normal_dist(0.0, 1.0); 
    const double z975 = quantile(normal_dist, 0.975); 

    // For each value of k ...
    Matrix<T, Dynamic, 1> autocorrs(0); 
    int k = 1;
    bool terminate = false;
    int n_noisy = 0;  
    while (!terminate)
    {
        // For each configuration ... 
        Matrix<T, Dynamic, 1> autocorrs_per_config_k
            = getTangentVectorAutocorrelation<T>(tangent_vectors, k);
        T autocorr_k = autocorrs_per_config_k.mean(); 

        // Keep track of the mean over all configurations for k 
        autocorrs.conservativeResize(k); 
        autocorrs(k - 1) = autocorr_k;

        // Calculate the standard error
        Array<T, Dynamic, 1> deviations = autocorrs_per_config_k.array() - autocorr_k; 
        T variance = deviations.pow(2).sum() / (n - 1);
        T std_error = sqrt(variance / n);

        // Check if the mean is statistically indistinguishable from zero 
        // using the Wald test
        //
        // More specifically, we check if the absolute value of the test
        // statistic, (mean - 0) / standard error, exceeds the Z-score for 
        // the 97.5-th percentile point of the standard normal 
        if (autocorr_k / std_error < z975)
            n_noisy++; 
        else     // If not, then reset n_noisy to zero 
            n_noisy = 0;

        // If we have reached 5 consecutive iterations where the mean is 
        // statistically indistiguishable from zero, then terminate  
        if (n_noisy >= 5)
            terminate = true;
        k++;
    }

    // Estimate the persistence length
    return mean_bond_length * (0.5 + autocorrs.sum()); 
}

/**
 * Parse the given file of polymer configurations. 
 *
 * @param filename Input filename.  
 * @param units Units for keeping track of Boltzmann's constant. 
 * @param temp Temperature (in Kelvin).
 * @returns Ensemble of polymer configurations in the given file, together with 
 *          the sampling parameters used to generate the ensemble.  
 */
template <typename T>
std::pair<PolymerEnsemble<T>, std::unordered_map<std::string, T> > parseEnsemble(const std::string& filename,
                                                                                 const Units units = Units::NANO, 
                                                                                 const T temp = 300.0)
{
    std::unordered_map<std::string, T> params; 

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
            if (token == "n_candidates")
                params["n_candidates"] = static_cast<T>(std::stoi(line));
            else if (token == "internal_move_tangent_stepsize")
                params["tangent_stepsize"] = static_cast<T>(std::stod(line));
            else if (token == "internal_move_generation_mode")
                params["mode"] = static_cast<T>(std::stoi(line));
            else if (token == "internal_move_n_attempts") 
                params["n_attempts"] = static_cast<T>(std::stoi(line));  
            else if (token == "internal_move_dx")
                params["dx"] = static_cast<T>(std::stod(line)); 
            else if (token == "internal_move_newton_tol")
                params["newton_tol"] = static_cast<T>(std::stod(line)); 
            else if (token == "internal_move_min_newton_stepsize")
                params["min_newton_stepsize"] = static_cast<T>(std::stod(line));
            else if (token == "internal_move_max_newton_iter")
                params["max_newton_iter"] = static_cast<T>(std::stoi(line)); 
            else if (token == "internal_move_armijo_const")
                params["armijo_const"] = static_cast<T>(std::stod(line));  
            else if (token == "move_prob_reptation")
                params["move_prob_reptation"] = static_cast<T>(std::stod(line));  
            else if (token == "move_prob_terminal_segment")
                params["move_prob_terminal_segment"] = static_cast<T>(std::stod(line)); 
            else if (token == "move_prob_internal_segment")
                params["move_prob_internal_segment"] = static_cast<T>(std::stod(line)); 
            else if (token == "terminal_segment_length")
                params["terminal_segment_length"] = static_cast<T>(std::stoi(line)); 
            else if (token == "internal_segment_length")
                params["internal_segment_length"] = static_cast<T>(std::stoi(line)); 
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

    // Now parse the first configuration, to get the polymer length
    int length = 0;
    while (std::getline(infile, line))
    {
        if (line.find("# CONFIG") == 0)   // If we reach the next configuration, break
            break;
        else
            length++; 
    }

    // Now parse the rest of the file to get the actual ensemble 
    PolymerEnsemble<T> ensemble;  
    Matrix<T, Dynamic, 3> coords = Matrix<T, Dynamic, 3>::Zero(length, 3);
    int curr_idx = 0; 
    while (std::getline(infile, line))
    {
        // If we reach a new configuration, keep parsing
        if (line.find("# CONFIG") == 0)
        {
            PolymerConfiguration<T> config(coords, units, temp); 
            ensemble.push_back(config); 
            curr_idx = 0;
            coords = Matrix<T, Dynamic, 3>::Zero(length, 3); 
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
            std::getline(ss, token, '\t');    // x-coordinate 
            coords(curr_idx, 0) = static_cast<T>(std::stod(token));
            std::getline(ss, token, '\t');    // y-coordinate
            coords(curr_idx, 1) = static_cast<T>(std::stod(token));
            std::getline(ss, token, '\t');    // z-coordinate
            coords(curr_idx, 2) = static_cast<T>(std::stod(token));
            curr_idx++; 
        }
    }

    return std::make_pair(ensemble, params); 
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
std::pair<PolymerConfiguration<T>, std::unordered_map<std::string, T> > parseFinalConfig(const std::string& filename,
                                                                                         const Units units = Units::NANO, 
                                                                                         const T temp = 300.0)
{
    std::unordered_map<std::string, T> params; 

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
            if (token == "n_candidates")
                params["n_candidates"] = static_cast<T>(std::stoi(line));
            else if (token == "internal_move_tangent_stepsize")
                params["tangent_stepsize"] = static_cast<T>(std::stod(line));
            else if (token == "internal_move_generation_mode")
                params["mode"] = static_cast<T>(std::stoi(line));
            else if (token == "internal_move_n_attempts") 
                params["n_attempts"] = static_cast<T>(std::stoi(line));  
            else if (token == "internal_move_dx")
                params["dx"] = static_cast<T>(std::stod(line)); 
            else if (token == "internal_move_newton_tol")
                params["newton_tol"] = static_cast<T>(std::stod(line)); 
            else if (token == "internal_move_min_newton_stepsize")
                params["min_newton_stepsize"] = static_cast<T>(std::stod(line));
            else if (token == "internal_move_max_newton_iter")
                params["max_newton_iter"] = static_cast<T>(std::stoi(line)); 
            else if (token == "internal_move_armijo_const")
                params["armijo_const"] = static_cast<T>(std::stod(line));  
            else if (token == "move_prob_reptation")
                params["move_prob_reptation"] = static_cast<T>(std::stod(line));  
            else if (token == "move_prob_terminal_segment")
                params["move_prob_terminal_segment"] = static_cast<T>(std::stod(line)); 
            else if (token == "move_prob_internal_segment")
                params["move_prob_internal_segment"] = static_cast<T>(std::stod(line)); 
            else if (token == "terminal_segment_length")
                params["terminal_segment_length"] = static_cast<T>(std::stoi(line)); 
            else if (token == "internal_segment_length")
                params["internal_segment_length"] = static_cast<T>(std::stoi(line)); 
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

    // Now parse the first configuration, to get the polymer length
    int length = 0;
    while (std::getline(infile, line))
    {
        if (line.find("# CONFIG") == 0)   // If we reach the next configuration, break
            break;
        else
            length++; 
    }

    // Now parse the rest of the file to get the final configuration 
    Matrix<T, Dynamic, 3> coords(length, 3);
    int curr_idx = 0; 
    while (std::getline(infile, line))
    {
        // If we reach a new configuration, keep parsing
        if (line.find("# CONFIG") == 0)
        {
            curr_idx = 0;
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
            std::getline(ss, token, '\t');    // x-coordinate 
            coords(curr_idx, 0) = static_cast<T>(std::stod(token));
            std::getline(ss, token, '\t');    // y-coordinate
            coords(curr_idx, 1) = static_cast<T>(std::stod(token));
            std::getline(ss, token, '\t');    // z-coordinate
            coords(curr_idx, 2) = static_cast<T>(std::stod(token));
            curr_idx++; 
        }
    }

    PolymerConfiguration<T> config(coords, units, temp); 
    return std::make_pair(config, params); 
}

/**
 * Parse the given file of polymer configurations, and add the configurations
 * to the given ensemble.
 *
 * It is assumed that the configurations in the new file are of the same-
 * length polymer as the configurations in the ensemble; if the ensemble is 
 * empty, then the length is inferred from the file. 
 *
 * The 0-th configuration in the file may be omitted. 
 *
 * @param filename Input filename. 
 * @param ensemble Input ensemble.
 * @param skip_config_0 Skip the 0-th configuration.
 * @param units Units for keeping track of Boltzmann's constant. Only used 
 *              if the given ensemble is empty.  
 * @param temp Temperature (in Kelvin). Only used if the given ensemble is 
 *             empty. 
 */
template <typename T>
void updateEnsemble(const std::string& filename, PolymerEnsemble<T>& ensemble,
                    const bool skip_config_0, Units units = Units::NANO, 
                    T temp = 300.0)
{
    // Parse the given file ...
    //
    // Skip to the initial configuraiton  
    std::ifstream infile(filename);  
    std::string line;
    while (std::getline(infile, line))
    {
        // Skip over all lines in the header (each starting with "##")
        if (line.find("##") != 0)
            break; 
    }

    // Now parse the first configuration, to get the polymer length if the 
    // ensemble is empty
    int length; 
    if (ensemble.size() == 0) 
    {
        length = 0;
        while (std::getline(infile, line))
        {
            if (line.find("# CONFIG") == 0)   // If we reach the next configuration, break
                break;
            else
                length++; 
        }
    }
    else 
    {
        length = ensemble[0].getLength();
        units = ensemble[0].getUnits(); 
        temp = ensemble[0].getTemp();  
    }

    // Skip to the first configuration to collect (either the 0-th or the 1st)
    if (skip_config_0)
    {
        while (std::getline(infile, line))
        {
            if (line.find("# CONFIG\t1") == 0)
                break; 
        }
    }  

    // Now parse the rest of the file to get the final configuration
    Matrix<T, Dynamic, 3> coords = Matrix<T, Dynamic, 3>::Zero(length, 3);
    int curr_idx = 0; 
    while (std::getline(infile, line))
    {
        // If we reach a new configuration, keep parsing
        if (line.find("# CONFIG") == 0)
        {
            PolymerConfiguration<T> config(coords, units, temp); 
            ensemble.push_back(config); 
            curr_idx = 0;
            coords = Matrix<T, Dynamic, 3>::Zero(length, 3); 
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
            std::getline(ss, token, '\t');    // x-coordinate 
            coords(curr_idx, 0) = static_cast<T>(std::stod(token));
            std::getline(ss, token, '\t');    // y-coordinate
            coords(curr_idx, 1) = static_cast<T>(std::stod(token));
            std::getline(ss, token, '\t');    // z-coordinate
            coords(curr_idx, 2) = static_cast<T>(std::stod(token));
            curr_idx++; 
        }
    }
}

#endif
