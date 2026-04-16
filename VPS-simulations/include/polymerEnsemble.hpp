/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     4/15/2026
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

using std::isinf; 
using boost::multiprecision::isinf; 
using std::isnan; 
using boost::multiprecision::isnan; 
using std::pow; 
using boost::multiprecision::pow;

using namespace Eigen;

template <typename T>
using PolymerEnsemble = std::vector<PolymerConfiguration<T> >;

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
    auto result = parseEnsemble<T>(filename, units, temp);
    PolymerConfiguration<T> config = result.first[result.first.size() - 1];  
    return std::make_pair(config, result.second); 
}

/**
 * Parse the configurations in the given .lammpstrj file.
 *
 * The .lammpstrj file is assumed to contain the coordinates of one polymer
 * molecule. 
 *
 * @param filename Input filename. 
 * @param dt Timestep. 
 * @param units Units for keeping track of Boltzmann's constant.
 * @param temp Temperature (in Kelvin). 
 * @returns Ensemble of polymer configurations in the given file.
 */
template <typename T>
std::pair<PolymerEnsemble<T>, std::vector<T> > parseLammpstrj(const std::string& filename,
                                                              const T dt, 
                                                              const Units units = Units::NANO, 
                                                              const T temp = 300.0)
{
    // Stitch together the ensemble, one configuration at a time ... 
    PolymerEnsemble<T> ensemble;
    std::vector<T> times; 

    // Parse the given file ... 
    std::ifstream infile(filename); 
    std::string line; 
    while (std::getline(infile, line))
    {
        // If we have arrived at a new timestep ... 
        if (line.find("ITEM: TIMESTEP") == 0)
        {
            // Read the next line to get the timepoint
            std::getline(infile, line);
            times.push_back(std::stod(line) * dt);

            // Read the next two lines to get the polymer length (which 
            // should be the same throughout the file)
            std::getline(infile, line); 
            std::getline(infile, line); 
            int length = std::stoi(line); 

            // Skip over the next four lines, which give the box bounds 
            std::getline(infile, line); 
            std::getline(infile, line);  
            std::getline(infile, line);  
            std::getline(infile, line);

            // Then parse the atom coordinates
            std::getline(infile, line);    // Header line 
            Matrix<T, Dynamic, 3> coords(length, 3); 
            for (int i = 0; i < length; ++i)
            {
                // Parse the line 
                std::getline(infile, line); 
                std::stringstream ss; 
                std::string token;
                ss << line; 

                // Atom index
                std::getline(ss, token, ' '); 
                int idx = std::stoi(token) - 1; 

                // Skip over the next two tokens 
                std::getline(ss, token, ' '); 
                std::getline(ss, token, ' '); 

                // x-, y-, and z-coordinates 
                std::getline(ss, token, ' '); 
                coords(idx, 0) = std::stod(token); 
                std::getline(ss, token, ' '); 
                coords(idx, 1) = std::stod(token); 
                std::getline(ss, token, ' ');
                coords(idx, 2) = std::stod(token);

                // Skip over the remaining tokens 
            }

            // Collect the polymer configuration 
            ensemble.emplace_back(PolymerConfiguration<T>(coords, units, temp)); 
        }
    } 

    return std::pair(ensemble, times); 
}

/**
 * Write a new configurations file with the given configurations. 
 *
 * The .lammpstrj file is assumed to contain the coordinates of one polymer
 * molecule.
 *
 * The timesteps are set to 0, 1, 2, ... by convention.  
 *
 * @param ensemble Ensemble of polymer configurations. 
 * @param outfilename Output filename. 
 */
template <typename T>
void writeConfigFile(PolymerEnsemble<T>& ensemble,
                     std::unordered_map<std::string, T>& params,
                     std::unordered_map<std::string, T>& lj_params, 
                     std::unordered_map<std::string, T>& fene_params, 
                     const AngleMode angle_mode,
                     std::unordered_map<std::string, T>& angle_params, 
                     std::unordered_map<std::string, T>& dihedral_params, 
                     const std::string& outfilename)
{
    const double kT = 1.380649e-2 * 300;    // Use "nano" units 
    const double neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"]; 

    // Write sampling parameters to file
    std::ofstream outfile(outfilename);  
    outfile << std::setprecision(10);
    if (params.find("n_candidates") != params.end())
    { 
        outfile << "## n_candidates = "
                << static_cast<int>(params["n_candidates"]) << std::endl;
    }
    else 
    {
        outfile << "## n_candidates = NA\n"; 
    }
    if (params.find("move_prob_reptation") != params.end())
    {
        outfile << "## move_prob_reptation = "
                << params["move_prob_reptation"] << std::endl; 
    }
    else 
    {
        outfile << "## move_prob_reptation = NA\n"; 
    }
    if (params.find("move_prob_multimer_reptation") != params.end())
    {
        outfile << "## move_prob_multimer_reptation = "
                << params["move_prob_multimer_reptation"] << std::endl;
    }
    else 
    {
        outfile << "## move_prob_multimer_reptation = NA\n"; 
    }
    if (params.find("move_prob_terminal_segment") != params.end())
    {
        outfile << "## move_prob_terminal_segment = "
                << params["move_prob_terminal_segment"] << std::endl;
    }
    else 
    {
        outfile << "## move_prob_terminal_segment = NA\n";
    }
    if (params.find("multimer_reptation_length") != params.end())
    {
        outfile << "## multimer_reptation_length = "
                << static_cast<int>(params["multimer_reptation_length"])
                << std::endl;
    }
    else 
    {
        outfile << "## multimer_reptation_length = NA\n";
    }
    if (params.find("terminal_segment_length") != params.end())
    {
        outfile << "## terminal_segment_length = "
                << static_cast<int>(params["terminal_segment_length"])
                << std::endl;
    }
    else 
    {
        outfile << "## terminal_segment_length = NA\n";
    }
    if (params.find("n_bins_fene_cdf") != params.end())
    {
        outfile << "## n_bins_fene_cdf = "
                << static_cast<int>(params["n_bins_fene_cdf"]) << std::endl;
    }
    else 
    {
        outfile << "## n_bins_fene_cdf = NA\n";
    }
    if (params.find("max_iter") != params.end())
    {
        outfile << "## max_iter = "
                << static_cast<int>(params["max_iter"]) << std::endl;
    }
    else 
    {
        outfile << "## max_iter = NA\n";
    }
    if (params.find("mod_collect") != params.end())
    {
        outfile << "## mod_collect = "
                << static_cast<int>(params["mod_collect"]) << std::endl;
    }
    else 
    {
        outfile << "## mod_collect = NA\n";
    }
    if (params.find("mod_write") != params.end())
    {
        outfile << "## mod_write = "
                << static_cast<int>(params["mod_write"]) << std::endl;
    }
    else 
    {
        outfile << "## mod_write = NA\n";
    }
    if (params.find("max_stall") != params.end())
    {
        outfile << "## max_stall = "
                << static_cast<int>(params["max_stall"]) << std::endl;
    }
    else 
    {
        outfile << "## max_stall = NA\n"; 
    }

    // Write the coordinates to file
    //
    // Note that this file will lack an initial configuration
    for (int i = 0; i < ensemble.size(); ++i)
    {
        const int length = ensemble[i].getLength();
        Matrix<T, Dynamic, 3> coords = ensemble[i].getSegment(0, length);

        // Exclude LJ terms between consecutive atoms from the non-bonded energy
        T energy_nonbonded = ensemble[i].getNonbondedEnergy(
            lj_params, neighbor_threshold, true
        );

        // Include LJ terms between consecutive atoms in the bond energy
        T energy_bond = ensemble[i].getBondEnergy(fene_params, true, lj_params);

        // Calculate bond angle and dihedral energies 
        T energy_angle = ensemble[i].getBondAngleEnergy(angle_mode, angle_params); 
        T energy_dihedral = ensemble[i].getDihedralAngleEnergy(dihedral_params); 

        // Re-calculate the total energy 
        T energy_total = ensemble[i].getTotalEnergy(
            lj_params, neighbor_threshold, fene_params, angle_mode, angle_params,
            dihedral_params
        );
        if (isnan(energy_total) || isinf(energy_total))
        {
            std::stringstream ss; 
            ss << "Found configuration " << "(" << i
               << ") with NaN or infinite energy\n"
               << "- Nonbonded energy: " << energy_nonbonded << std::endl
               << "- Bond energy: " << energy_bond << std::endl
               << "- Angle energy: " << energy_angle << std::endl
               << "- Dihedral energy: " << energy_dihedral << std::endl
               << "- Total energy: " << energy_total << std::endl;
            throw std::runtime_error(ss.str()); 
        } 

        // Calculate the radius of gyration  
        T radius = ensemble[i].radiusOfGyration(); 

        // Write annotations and coordinates 
        outfile << "# CONFIG\t" << i << std::endl
                << "# ENERGY_TOTAL\t" << energy_total << std::endl
                << "# ENERGY_NONBONDED\t" << energy_nonbonded << std::endl
                << "# ENERGY_BOND\t" << energy_bond << std::endl
                << "# ENERGY_ANGLE\t" << energy_angle << std::endl
                << "# ENERGY_DIHEDRAL\t" << energy_dihedral << std::endl 
                << "# RADIUS_OF_GYRATION\t" << radius << std::endl; 
        for (int j = 0; j < length; ++i)
        { 
            outfile << coords(j, 0) << '\t' << coords(j, 1) << '\t'
                    << coords(j, 2) << std::endl; 
        }
    }
    outfile.close(); 
}

/**
 * Write a new .lammpstrj file with the given configurations.
 *
 * The .lammpstrj file is assumed to contain the coordinates of one polymer
 * molecule.
 *
 * The timesteps are set to 0, 1, 2, ... by convention.  
 *
 * @param ensemble Ensemble of polymer configurations. 
 * @param outfilename Output filename. 
 */
template <typename T>
void writeLammpstrj(PolymerEnsemble<T>& ensemble, const std::string& outfilename,
                    const T xmin, const T xmax, const T ymin, const T ymax, 
                    const T zmin, const T zmax)
{
    std::ofstream outfile(outfilename);

    // For each configuration ... 
    for (int i = 0; i < ensemble.size(); ++i)
    {
        // Write two lines for the timestep 
        outfile << "ITEM: TIMESTEP\n"
                << i << std::endl; 

        // Write two lines for the polymer length
        const int length = ensemble[i].getLength();  
        outfile << "ITEM: NUMBER OF ATOMS\n"
                << length << std::endl;

        // Write four lines for the box bounds (assume periodic boundary conditions)
        outfile << "ITEM: BOX BOUNDS pp pp pp\n"
                << xmin << " " << xmax << std::endl
                << ymin << " " << ymax << std::endl
                << zmin << " " << zmax << std::endl; 

        // Write the coordinates of each atom in the polymer 
        outfile << "ITEM: ATOMS id mol type xu yu zu\n";
        Matrix<T, Dynamic, 3> coords = ensemble[i].getSegment(0, length); 
        for (int j = 0; j < length; ++j)
        {
            outfile << j + 1 << " 1 1 " << coords(j, 0) << " "
                    << coords(j, 1) << " " << coords(j, 2) << std::endl; 
        } 
    }
    outfile.close(); 
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
