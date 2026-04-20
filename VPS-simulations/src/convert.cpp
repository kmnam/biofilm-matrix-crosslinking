/**
 * Inter-convert output files between different formats. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     4/11/2026
 */

#include <fstream>
#include <iomanip>
#include "../include/utils.hpp"
#include "../include/polymerConfiguration.hpp"
#include "../include/polymerEnsemble.hpp"
#include "../include/polymerMelt.hpp"

/**
 * Write a configurations file from a .lammpstrj file. 
 *
 * Each configuration in the input .lammpstrj file is written as a
 * configuration in the output configurations file, in the same order.
 *
 * Since the .lammpstrj file will not contain information that belongs in the
 * header in the configurations file, this function requires a JSON file
 * containing this information.  
 *
 * @param json_filename Path to JSON file containing header information. 
 * @param lammpstrj_filename Path to input .lammpstrj file. 
 * @param outfilename Path to output file. 
 */
void lammpstrjToConfig(const std::string& json_filename, 
                       const std::string& lammpstrj_filename, 
                       const std::string& outfilename)
{
    // Parse the given .json and .lammpstrj files
    boost::json::object json_data = parseConfigFile(json_filename).as_object();
    double dt = json_data["dt"].as_double(); 
    auto result = parseLammpstrj<double>(lammpstrj_filename, dt, Units::NANO, 300.0);
    PolymerEnsemble<double> ensemble = result.first; 
    std::vector<double> times = result.second;

    // Collect potential parameters for energy calculations 
    const double kT = 1.380649e-2 * 300.0;  
    std::unordered_map<std::string, double> lj_params, 
                                            fene_params, 
                                            angle_params,
                                            dihedral_params; 
    lj_params["eps"] = json_data["lj_eps"].as_double() * kT;
    lj_params["sigma"] = json_data["lj_sigma"].as_double(); 
    fene_params["K"] = json_data["fene_K"].as_double() * kT; 
    fene_params["R0"] = json_data["fene_R0"].as_double();
    AngleMode angle_mode = (
        json_data["angle_mode"].as_int64() == 0 ? AngleMode::COSINE : AngleMode::GAUSSIAN
    );
    if (angle_mode == AngleMode::COSINE)
    {
        angle_params["K"] = json_data["cosine_K"].as_double() * kT; 
        angle_params["theta0"] = (
            json_data["cosine_theta0"].as_double() * boost::math::constants::pi<double>() / 180
        );
    }
    else 
    {
        angle_params["A1"] = json_data["gaussian_A1"].as_double(); 
        angle_params["A2"] = json_data["gaussian_A2"].as_double(); 
        angle_params["w1"] = json_data["gaussian_w1"].as_double(); 
        angle_params["w2"] = json_data["gaussian_w2"].as_double(); 
        angle_params["theta1"] = (
            json_data["gaussian_theta1"].as_double() * boost::math::constants::pi<double>() / 180
        ); 
        angle_params["theta2"] = (
            json_data["gaussian_theta2"].as_double() * boost::math::constants::pi<double>() / 180
        );
    } 
    dihedral_params["K"] = json_data["dihedral_K"].as_double() * kT;
    dihedral_params["d"] = 1; 
    dihedral_params["n"] = 1;

    // Write the configurations to an output file
    std::unordered_map<std::string, double> params;    // Leave empty 
    writeConfigFile<double>(
        ensemble, params, lj_params, fene_params, angle_mode, angle_params,
        dihedral_params, outfilename
    );  
}

/**
 * Write a .lammpstrj file from a configurations file.
 *
 * Each configuration in the input configurations file is written as a 
 * configuration in the output .lammpstrj file, in the same order.
 *
 * The timesteps are defined as 0, 1, 2, ...  
 *
 * @param config_filename Path to input configurations file. 
 * @param outfilename Path to output file. 
 */
void configToLammpstrj(const std::string& config_filename,
                       const std::string& outfilename, const double xmin = -500.0,
                       const double xmax = 500.0, const double ymin = -500.0,
                       const double ymax = 500.0, const double zmin = -500.0,
                       const double zmax = 500.0)
{
    // Parse the given configurations file 
    auto result = parseEnsemble<double>(config_filename, Units::NANO, 300.0); 
    PolymerEnsemble<double> ensemble = result.first;
    writeLammpstrj<double>(ensemble, outfilename, xmin, xmax, ymin, ymax, zmin, zmax); 
}

/**
 * Write a .lammpstrj file from a configurations file for a polymer melt.
 *
 * Each configuration in the input configurations file is written as a 
 * configuration in the output .lammpstrj file, in the same order.
 *
 * The timesteps are defined as 0, 1, 2, ...  
 *
 * @param config_filename Path to input melt configurations file. 
 * @param outfilename Path to output file. 
 */
void meltConfigToLammpstrj(const std::string& config_filename,
                           const std::string& outfilename)
{
    // Parse the given configurations file 
    auto result = parseMeltConfigFile<double>(config_filename, Units::NANO, 300.0); 
    auto ensemble = result.first;
    auto bounds = ensemble[0].getBounds();
    writeMeltLammpstrj<double>(
        ensemble, outfilename, std::get<0>(bounds), std::get<1>(bounds), 
        std::get<2>(bounds), std::get<3>(bounds), std::get<4>(bounds), 
        std::get<5>(bounds)
    ); 
}

int main(int argc, char** argv)
{
    // Parse conversion mode 
    const int mode = std::stoi(argv[1]); 

    // Run the desired conversion
    if (mode == 0)
    {
        const std::string json_filename = argv[2]; 
        const std::string infilename = argv[3]; 
        const std::string outfilename = argv[4]; 
        lammpstrjToConfig(json_filename, infilename, outfilename);
    }
    else if (mode == 1)
    {
        const std::string infilename = argv[2]; 
        const std::string outfilename = argv[3]; 
        meltConfigToLammpstrj(infilename, outfilename);
    } 

    return 0; 
}
