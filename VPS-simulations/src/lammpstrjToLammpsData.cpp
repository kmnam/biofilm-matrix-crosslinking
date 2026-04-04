/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     4/3/2026
 */

#include <fstream>
#include <iomanip>
#include "../include/utils.hpp"
#include "../include/polymerMelt.hpp"

int main(int argc, char** argv)
{
    // Parse the given .json and .lammpstrj files
    std::string json_filename = argv[1];  
    std::string lammpstrj_filename = argv[2]; 
    boost::json::object json_data = parseConfigFile(json_filename).as_object();
    double dt = json_data["dt"].as_double();
    double xmax = json_data["domain_xmax"].as_double(); 
    double xmin = -xmax; 
    double ymax = json_data["domain_ymax"].as_double(); 
    double ymin = -ymax; 
    double zmax = json_data["domain_zmax"].as_double(); 
    double zmin = -zmax; 
    double mass = json_data["monomer_mass"].as_double();  
    auto result = parseMeltLammpstrj<double>(lammpstrj_filename, dt, Units::NANO, 300.0);
    auto ensemble = result.first; 
    std::vector<double> times = result.second;
    
    // Extract only the final configuration
    PolymerMeltConfiguration<double> final_config = ensemble[ensemble.size() - 1]; 

    // Collect potential parameters
    std::unordered_map<std::string, double> lj_params, 
                                            fene_params, 
                                            angle_params,
                                            dihedral_params; 
    const double kT = 1.380649e-2 * 300;    // Use "nano" units 
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

    // Write the final configuration and potential parameters to an output file
    std::string outfilename = argv[3];
    std::string header = ""; 
    final_config.writeLammps(
        outfilename, lj_params, fene_params, angle_mode, angle_params,
        dihedral_params, header, xmin, xmax, ymin, ymax, zmin, zmax, mass
    );  
    
    return 0; 
}
