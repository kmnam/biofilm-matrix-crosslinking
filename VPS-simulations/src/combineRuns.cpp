/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     4/23/2026
 */
#include <iostream>
#include <filesystem>
#include "../include/polymerMelt.hpp"

int main(int argc, char** argv)
{
    std::string prefix = argv[1]; 
    std::filesystem::path path(prefix);
    std::filesystem::path dir = path.parent_path(); 
    std::filesystem::path stem = path.stem();

    // Get the original run output file 
    std::stringstream ss0, ss1;
    ss0 << prefix << "_cbmc_configs.txt";  
    ss1 << (dir / "run1" / stem).string() << "_cbmc_configs.txt";
    std::string filename0 = ss0.str(); 
    std::string filename1 = ss1.str();

    // Make a new combined directory if one does not exist
    if (!std::filesystem::is_directory(dir / "combined"))
        std::filesystem::create_directory(dir / "combined");
    std::stringstream ss2; 
    ss2 << (dir / "combined" / stem).string() << "_cbmc_configs.txt"; 
    std::string outfilename = ss2.str(); 

    // Write a combined configurations file to that directory 
    auto result0 = parseMeltConfigFile<double>(filename0, Units::NANO, 300.0); 
    auto result1 = parseMeltConfigFile<double>(filename1, Units::NANO, 300.0); 
    auto melt_configs0 = result0.first; 
    auto params = result0.second; 
    auto melt_configs1 = result1.first;
    std::cout << "... parsed " << melt_configs0.size() << " configurations from:\n"
              << filename0 << std::endl; 
    std::cout << "... parsed " << melt_configs1.size() << " configurations from:\n"
              << filename1 << std::endl; 

    // Re-parse the first file to get the initial configuration ... 
    std::ifstream infile(filename0);  
    std::string line;
    while (std::getline(infile, line))
    {
        // Parse until we encounter a line that starts with "# CONFIG"
        if (line.find("# CONFIG") == 0)
            break; 
    }

    // Now parse the first configuration, to get the polymer lengths
    const int n_chains = melt_configs0[0].numChains(); 
    std::vector<int> lengths;
    for (int i = 0; i < n_chains; ++i)
        lengths.push_back(melt_configs0[0].getLength(i));
    std::vector<Matrix<double, Dynamic, 3> > init_coords; 
    for (int i = 0; i < n_chains; ++i)
        init_coords.push_back(Matrix<double, Dynamic, 3>::Zero(lengths[i], 3));   
    while (std::getline(infile, line))
    {
        if (line.find("# CONFIG") == 0)   // If we reach the next configuration, break
        {
            break;
        }
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
            double rx = static_cast<double>(std::stod(token)); 
            std::getline(ss, token, '\t');    // y-coordinate
            double ry = static_cast<double>(std::stod(token)); 
            std::getline(ss, token, '\t');    // z-coordinate
            double rz = static_cast<double>(std::stod(token));
            init_coords[polymer_idx](atom_idx, 0) = rx; 
            init_coords[polymer_idx](atom_idx, 1) = ry; 
            init_coords[polymer_idx](atom_idx, 2) = rz;  
        } 
    }
    PolymerMeltConfiguration<double> init_config(
        n_chains, init_coords, Units::NANO, 300.0, params["domain_xmin"], 
        params["domain_xmax"], params["domain_ymin"], params["domain_ymax"], 
        params["domain_zmin"], params["domain_zmax"]  
    );
    std::cout << "... parsed initial configuration from original run\n"; 

    // Parse the .json file to get potential parameters 
    std::string json_filename = argv[2]; 
    boost::json::object json_data = parseConfigFile(json_filename).as_object();
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
    const double neighbor_threshold = 1.1 * pow(2, 1. / 6.) * lj_params["sigma"]; 

    // Calculate the energy of each configuration from the original run
    std::vector<double> energies_total,
                        energies_nonbonded,
                        energies_bond, 
                        energies_angle, 
                        energies_dihedral; 
    for (int i = 0; i < melt_configs0.size(); ++i)
    {
        double energy_nonbonded = melt_configs0[i].getTotalNonbondedEnergy(
            lj_params, neighbor_threshold, true
        );
        double energy_bond = melt_configs0[i].getTotalBondEnergy(
            fene_params, true, lj_params
        ); 
        double energy_angle = melt_configs0[i].getTotalBondAngleEnergy(
            angle_mode, angle_params
        ); 
        double energy_dihedral = melt_configs0[i].getTotalDihedralAngleEnergy(
            dihedral_params
        );
        energies_total.push_back(energy_nonbonded + energy_bond + energy_angle + energy_dihedral);
        energies_nonbonded.push_back(energy_nonbonded); 
        energies_bond.push_back(energy_bond); 
        energies_angle.push_back(energy_angle); 
        energies_dihedral.push_back(energy_dihedral);  
    } 

    // Remove the 0-th configuration in the second run, and collect all other
    // configurations
    for (int i = 1; i < melt_configs1.size(); ++i)
    {
        melt_configs0.push_back(melt_configs1[i]);
        double energy_nonbonded = melt_configs1[i].getTotalNonbondedEnergy(
            lj_params, neighbor_threshold, true
        );
        double energy_bond = melt_configs1[i].getTotalBondEnergy(
            fene_params, true, lj_params
        ); 
        double energy_angle = melt_configs1[i].getTotalBondAngleEnergy(
            angle_mode, angle_params
        ); 
        double energy_dihedral = melt_configs1[i].getTotalDihedralAngleEnergy(
            dihedral_params
        );
        energies_total.push_back(energy_nonbonded + energy_bond + energy_angle + energy_dihedral);
        energies_nonbonded.push_back(energy_nonbonded); 
        energies_bond.push_back(energy_bond); 
        energies_angle.push_back(energy_angle); 
        energies_dihedral.push_back(energy_dihedral);  
    }

    // Write to file
    std::cout << "... writing to file: " << outfilename << std::endl;  
    writeMeltConfigFile<double>(
        melt_configs0, init_config, energies_total, energies_nonbonded,
        energies_bond, energies_angle, energies_dihedral, params, outfilename 
    );  

    return 0; 
}
