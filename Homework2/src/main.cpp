#include <iostream>
#include <string>
#include <format>   // C++20 format library
#include <chrono>   // used for timing

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

#include "model.hpp"

int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <input.obj> <output.obj> <scale>" << std::endl;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];
    double scale = std::stod(argv[3]);

    if (scale <= 0 || scale > 1) {
        std::cerr << "Scale must be between 0 and 1." << std::endl;
        return 1;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();

    TriMesh mesh;
    if (!IO::read_mesh(mesh, input_file))
    {
        std::cerr << "Error loading mesh" << std::endl;
        return 1;
    }

    std::cout << std::format("Loaded mesh with {} faces, {} edges and {} vertexs.", \
        mesh.n_faces(), mesh.n_edges(), mesh.n_vertices()) << std::endl;

    std::cout << std::format("Simplifying begin with scale: {}.", scale) << std::endl;

    MeshSimplifier simplifier(mesh);
    size_t target_face_count = static_cast<size_t>(mesh.n_faces() * scale);

    simplifier.simplify(target_face_count); 

    if (!IO::write_mesh(mesh, output_file))
    {
        std::cerr << "Error saving mesh" << std::endl;
        return 1;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    std::cout << "Mesh simplified and saved to " << output_file << std::endl;
    std::cout << "Program execution time: " << elapsed_time.count() << " seconds." << std::endl;
    
    return 0;
}