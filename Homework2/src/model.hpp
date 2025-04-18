#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <Eigen/Dense>
#include <vector>
#include <set>
#include <stdexcept>
#include <boost/heap/fibonacci_heap.hpp>

using namespace OpenMesh;
using TriMesh = TriMesh_ArrayKernelT<>;

class MeshSimplifier
{
public:
    MeshSimplifier(TriMesh &mesh) : mesh_(mesh)
    {
        mesh_.add_property(vertex_Q_, "vertex_Q");
        mesh_.add_property(face_Q_, "face_Q");
        mesh_.add_property(face_normal_, "face_normal");
        mesh_.add_property(best_pos_, "best_pos");

        compute_face_normals();
        init_vertex_quadrics();
        build_priority_queue();
    }

    void simplify(size_t target_face_count)
    {
        std::cout << "Simplifying mesh to " << target_face_count << " faces..." << std::endl;
        std::cout << "It takes some time, please wait..." << std::endl;
        mesh_.garbage_collection();

        mesh_.request_vertex_status();
        mesh_.request_edge_status();
        mesh_.request_face_status();


        size_t current_face_count = mesh_.n_faces();

        while(current_face_count > target_face_count && !priority_queue_.empty())
        {
            EdgeCost ec = pop_min();

            if (!mesh_.is_valid_handle(ec.eh) || mesh_.status(ec.eh).deleted())
                continue;

            TriMesh::HalfedgeHandle heh = mesh_.halfedge_handle(ec.eh, 0); // try the first halfedge
            TriMesh::VertexHandle vh_from = mesh_.from_vertex_handle(heh);
            TriMesh::VertexHandle vh_to = mesh_.to_vertex_handle(heh);

            Eigen::Vector3d best_pos = mesh_.property(best_pos_, ec.eh);
            Eigen::Matrix4d Q_combined = mesh_.property(vertex_Q_, vh_from) + mesh_.property(vertex_Q_, vh_to);

            // store vh_to's position
            Eigen::Vector3d vh_to_pos = to_eigen_vec3(mesh_.point(vh_to));

            TriMesh::Point best_vertex = TriMesh::Point(best_pos[0], best_pos[1], best_pos[2]);

            // Before collapsing, store how many faces are affected for updating current_face_count
            std::set<TriMesh::FaceHandle> affected;
            for (auto vf_it = mesh_.vf_iter(vh_from); vf_it.is_valid(); ++vf_it)
                affected.insert(*vf_it);
            for (auto vf_it = mesh_.vf_iter(vh_to); vf_it.is_valid(); ++vf_it)
                affected.insert(*vf_it);
            std::unordered_map<int, bool> was_deleted;
            for (auto fh : affected)
                was_deleted[fh.idx()] = mesh_.status(fh).deleted();

            mesh_.set_point(vh_to, best_vertex);
            if (!mesh_.is_collapse_ok(heh))
            {
                // If the collapse is not ok, revert the position
                mesh_.set_point(vh_to, TriMesh::Point(vh_to_pos[0], vh_to_pos[1], vh_to_pos[2]));
                // Try the other halfedge
                heh = mesh_.halfedge_handle(ec.eh, 1);
                vh_from = mesh_.from_vertex_handle(heh);
                vh_to = mesh_.to_vertex_handle(heh);

                Eigen::Vector3d vh_to_pos = to_eigen_vec3(mesh_.point(vh_to));
                mesh_.set_point(vh_to, best_vertex);
                if (!mesh_.is_collapse_ok(heh))
                {
                    // If the collapse is not ok, revert the position
                    mesh_.set_point(vh_to, TriMesh::Point(vh_to_pos[0], vh_to_pos[1], vh_to_pos[2]));
                    continue; // skip this edge
                }
                else
                    mesh_.collapse(heh);
            }
            else
                mesh_.collapse(heh);

            // Update current_face_count
            size_t removed = 0;
            for (auto fh : affected)
                if (!was_deleted[fh.idx()] && mesh_.status(fh).deleted())
                    ++removed;
            current_face_count -= removed;

            // Update norm and Q around the affected faces
            for ( TriMesh::VertexFaceIter vf_it = mesh_.vf_iter(vh_to); vf_it.is_valid(); ++vf_it)
            {
                TriMesh::ConstFaceVertexIter fv_it = mesh_.cfv_iter(*vf_it);
                Eigen::Vector3d p0 = to_eigen_vec3(mesh_.point(*fv_it));
                Eigen::Vector3d p1 = to_eigen_vec3(mesh_.point(*(++fv_it)));
                Eigen::Vector3d p2 = to_eigen_vec3(mesh_.point(*(++fv_it)));

                Eigen::Vector3d v1 = p1 - p0;
                Eigen::Vector3d v2 = p2 - p0;
                Eigen::Vector3d normal = v1.cross(v2);

                normal /= normal.norm();
                mesh_.property(face_normal_, *vf_it) = normal;

                double d = -normal.dot(p0);
                Eigen::Vector4d n_d = Eigen::Vector4d(normal[0], normal[1], normal[2], d);

                Eigen::Matrix4d Q = n_d * n_d.transpose();

                mesh_.property(face_Q_, *vf_it) = Q;
            }

            // Update the quadrics for the affected vertices and update the priority queue
            std::unordered_map<int, bool> was_modified;
            for (TriMesh::VertexFaceIter vf_it = mesh_.vf_iter(vh_to); vf_it.is_valid(); ++vf_it)
            {
                for (TriMesh::FaceVertexIter fv_it = mesh_.fv_iter(*vf_it); fv_it.is_valid(); ++fv_it)
                {
                    if (was_modified.find(fv_it->idx()) != was_modified.end())
                        continue;
                    was_modified[fv_it->idx()] = true;
                    if (fv_it->idx() == vh_to.idx())
                    {
                        mesh_.property(vertex_Q_, *fv_it) = Q_combined;
                        continue;   
                    }
                    Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
                    for (TriMesh::VertexFaceIter vf_it2 = mesh_.vf_iter(*fv_it); vf_it2.is_valid(); ++vf_it2)
                    {
                        Q += mesh_.property(face_Q_, *vf_it2);
                    }
                    

                    mesh_.property(vertex_Q_, *fv_it) = Q;
                    for (TriMesh::VertexEdgeIter ve_it = mesh_.ve_iter(*fv_it); ve_it.is_valid(); ++ve_it)
                    {
                        double cost = compute_edge_cost(*ve_it);
                        bool removed = remove_element(*ve_it);  // remove the old element
                        push_element({cost, *ve_it});
                    }
                }
            }
        }
        
        mesh_.garbage_collection();
        mesh_.release_vertex_status();
        mesh_.release_edge_status();
        mesh_.release_face_status();
    }

    
private:
    struct EdgeCost
    {
        double cost;
        TriMesh::EdgeHandle eh;

        bool operator<(const EdgeCost &other) const
        {
            return cost > other.cost; // min-heap
        }
    };

    
    TriMesh &mesh_;
    OpenMesh::FPropHandleT<Eigen::Matrix4d> face_Q_;        // face quadrics
    OpenMesh::VPropHandleT<Eigen::Matrix4d> vertex_Q_;      // vertex quadrics
    OpenMesh::EPropHandleT<Eigen::Vector3d> best_pos_;      // best position for each edge
    OpenMesh::FPropHandleT<Eigen::Vector3d> face_normal_;

    // ############ Modifiable Priority Queue ############ //
    using Heap = boost::heap::fibonacci_heap<EdgeCost>;
    Heap priority_queue_;

    std::unordered_map<TriMesh::EdgeHandle, Heap::handle_type> handles;

    void push_element(EdgeCost ec)
    {
        handles[ec.eh] = priority_queue_.push(ec);
    }

    bool remove_element(TriMesh::EdgeHandle eh)
    {
        if (handles.count(eh))
        {
            priority_queue_.erase(handles[eh]);
            handles.erase(eh);
            return true;
        }
        return false;
    }

    EdgeCost pop_min()
    {
        EdgeCost top = priority_queue_.top();
        priority_queue_.pop();
        handles.erase(top.eh);
        return top;
    }
    // ############ Modifiable Priority Queue ############ //

    // Compute face unit normals using eigen 
    void compute_face_normals()
    {
        for (TriMesh::FaceIter f_it = mesh_.faces_begin(); f_it != mesh_.faces_end(); ++f_it)
        {
            // Get the three vertices of the face
            TriMesh::ConstFaceVertexIter fv_it = mesh_.cfv_iter(*f_it);
            Eigen::Vector3d p0 = to_eigen_vec3(mesh_.point(*fv_it));
            Eigen::Vector3d p1 = to_eigen_vec3(mesh_.point(*(++fv_it)));
            Eigen::Vector3d p2 = to_eigen_vec3(mesh_.point(*(++fv_it)));

            Eigen::Vector3d v1 = p1 - p0;
            Eigen::Vector3d v2 = p2 - p0;
            Eigen::Vector3d normal = v1.cross(v2);

            normal /= normal.norm();  

            mesh_.property(face_normal_, *f_it) = normal;
        }
    }


    /*
        Compute the face quadrics for each face first.
        Then, the quadric for a vertex is the sum of the quadrics of all faces adjacent to it.

        Given the face normals (n), and the face (v_0, v_1, v_2) vertices,
            1. d = - n^T * v0
            2. A_{3x3} = n * n^T
            3. Q_face = [A_{3x3}  n^T * d]
                        [n * d^T  d^2]
    */
    void init_vertex_quadrics()
    {
        // Compute quadrics for each face first
        for (TriMesh::FaceIter f_it = mesh_.faces_begin(); f_it != mesh_.faces_end(); ++f_it)
        {
            Eigen::Vector3d n = mesh_.property(face_normal_, *f_it);
            TriMesh::FaceVertexIter fv_it = mesh_.fv_iter(*f_it);
            Eigen::Vector3d p0 = to_eigen_vec3(mesh_.point(*fv_it));
            Eigen::Vector3d p1 = to_eigen_vec3(mesh_.point(*(++fv_it)));
            Eigen::Vector3d p2 = to_eigen_vec3(mesh_.point(*(++fv_it)));

            double d = -n.dot(p0);

            Eigen::Vector4d n_d = Eigen::Vector4d(n[0], n[1], n[2], d);

            Eigen::Matrix4d Q = n_d * n_d.transpose();

            mesh_.property(face_Q_, *f_it) = Q;
        }

        // Compute quadrics for each vertex (sum of the adjacent faces' quadrics)
        for (TriMesh::VertexIter v_it = mesh_.vertices_begin(); v_it != mesh_.vertices_end(); ++v_it)
        {
            Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
            for (TriMesh::VertexFaceIter vf_it = mesh_.vf_iter(*v_it); vf_it.is_valid(); ++vf_it)
            {
                Q += mesh_.property(face_Q_, *vf_it);
            }
            mesh_.property(vertex_Q_, *v_it) = Q;
        }

        std::cout << "Initialized vertex quadrics successfully." << std::endl;
    }

    /*
        The naive version of the edge cost function.
        Given the two vertices of the edge (v1, v2), and their quadrics (Q1, Q2),
        the cost function is:
            cost = x^T * (Q1 + Q2) * x + 2 * b^T * x + c
        and
            best_pos(x) = - (Q1 + Q2).inverse() * b, if (Q1 + Q2) is invertible
            best_pos(x) = (v1 + v2) / 2, otherwise
        Then upadate the best position for the edge.
    */
    double compute_edge_cost(TriMesh::EdgeHandle eh)
    {
        TriMesh::VertexHandle vh1 = mesh_.to_vertex_handle(mesh_.halfedge_handle(eh, 0));
        TriMesh::VertexHandle vh2 = mesh_.from_vertex_handle(mesh_.halfedge_handle(eh, 0));

        Eigen::Matrix4d Q = mesh_.property(vertex_Q_, vh1) + mesh_.property(vertex_Q_, vh2);

        Eigen::Matrix3d A = Q.block<3, 3>(0, 0);
        Eigen::Vector3d b = Q.block<3, 1>(0, 3);
        double c = Q(3, 3);

        Eigen::Vector3d p1 = to_eigen_vec3(mesh_.point(vh1));
        Eigen::Vector3d p2 = to_eigen_vec3(mesh_.point(vh2));

        // Robust solve
        Eigen::Vector3d x_hat = 0.5 * (p1 + p2);
        Eigen::Vector3d x = robustSolve(A, -b, x_hat);

        // Naive solve
        // Eigen::Vector3d x = solveQuadraticCost(A, b, c, p1, p2);

        // Update the best position for the edge
        mesh_.property(best_pos_, eh) = x;

        return x.transpose() * A * x + 2 * b.dot(x) + c;
    }

    Eigen::Vector3d solveQuadraticCost(const Eigen::Matrix3d &A, const Eigen::Vector3d &b, double c,
                                       const Eigen::Vector3d &p1, const Eigen::Vector3d &p2)
    {
        Eigen::Vector3d x;

        // Check if A is invertible (full rank)
        Eigen::ColPivHouseholderQR<Eigen::Matrix3d> qr(A);
        if (qr.isInvertible())
        {
            x = qr.solve(-b);
        }
        else
        {
            Eigen::Vector3d x_mid = (p1 + p2) * 0.5;

            auto evaluateCost = [&](const Eigen::Vector3d &point)
            {
                return point.transpose() * A * point + 2 * b.dot(point) + c;
            };

            double cost_p1 = evaluateCost(p1);
            double cost_p2 = evaluateCost(p2);
            double cost_mid = evaluateCost(x_mid);

            if (cost_p1 <= cost_p2 && cost_p1 <= cost_mid)
                x = p1;
            else if (cost_p2 <= cost_p1 && cost_p2 <= cost_mid)
                x = p2;
            else
                x = x_mid;
        }

        return x;
    }

    Eigen::Vector3d robustSolve(const Eigen::Matrix3d &A, const Eigen::Vector3d &b, 
                                const Eigen::Vector3d &x_hat, double epsilon = 1e-3)
    {   
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

        Eigen::Vector3d singular_values = svd.singularValues();
        double sigma1 = singular_values(0);

        // Calculate Sigma^+: sigma_i^+ = 1 / sigma_i if sigma_i > epsilon, else 0
        Eigen::Vector3d sigma_plus(singular_values.size());
        for (int i = 0; i < singular_values.size(); ++i)
        {
            if (singular_values(i) / sigma1 > epsilon)
                sigma_plus(i) = 1.0 / singular_values(i);
            else
                sigma_plus(i) = 0.0;
        }

        // x = x_hat + V * Sigma^+ * U^T * (b - A * x_hat)
        Eigen::Matrix3d Sigma_plus = sigma_plus.asDiagonal();
        Eigen::Vector3d residual = b - A * x_hat;
        Eigen::Vector3d x = x_hat + svd.matrixV() * Sigma_plus * svd.matrixU().transpose() * residual;

        return x;
    }

    void build_priority_queue()
    {
        for (TriMesh::EdgeIter e_it = mesh_.edges_begin(); e_it != mesh_.edges_end(); ++e_it)
        {
            double cost = compute_edge_cost(*e_it);
            push_element({cost, *e_it});
        }
        std::cout << "Build priority queue successfully." << std::endl;
    }

    static Eigen::Vector3d to_eigen_vec3(const TriMesh::Point &p)
    {
        return Eigen::Vector3d(p[0], p[1], p[2]);
    }
};

